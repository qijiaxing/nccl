# NCCL EP HT `dispatch_kernel` Compute and Performance Analysis

This report analyzes `dispatch_kernel` in `contrib/nccl_ep/device/hybrid_ep.cuh`, with a focus on the single-node, multi-GPU case where experts are distributed across GPUs inside one LSA/NVLink domain. In this case `NUM_LSA_TEAMS == 1`, so the inter-node RDMA path is compiled out and the kernel is dominated by intra-node token staging, TMA copies, routing metadata, and synchronization.

## 1. Where the Kernel Fits

HT dispatch is launched through:

- `contrib/nccl_ep/device/hybridep_adapter.cu`: `call_dispatch()` resolves token dtype, number of LSA teams, and LSA team size.
- `contrib/nccl_ep/device/hybrid_ep.cuh`: `hybrid_ep::dispatch()` computes dynamic shared memory and launches `dispatch_kernel`.
- `contrib/nccl_ep/device/hybridep_configs.cuh`: dispatch constants are currently:
  - `HYBRIDEP_DISPATCH_NUM_OF_STAGES = 12`
  - `HYBRIDEP_DISPATCH_NUM_OF_IN_FLIGHT_S2G = 4`
  - `HYBRIDEP_DISPATCH_NUM_OF_BLOCKS = 16`
  - `HYBRIDEP_DISPATCH_NUM_OF_PIPELINES_PER_BLOCK = 2`
  - `HYBRIDEP_DISPATCH_N2N_WARPS = 2`, used only for multi-node
- `contrib/nccl_ep/include/common.hpp`:
  - `HT_OF_NUM_TOKENS_PER_CHUNK = 64`
  - `MAX_SUPPORTED_TOKENS_PER_RANK = 8192`

For a single node, `NUM_LSA_TEAMS == 1`, so there is no N2N/RDMA warp group. The active dispatch block shape is:

| Layout | Warps per CTA | Threads per CTA | Active groups |
|---|---:|---:|---|
| Flat | 4 | 128 | 2 G2S warps + 2 S2G warps |
| Expert-major | 5 | 160 | 2 G2S warps + 2 S2G warps + 1 PAD warp |

The launch uses `__launch_bounds__(..., 1)`, so the kernel is intentionally designed around at most one resident CTA per SM. That makes shared-memory size a fit/no-fit constraint first, and an occupancy knob only second.

## 2. High-Level Single-Node Dataflow

In single-node dispatch, each source GPU owns `num_tokens_per_rank` attention tokens. Metadata preprocessing has already produced:

- `rdma_to_attn_map`: for each source node/rank view, whether a token is needed by the local node. In single-node mode this is effectively the local routing mask.
- `sparse_to_dense_map`: maps each input token to output slots on destination ranks. In flat layout, the inner dimension is `num_ranks_per_node`. In expert-major layout, it is `top_k` and each entry packs `(rank, slot)`.
- Optional expert-major counts and offsets for per-expert output zones and padding.

`dispatch_kernel` then runs one CTA per configured block/SM. Each CTA processes a strided subset of token chunks:

```text
chunk i = blockIdx.x, blockIdx.x + NUM_OF_BLOCKS, ...
chunk size = HT_OF_NUM_TOKENS_PER_CHUNK = 64 tokens
```

Within each CTA, the two pipelines split those chunks by `chunk_iter % NUM_PIPELINES`. With the current constants, each CTA has two independent producer/consumer pipelines.

## 3. Warp Group Responsibilities

### 3.1 G2S: Global to Shared

`G2S_warp_group_device_function` is the producer side. In single-node mode it:

1. Iterates over this CTA's assigned chunks.
2. Uses only chunks assigned to the warp's pipeline rank.
3. Reads the routing bitmap for the chunk.
4. For every routed token, issues `cp_async_bulk` from global input token/prob/scale buffers into that pipeline's shared-memory stage.
5. Uses `mbarrier_arrive_expect_tx` so S2G can wait until the async copy into shared memory has completed.

Important behavior:

- Only the elected lane issues the TMA load sequence, so this group is mostly a control/TMA producer, not a SIMD math unit.
- The number of staged tokens depends on routing density, not merely on `num_tokens_per_rank`.
- In forward dispatch, probability data is also staged.
- In FP8 dispatch, scaling factors are also staged.

### 3.2 S2G: Shared to Global

`S2G_warp_group_device_function` is the consumer/scatter side. It:

1. Prefetches the current chunk's `sparse_to_dense_map` into a per-pipeline two-stage ping-pong shared-memory buffer.
2. Walks the same routing bitmap as G2S.
3. Waits for the matching token stage's producer mbarrier.
4. For each valid output entry in the S2D row, issues `cp_async_bulk` from shared memory to the destination rank's expert output buffer.
5. Tracks `NUM_OF_IN_FLIGHT_S2G` outstanding global stores before waiting and releasing shared-memory stages back to G2S.

Important behavior:

- S2G lanes are used across the S2D inner dimension. Flat layout with 8 ranks uses only lanes 0-7 for output issue; expert-major with top-k 8 also uses only lanes 0-7. If `s2d_inner_dim` grows, more lanes can issue concurrent output copies.
- Each routed token can generate multiple destination writes, depending on top-k/routing.
- `NUM_OF_IN_FLIGHT_S2G` hides global-store latency but must be smaller than `STAGES_PER_PIPELINE`.

### 3.3 PAD Warp, Expert-Major Only

For `NCCL_EP_LAYOUT_EXPERT_MAJOR`, one extra warp runs `PAD_warp_group_device_function`. It zeroes one token row in shared memory and TMA-copies that row into expert padding slots. This is independent of the main G2S/S2G path because padding rows are beyond actual token rows in each expert zone.

This costs:

- One additional warp per CTA.
- One extra shared-memory token-sized TMA slot.
- Extra global writes proportional to padding, especially when many experts have small or zero token counts.

## 4. Shared-Memory Layout and 100 KB Fit

For single-node dispatch, the dynamic shared memory is approximately:

```text
token_fifo =
  NUM_OF_STAGES * align128(hidden_dim * sizeof(token))

s2d_pingpong =
  2 * NUM_PIPELINES * align128(NUM_OF_TOKENS_PER_CHUNK * s2d_inner_dim * sizeof(int32))

prob_fifo, forward only =
  NUM_OF_STAGES * align16(experts_per_rank * num_ranks_per_node * sizeof(float))

scale_fifo, FP8 only =
  NUM_OF_STAGES * align16((hidden_dim / 128) * sizeof(float))

mbarriers =
  NUM_OF_STAGES * 2 * sizeof(uint64)
  + 2 * NUM_PIPELINES * sizeof(uint64)
  + NUM_PIPELINES * sizeof(uint64)

pad_slot, expert-major only =
  align128(hidden_dim * sizeof(token))
```

The dominant term is the token FIFO. With BF16 and `hidden_dim = 7168`, each token stage is:

```text
align128(7168 * 2) = 14,336 bytes
12 stages -> 172,032 bytes
```

That alone exceeds a 100 KB shared-memory budget. Therefore the current BF16 dispatch configuration cannot fit on a GPU limited to 100 KB shared memory per SM for this hidden size, even before S2D/probability buffers are counted.

### Example: `hidden_dim=7168`, `num_ranks_per_node=8`, `experts_per_rank=32`, `top_k=8`

These numbers match the common 8-GPU, 256-expert style configuration in the README. The table assumes forward dispatch and `NUM_PIPELINES=2`.

| Token dtype / layout | Stages | Estimated SMEM | Fits 100 KiB? | Current `in_flight=4` valid? |
|---|---:|---:|---|---|
| BF16 flat | 12 | 192,768 B | No | Yes |
| BF16 flat | 6 | 100,608 B | Yes | No, must reduce to <= 2 |
| BF16 expert-major | 12 | 207,104 B | No | Yes |
| BF16 expert-major | 4 | 84,096 B | Yes | No, must reduce to 1 |
| FP8 flat | 12 | 109,440 B | No | Yes |
| FP8 flat | 10 | 92,672 B | Yes | Yes |
| FP8 expert-major | 12 | 116,608 B | No | Yes |
| FP8 expert-major | 10 | 99,840 B | Yes | Yes |

The exact value can differ with `s2d_inner_dim`, `experts_per_rank`, `num_ranks_per_node`, forward/backward mode, and layout, but the conclusion is stable: current `NUM_OF_STAGES=12` is the main obstacle for 100 KB, especially for BF16.

## 5. Parameter Effects on Performance

### `hidden_dim`

This is the strongest parameter. It scales:

- Shared-memory token FIFO linearly.
- G2S input TMA bytes per routed token.
- S2G output TMA bytes per destination write.
- FP8 scale bytes as `hidden_dim / 128`.

Large hidden dimensions quickly make `NUM_OF_STAGES=12` untenable on 100 KB SMEM. Reducing stages lowers memory footprint linearly, but also reduces buffering against global/NVLink latency.

### Token dtype: BF16 vs FP8

BF16 uses 2 bytes per hidden element; FP8 uses 1 byte plus scaling factors. FP8 roughly halves the token FIFO and payload bandwidth, so it is much easier to fit under 100 KB. For `hidden_dim=7168`, FP8 can likely use 10 stages under 100 KB, while BF16 needs far fewer stages.

### `NUM_OF_STAGES`

This controls the shared-memory FIFO depth per CTA. It must be divisible by `NUM_PIPELINES`.

More stages:

- Better producer/consumer decoupling.
- More tolerance for S2G stalls and TMA latency.
- Higher SMEM use.

Fewer stages:

- Required for 100 KB on BF16.
- Increases the chance G2S stalls waiting for S2G to release a stage.
- Forces `NUM_OF_IN_FLIGHT_S2G` down because `NUM_OF_IN_FLIGHT_S2G < NUM_OF_STAGES / NUM_PIPELINES`.

For BF16 `hidden_dim=7168`, practical 100 KB candidates are likely:

- Flat layout: `NUM_OF_STAGES=6`, `NUM_OF_IN_FLIGHT_S2G=2`.
- Expert-major layout: `NUM_OF_STAGES=4`, `NUM_OF_IN_FLIGHT_S2G=1`.

Both reduce buffering substantially relative to the current 12-stage design.

### `NUM_PIPELINES`

Currently there are two G2S warps and two S2G warps per CTA. Pipelines split chunks by `chunk_iter % NUM_PIPELINES`.

Increasing pipelines:

- Adds G2S/S2G warps and potentially improves chunk-level parallelism.
- Reduces `STAGES_PER_PIPELINE` if total stages are fixed.
- Increases S2D ping-pong shared memory as `2 * NUM_PIPELINES`.
- Can make the `NUM_OF_IN_FLIGHT_S2G < STAGES_PER_PIPELINE` constraint harder to satisfy.

On a 100 KB SMEM GPU, increasing pipelines is unlikely to be the first tuning lever. Keeping `NUM_PIPELINES=2` is conservative.

### `NUM_OF_IN_FLIGHT_S2G`

This is the max S2G store backlog before S2G waits and releases an old stage. It affects latency hiding, not the shared-memory formula directly.

Higher values:

- Hide output-copy latency better.
- Delay stage release to G2S.
- Require more stages per pipeline.

The current value 4 requires `STAGES_PER_PIPELINE >= 5`, so with `NUM_PIPELINES=2`, total stages must be at least 10. This works for FP8 100 KB configurations, but not for BF16 100 KB configurations at `hidden_dim=7168`.

### `NUM_OF_TOKENS_PER_CHUNK`

Currently 64. This affects:

- Number of loop iterations: smaller chunks mean more chunks and more metadata/TMA setup overhead.
- S2D shared memory: `chunk * s2d_inner_dim * 4`, ping-ponged twice per pipeline.
- Routing-map load granularity and tail handling.

For single-node BF16 with large hidden size, chunk size is not the main shared-memory problem. Token FIFO stages dominate. Reducing chunk size can save S2D memory, but it will not solve a 12-stage BF16 fit issue.

### `NUM_OF_BLOCKS`

Currently 16 and intended as one CTA per SM up to 16 SMs per rank. It affects:

- How chunks are striped across CTAs.
- Grid-wide tail synchronization cost.
- PAD warp striping across padding rows.

It does not materially change per-CTA shared memory. On a GPU with more than 16 SMs, this kernel will only use 16 CTAs per rank, so not all SMs are necessarily occupied. On a GPU with 100 KB shared memory per SM, the more immediate issue is whether each CTA fits.

### `num_ranks_per_node`

For single-node dispatch, this equals the LSA team size, usually the number of GPUs in the node.

It affects:

- S2D inner dimension in flat layout.
- Probability buffer size through `experts_per_rank * num_ranks_per_node`.
- Number of destination rank pointers and potential output writes.

Going from 8 to 16 or 32 ranks per node increases metadata/probability costs and may improve S2G lane utilization, but also increases scattered writes and pressure on destination buffers.

### `experts_per_rank`

This mostly affects forward probability staging:

```text
prob stage bytes = align16(experts_per_rank * num_ranks_per_node * 4)
```

It also affects expert-major padding behavior and output-slot distribution. It is usually secondary to `hidden_dim` for SMEM, but can matter when many experts per rank are used.

### `s2d_inner_dim`

Flat layout uses `num_ranks_per_node`; expert-major uses `top_k`.

It affects:

- S2D ping-pong SMEM.
- Number of S2G lanes that can issue output copies.
- Number of possible destination writes checked per routed token.

Small values like 8 leave most lanes idle during S2G issue. Larger values improve lane utilization but increase metadata bandwidth and shared-memory footprint.

### Layout: Flat vs Expert-Major

Flat layout:

- No PAD warp.
- No pad TMA slot.
- S2D inner dimension is rank count.
- Lower SMEM and fewer side writes.

Expert-major layout:

- Adds one PAD warp and a token-sized zero slot.
- S2D entries pack `(rank, slot)`.
- Supports per-expert aligned zones.
- Can improve downstream expert locality but costs more inside dispatch.

For a 100 KB GPU, flat is easier to fit and should be preferred unless expert-major output layout is required by the consumer.

### Forward vs Backward Dispatch

Forward dispatch stages probability data. Backward dispatch does not, so it uses less shared memory and less G2S/S2G bandwidth. For BF16 `hidden_dim=7168`, removing probability data helps but does not make the current 12-stage FIFO fit under 100 KB.

## 6. Expected Bottlenecks on a 100 KB Shared-Memory GPU

1. **Fit failure with current constants**: BF16 and many FP8 layouts exceed 100 KB with 12 stages.
2. **Lower FIFO depth after tuning**: Reducing stages makes G2S/S2G coupling tighter and can expose TMA/global-store latency.
3. **S2G scatter pressure**: The output side performs many small-to-medium TMA stores into per-rank expert output buffers. Routing imbalance can make some CTAs or destination ranks slower.
4. **Low lane utilization for small `s2d_inner_dim`**: With 8 ranks or top-k 8, only 8 lanes issue output copies per token.
5. **Expert-major padding overhead**: Padding is concurrent, but it still consumes one warp, one token-sized SMEM slot, and global bandwidth.
6. **Limited grid size**: `NUM_OF_BLOCKS=16` caps dispatch CTAs. This is consistent with the existing tuning target but may underuse larger GPUs.

## 7. Recommendations for a 100 KB SMEM Target

For single-node BF16 with `hidden_dim` around 7168:

- Do not use the current `NUM_OF_STAGES=12`; it cannot fit.
- Start with flat layout if possible.
- Try `NUM_OF_STAGES=6`, `NUM_PIPELINES=2`, `NUM_OF_IN_FLIGHT_S2G=2` for flat layout.
- For expert-major, try `NUM_OF_STAGES=4`, `NUM_PIPELINES=2`, `NUM_OF_IN_FLIGHT_S2G=1`.
- Expect lower latency hiding than the current 12-stage version; benchmark routing-heavy and routing-light cases separately.

For single-node FP8 with `hidden_dim` around 7168:

- `NUM_OF_STAGES=10`, `NUM_PIPELINES=2`, `NUM_OF_IN_FLIGHT_S2G=4` is a plausible first 100 KB configuration.
- Both flat and expert-major can fit in the example calculation, though expert-major is close to the limit.

For all configurations:

- Keep `NUM_OF_STAGES % NUM_PIPELINES == 0`.
- Keep `NUM_OF_IN_FLIGHT_S2G < NUM_OF_STAGES / NUM_PIPELINES`.
- Recompute SMEM after changing `hidden_dim`, dtype, layout, top-k, ranks per node, or experts per rank.
- If the target GPU's "100 KB" is a decimal limit rather than 100 KiB, leave extra margin; expert-major FP8 at 10 stages is close.
- Verify with `cudaFuncSetAttribute(... cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_SIZE)` and the launch-time `SMEM_SIZE` calculated by `calculate_dispatch_smem_layout_size`.

## 8. Practical Tuning Order

1. Pick layout. Use flat unless expert-major is required.
2. Pick dtype. FP8 is much easier to fit under 100 KB.
3. Choose the largest stage count that fits.
4. Set `NUM_OF_IN_FLIGHT_S2G` to the largest value satisfying the static assert.
5. Benchmark with representative routing distributions, because performance depends on routed-token density and destination skew.
6. Only then consider changing chunk size or block count.

The main performance tradeoff is straightforward: on a 100 KB SMEM GPU, dispatch must give up shared-memory FIFO depth, and that reduces the ability of the G2S producer and S2G scatter consumer to hide each other's stalls. The current kernel is tuned for a larger shared-memory budget; fitting BF16 large-hidden dispatch under 100 KB requires materially shallower pipelines.
