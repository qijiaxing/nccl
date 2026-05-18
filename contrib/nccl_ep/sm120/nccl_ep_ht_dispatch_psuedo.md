# NCCL EP HT Dispatch psuedo code

```c++
constexpr int NUM_PIPELINES = HYBRIDEP_DISPATCH_NUM_OF_PIPELINES_PER_BLOCK; // 2
constexpr int NUM_STAGES = HYBRIDEP_DISPATCH_NUM_OF_STAGES;                 // 12
constexpr int STAGES_PER_PIPELINE = NUM_STAGES / NUM_PIPELINES;             // 6
constexpr int IN_FLIGHT_S2G = HYBRIDEP_DISPATCH_NUM_OF_IN_FLIGHT_S2G;        // 4

// CTA layout, flat + single-node:
//   G2S warps: pipeline 0, pipeline 1
//   S2G warps: pipeline 0, pipeline 1

void dispatch_cta(int block_id) {
  parallel_warp_groups {
    G2S_warp(pipeline_id);
    S2G_warp(pipeline_id);
  }
}

void G2S_warp(int pipeline_id) {
  int stage = 0;
  int tokens_produced = 0;
  int consumer_parity = 1;

  for (int chunk = block_id; chunk < num_chunks; chunk += NUM_BLOCKS) {
    if ((chunk_iter++ % NUM_PIPELINES) != pipeline_id)
      continue;

    for (int token : tokens_in_chunk(chunk)) {
      if (!token_needed_by_this_node(token))
        continue;

      // Ring is full: wait until S2G releases this stage.
      if (tokens_produced >= STAGES_PER_PIPELINE) {
        wait(consumer_mbarrier[pipeline_id][stage], consumer_parity);
      }

      // Global -> shared.
      cp_async_global_to_shared(
        smem_token[pipeline_id][stage],
        global_input_token[token]);

      // Tell S2G this stage is ready.
      arrive_expect_tx(producer_mbarrier[pipeline_id][stage]);

      tokens_produced++;

      stage++;
      if (stage == STAGES_PER_PIPELINE) {
        stage = 0;
        consumer_parity ^= 1;
      }
    }
  }
}

void S2G_warp(int pipeline_id) {
  int stage = 0;
  int producer_parity = 0;
  int in_flight_s2g = 0;

  for (int chunk = block_id; chunk < num_chunks; chunk += NUM_BLOCKS) {
    if ((chunk_iter++ % NUM_PIPELINES) != pipeline_id)
      continue;

    // Prefetch token -> output slot metadata for this chunk.
    cp_async_global_to_shared(
      smem_s2d_map[pipeline_id][pingpong],
      global_sparse_to_dense_map[chunk]);

    wait(s2d_map_ready[pipeline_id][pingpong]);

    for (int token : tokens_in_chunk(chunk)) {
      if (!token_needed_by_this_node(token))
        continue;

      // Wait until G2S has filled this token stage.
      wait(producer_mbarrier[pipeline_id][stage], producer_parity);

      for (int rank = lane_id; rank < num_ranks_per_node; rank += 32) {
        int output_slot = smem_s2d_map[pipeline_id][pingpong][token][rank];

        if (output_slot != -1) {
          // Shared -> global destination expert output.
          cp_async_shared_to_global(
            expert_output[rank][output_slot],
            smem_token[pipeline_id][stage]);
        }
      }

      commit_s2g_copy_group();
      in_flight_s2g++;

      // Too many outstanding shared->global copies:
      // wait for older copies, then release that stage back to G2S.
      if (in_flight_s2g > IN_FLIGHT_S2G) {
        wait_s2g_copies_keep_last(IN_FLIGHT_S2G);

        int released_stage = stage - IN_FLIGHT_S2G;
        if (released_stage < 0)
          released_stage += STAGES_PER_PIPELINE;

        arrive(consumer_mbarrier[pipeline_id][released_stage]);
        in_flight_s2g--;
      }

      stage++;
      if (stage == STAGES_PER_PIPELINE) {
        stage = 0;
        producer_parity ^= 1;
      }
    }

    pingpong ^= 1;
  }

  // Drain all remaining S2G copies before kernel tail sync.
  wait_all_s2g_copies();
}
```
