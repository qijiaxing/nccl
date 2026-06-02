# NCCL EP patch for RTX 5k/6kd

To run NCCL-EP on RTX 5k/6kd GPUs, one needs to make a few changes to reduce nccl-ep's kernel shared memory requirement.
For this, we have prepared a patch file.
This patch makes a few changes in `contrib/nccl_ep/device/low_latency.cu`:

| Parameter              | Vanilla -> tuned |
|------------------------|------------------|
| kNumSendUnrolls        | conditional 4/2 -> fixed 2 |
| kCombineMaxUnrolls     | 4 -> 2 |
| kNumStages (combine)   | 3 -> 2 (3 sites + host sizing) |
| EP\_STATIC\_ASSERT     | kNumStages == 3 -> >= 2 |

## Apply patch

```bash
  cd /path/to/nccl
  git apply contrib/nccl_ep/sm120/nccl_ep_sm120_ll_low_latency.patch
```

## Build NCCL and NCCL-EP
```bash
  export NVCC_GENCODE="-gencode=arch=compute_120,code=sm_120"
  export CUDA_HOME=/usr/local/cuda
  export MPI_HOME=/usr/mpi/gcc/openmpi-4.1.9a1/

  cd /path/to/nccl
  make -C . src.build BUILDDIR=$PWD/build -j
  make -C contrib/nccl_ep lib ep_bench MPI=1 BUILDDIR=$PWD/build -j
```

## Validate NCCL-EP

Once build up, NCCL and NCCL-EP's lib can be at `/path/to/nccl/build/lib`.

To validate, one can run this nccl-ep benchmark with 4 GPUs:
```
  export GPUS=4
  export NCCL_HOME=/path/to/nccl/build/
  export CUDA_HOME=/usr/local/cuda
  export MPI_HOME=/usr/mpi/gcc/openmpi-4.1.9a1/
  export LD_LIBRARY_PATH="${NCCL_HOME}/lib:${MPI_HOME}/lib:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
  export CUDA_VISIBLE_DEVICES=0,1,2,3
  export NCCL_EP_JIT_CACHE_DIR=".jit-cache/nccl_ep_jit_sm120_lsa1"
  USE_NIC="-x NCCL_LSA_TEAM_SIZE=1 -x NCCL_P2P_DISABLE=1 -x NCCL_SHM_DISABLE=1 \
     -x NCCL_IB_HCA=mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1 \
     -x NCCL_IB_GID_INDEX=-1 -x NCCL_GIN_TYPE=3"

  export AL=ll
  export HIDDEN=4096
  export EXPERTS=256
  export AL=ll
  export LAYOUT=rm   # rm for Rank-major, em for Expert-major
  export TOKENS=4096
  export NUM_SMS=32
  
  mpirun --allow-run-as-root -np ${GPUS} \
	-x LD_LIBRARY_PATH -x CUDA_VISIBLE_DEVICES \
        -x RDMAV_FORK_SAFE=1 ${USE_NIC} \
        -x NCCL_CUMEM_ENABLE=1 -x NCCL_WIN_ENABLE=1 \
        -x NCCL_EP_JIT_CACHE_DIR \
	${NCCL_HOME}/test/nccl_ep/ep_bench \
	--algorithm ${AL} \
	--layout ${LAYOUT} \
	--tokens ${TOKENS} \
	--hidden ${HIDDEN} \
	--max-num-sms ${NUM_SMS} \
	--validate \
	--top-k 6 \
	--experts ${EXPERTS} \
	--warmup 10 --iters 50
```
