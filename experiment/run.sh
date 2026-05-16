#!/bin/bash
# MODEL_MOUNT="/root/jqi:/jqi/"
# IMG=nvcr.io/nvidia/pytorch:26.03-py3

WORKDIR=/root/jqi/work/nccl
export CUDA_HOME=/usr/local/cuda
export MPI_HOME=/usr/mpi/gcc/openmpi-4.1.9a1
export NCCL_HOME=${WORKDIR}/build
export NVCC_GENCODE="-gencode=arch=compute_120,code=sm_120"

export PATH="$CUDA_HOME/bin:$MPI_HOME/bin:$PATH"
export LD_LIBRARY_PATH="${NCCL_HOME}/lib:${MPI_HOME}/lib:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

unset NCCL_DEBUG NCCL_DEBUG_SUBSYS NCCL_DEBUG_FILE NCCL_EP_JIT_LOG NCCL_EP_TRACE_ITERS

export HIDDEN=4096
export EXPERT=256

EXE="cd ${WORKDIR} ; pwd ;"

read -p "Enter task (build, bench, check): " TASK
echo TASK: ${TASK}

# Build NCCL-EP
if [ $TASK == "build" ]; then
  EXE+="make src.build BUILDDIR=${NCCL_HOME} CUDA_HOME=${CUDA_HOME} NVCC_GENCODE=${NVCC_GENCODE} -j ;"
  EXE+="make -C contrib/nccl_ep MPI=1 NVCC_GENCODE=${NVCC_GENCODE} -j ;"
  bash -c "${EXE}"
  exit 0
fi


# DISALBE_NVLINK="--disable-nvlink"
#       --profile \
#       --user-handle-mem  \
#        -x NCCL_LSA_TEAM_SIZE \
#        -x NCCL_P2P_DISABLE -x NCCL_SHM_DISABLE \
#        -x NCCL_IB_HCA \
#        -x NCCL_IB_GID_INDEX \
#        -x NCCL_GIN_TYPE \
#        -x NCCL_CUMEM_ENABLE=1 -x NCCL_WIN_ENABLE=1 \
if [ $TASK == "bench" ]; then
  NCCL_EP_JIT_CACHE_DIR="${NCCL_HOME}/.jit-cache/nccl_ep_ll_sm120"
  mkdir -p "$NCCL_EP_JIT_CACHE_DIR"
  mpirun --allow-run-as-root -np 4  \
	 -x PATH -x LD_LIBRARY_PATH -x CUDA_VISIBLE_DEVICES=0,1,2,3 \
         -x RDMAV_FORK_SAFE=1 \
         -x NCCL_EP_JIT_CACHE_DIR \
	 -x OMPI_ALLOW_RUN_AS_ROOT=1 -x OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 \
	 ${NCCL_HOME}/test/nccl_ep/ep_bench \
	--algorithm ll \
	--layout em \
	--tokens 128 \
	--hidden ${HIDDEN} \
	--top-k 6 \
	--experts 256 ${DISALBE_NVLINK} \
	--warmup 10 -- iters 50 \
	--use-fp8 \
        --validate
fi


#   -e CUDA_VISIBLE_DEVICES="4,5,6,7" \
# docker run \
#     -it \
#     --rm \
#     --gpus all \
#     --device=/dev/infiniband \
#     --device=/dev/gdrdrv:/dev/gdrdrv \
#     --ipc host \
#     --network host \
#     --name nccl-ep \
#     --shm-size 32G \
#     --ulimit memlock=-1 \
#     -e CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
#     -e NCCL_IB_HCA=mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1 \
#     -e NCCL_IB_GID_INDEX=-1 \
#     -e NCCL_GIN_TYPE=3 \
#     -e NCCL_LSA_TEAM_SIZE=1 \
#     -e NCCL_P2P_DISABLE=1 \
#     -e NCCL_SHM_DISABLE=1 \
#     -e NCCL_CUMEM_ENABLE=1 \
#     -e NCCL_WIN_ENABLE=1 \
#     -e RDMAV_FORK_SAFE=1 \
#     -e MPI_HOME=/usr/local/mpi \
#     -e LD_LIBRARY_PATH="${CUDA_HOME}/lib:${CUDA_HOME}/lib64:${CUDA_HOME}/extras/CUPTI/lib64:${NCCL_HOME}/lib:$LD_LIBRARY_PATH" \
#     -e PATH="${CUDA_HOME}/bin:${NCCL_HOME}/bin:${MPI_HOME}/bin:$PATH" \
#     -e OMPI_ALLOW_RUN_AS_ROOT=1 \
#     -e OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 \
#     -v ${MODEL_MOUNT} \
#     ${IMG} \
#     bash -c "${EXE}"
