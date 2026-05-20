#!/bin/bash
# MODEL_MOUNT="/root/jqi:/jqi/"
# IMG=nvcr.io/nvidia/pytorch:26.03-py3

WORKDIR=/root/jqi/work/nccl-jqi
export CUDA_HOME=/usr/local/cuda
export MPI_HOME=/usr/mpi/gcc/openmpi-4.1.9a1
export NCCL_HOME=${WORKDIR}/build
export NVCC_GENCODE="-gencode=arch=compute_120,code=sm_120"

export PATH="$CUDA_HOME/bin:$MPI_HOME/bin:$PATH"
export LD_LIBRARY_PATH="${NCCL_HOME}/lib:${MPI_HOME}/lib:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

unset NCCL_DEBUG NCCL_DEBUG_SUBSYS NCCL_DEBUG_FILE NCCL_EP_JIT_LOG NCCL_EP_TRACE_ITERS

export HIDDEN=4096
export EXPERT=256
export GPUS=4
# export CUDA_VISIBLE_DEVICES="0,2,4,6"
export CUDA_VISIBLE_DEVICES="0,1,2,3"
export NCCL_DEBUG=WARN

EXE="cd ${WORKDIR} ; pwd ;"

read -p "Enter task (build, bench, check): " TASK
echo TASK: ${TASK}

# Build NCCL-EP
if [ $TASK == "build" ]; then
  cd ${WORKDIR} ; pwd
  make src.build BUILDDIR=${NCCL_HOME} CUDA_HOME=${CUDA_HOME} NVCC_GENCODE=${NVCC_GENCODE} -j
  make -C contrib/nccl_ep MPI=1 NVCC_GENCODE=${NVCC_GENCODE} -j
  exit 0
fi


# Run Benchmark
if [ $TASK == "bench" ]; then
  export NCCL_EP_JIT_CACHE_DIR="${NCCL_HOME}/.jit-cache/nccl_ep_sm120"
  mkdir -p "$NCCL_EP_JIT_CACHE_DIR"

  read -p "NIC Only (1 for yes, 0 for no): " NIC
  NIC_ONLY=""
  if [ ${NIC} == "1" ]; then
  	export NIC_ONLY="-x NCCL_LSA_TEAM_SIZE=2 -x NCCL_P2P_DISABLE=1 -x NCCL_SHM_DISABLE=1 -x NCCL_IB_HCA=mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1"
	echo Use NIC only, setting NIC_ONLY=${NIC_ONLY}
  fi

  read -p "Algorithm (1 for HT, others for LL): " INPUT
  AL=ll
  TOKENS=128
  if [ ${INPUT} == "1" ]; then
	  AL=ht
	  TOKENS=4096
  fi

  mpirun --allow-run-as-root -np ${GPUS}  \
	 -x PATH -x LD_LIBRARY_PATH -x CUDA_VISIBLE_DEVICES \
         -x RDMAV_FORK_SAFE=1 ${NIC_ONLY} \
	 -x NCCL_DEBUG \
	 -x NCCL_GIN_TYPE=3 -x NCCL_IB_GID_INDEX=3 \
	 -x NCCL_CUMEM_ENABLE=1 -x NCCL_WIN_ENABLE=1 \
         -x NCCL_EP_JIT_CACHE_DIR \
	 -x OMPI_ALLOW_RUN_AS_ROOT=1 -x OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 \
	 ${NCCL_HOME}/test/nccl_ep/ep_bench \
	--algorithm ${AL} --tokens ${TOKENS} --hidden ${HIDDEN} --validate --use-fp8 --top-k 6 --experts 256 ${DISALBE_NVLINK} --warmup 10 --iters 50
fi
