#!/bin/bash
#SBATCH --job-name=qTMD
#SBATCH --account=pion3d.lq2_gpu
#SBATCH --partition=lq2_gpu
#SBATCH --nodes=4
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4     # 4 tasks per node
#SBATCH --cpus-per-task=16      # 16 CPU cores per task
#SBATCH --qos=normal
#SBATCH --time=01:00:00
#SBATCH --distribution=cyclic
#SBATCH --output=/lustre2/pion3d/jinchen/debug/PyQUDA_qTMD/example_scripts/LQ2/logs/gpt_lq2.%j.out
#SBATCH --error=/lustre2/pion3d/jinchen/debug/PyQUDA_qTMD/example_scripts/LQ2/logs/gpt_lq2.%j.err


# switch to the submit directory
WORKDIR=/lustre2/pion3d/jinchen/debug/PyQUDA_qTMD/example_scripts/LQ2
cd $WORKDIR

# Enable GPU support for MPI
export MPICH_GPU_SUPPORT_ENABLED=1

# Output node information
echo -e "\n>>> SLURM_JOB_NODELIST content:"
scontrol show hostname $SLURM_JOB_NODELIST
NODES=$SLURM_JOB_NUM_NODES
TASKS=$SLURM_NTASKS
echo -e "${NODES}n*${TASKS}t\n"

# show current time
start_time=$(date +%s)
echo "Start time: $start_time"

# env
source /lustre2/pion3d/jinchen/env/gpt.env

# check python version
python3 --version

# check python path
which python3

# check cupy version
python3 -c "import cupy; print('CuPy version:', cupy.__version__)"
python3 -c "import cupy.cuda; print('CUDA module OK')"

nvidia-smi
nvcc --version

# create QUDA resource directory
QUDA_RPATH=${WORKDIR}/.cache
mkdir -p ${QUDA_RPATH}

#export SLURM_CPU_BIND="cores"
export OMP_NUM_THREADS=32
export QUDA_ENABLE_TUNING=1
export QUDA_RESOURCE_PATH=${QUDA_RPATH}
export QUDA_PROFILE_OUTPUT_BASE=${QUDA_RPATH}/profile_
export QUDA_ENABLE_P2P=0
export QUDA_ENABLE_MPS=1
export QUDA_ENABLE_DEVICE_MEMORY_POOL=0

# run
main=gpt_main.py
echo -e "\n>>> Run Python script ${main}"

srun -N 4 -n 16 --mpi=pmix --gpus-per-task=1 -u \
    python3 ${main} \
    --mpi 2.2.2.2 --grid 64.64.64.64 \
    --shm-mpi 1 --shm 2048 \
    --comms-sequential \
    --accelerator-threads 16 \
    --device-mem 26000 --comms-overlap --comms-concurent

# calculate total time
end_time=$(date +%s)
total_time=$(echo "$end_time - $start_time" | bc)
hours=$(($total_time / 3600))
minutes=$(($total_time % 3600 / 60))
seconds=$(($total_time % 60))

echo -e "\n>>> Total runtime: ${hours}:${minutes}:${seconds}"