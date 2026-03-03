#!/bin/bash -l
#SBATCH -J qTMD                              # Job name
#SBATCH -A NPH174                            # Project account
#SBATCH -N 4                                 # Number of nodes
#SBATCH -n 32                                # Number of tasks/processes
#SBATCH -p batch
#SBATCH -t 0:30:00                          # Walltime
#SBATCH -o /lustre/orion/nph158/proj-shared/jinchen/debug/PyQUDA_qTMD/example_scripts/Frontier/logs/pycontract.%j.out  
#SBATCH -e /lustre/orion/nph158/proj-shared/jinchen/debug/PyQUDA_qTMD/example_scripts/Frontier/logs/pycontract.%j.err

# switch to the submit directory
WORKDIR=/lustre/orion/nph158/proj-shared/jinchen/debug/PyQUDA_qTMD/example_scripts/Frontier
cd $WORKDIR

# output node info
echo ' '
echo ">>> SLURM_NODELIST content:"
echo $SLURM_NODELIST
NODES=$(echo $SLURM_NODELIST | tr ',' '\n' | uniq | wc -l)
TASKS=$(echo $SLURM_NODELIST | tr ',' '\n' | wc -l)
echo "${NODES}n*${TASKS}t"

# AMD GPU info
echo ' '
echo ">>> AMD GPU info:"
/opt/rocm-6.2.4/bin/rocm-smi

# show current time
start_time=$(date +%s)

# Load environment
source /lustre/orion/nph158/proj-shared/jinchen/env/pyq_test_env.sh

export PYTHONPATH=/lustre/orion/nph158/proj-shared/jinchen/debug/PyQUDA_qTMD:$PYTHONPATH
which python

# Output LD_LIBRARY_PATH
echo -e "\n>>> Output LD_LIBRARY_PATH:"
echo $LD_LIBRARY_PATH


# QUDA global environment
QUDA_RPATH=${WORKDIR}/.cache
mkdir -p ${QUDA_RPATH}

export QUDA_ENABLE_TUNING=1
export QUDA_ENABLE_P2P=0
export QUDA_ENABLE_DEVICE_MEMORY_POOL=0

main=pycontract_main.py

echo "SLURM_NTASKS=$SLURM_NTASKS"

srun -N 4 -n 32 --mpi=cray_shasta --gpus-per-task=1 -u \
    python3 ${main}

# calculate total time
end_time=$(date +%s)
total_time=$((end_time - start_time))
hours=$((total_time / 3600))
minutes=$(((total_time % 3600) / 60))
seconds=$((total_time % 60))
echo " "
echo "Total time: $hours hours $minutes minutes $seconds seconds"
