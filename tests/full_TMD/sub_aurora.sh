#!/bin/bash -l

#PBS -N qTMD
#PBS -A StructNGB
#PBS -l select=1:ngpus=4
#PBS -l filesystems=home
#PBS -q debug
#PBS -j oe
#PBS -l walltime=1:00:00
#PBS -o /lus/flare/projects/StructNGB/jinchen/package/PyQUDA_qTMD/tests/full_TMD/logs/full_TMD_S8T8_aurora.log

# switch to the submit directory
WORKDIR=/lus/flare/projects/StructNGB/jinchen/package/PyQUDA_qTMD/tests/full_TMD
cd $WORKDIR

# output node info
echo ' '
echo ">>> PBS_NODEFILE content:"
cat $PBS_NODEFILE
NODES=$(cat $PBS_NODEFILE | uniq | wc -l)
TASKS=$(wc -l < $PBS_NODEFILE)
echo "${NODES}n*${TASKS}t"

# Get GPU info
# nvidia-smi
# nvcc --version

# show current time
start_time=$(date +"%Y-%m-%d %H:%M:%S")
echo "Start time: $start_time"

# Initialize python and pyquda properly
source /lus/flare/projects/StructNGB/jinchen/env/pyquda_env.sh

# check python version
python --version

# check python path
export PYTHONPATH="/lus/flare/projects/StructNGB/jinchen/package/PyQUDA_qTMD/tests/full_TMD:/lus/flare/projects/StructNGB/jinchen/package/PyQUDA_qTMD"
echo "Python path: $(which python)"
echo "PYTHONPATH: $PYTHONPATH"


# export PYQ_LIB_PATH=/home/jinchen/software/pyq/lib
# export SITE_PACKAGES=/home/jinchen/software/pyq/lib/python3.10/site-packages
export PYQ_LIB_PATH=/lus/flare/projects/StructNGB/jinchen/env/pyq_venv/lib
export SITE_PACKAGES=/lus/flare/projects/StructNGB/jinchen/env/pyq_venv/lib/python3.10/site-packages
export LD_LIBRARY_PATH=$PYQ_LIB_PATH:$SITE_PACKAGES/dpctl:$SITE_PACKAGES/dpnp:$LD_LIBRARY_PATH

# print to confirm the path contains libur_loader.so
echo "Checking for libur_loader.so in PYQ_LIB_PATH:"
ls -l $PYQ_LIB_PATH/libur_loader.so* 2>/dev/null || echo "Not found in $PYQ_LIB_PATH"


echo ">>> Running pyquda_aurora.py"
mpirun -n 1 python3 pyquda_aurora.py

# calculate total time
end_time=$(date +"%Y-%m-%d %H:%M:%S")
echo "End time: $end_time"

# total time
start_seconds=$(date --date="$start_time" +%s)
end_seconds=$(date --date="$end_time" +%s)
duration=$((end_seconds - start_seconds))
hours=$((duration / 3600))
minutes=$(( (duration % 3600) / 60 ))
seconds=$((duration % 60))
echo "Total time: ${hours}h ${minutes}m ${seconds}s"
