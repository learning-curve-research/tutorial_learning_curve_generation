#!/bin/sh
#SBATCH --partition=general --qos=short
#SBATCH --time=4:00:00
#SBATCH --mincpus=1
#SBATCH --mem=8000
#SBATCH --job-name=lc11
#SBATCH --output=logs/lc%a.txt
#SBATCH --error=logs/lc%a.txt
#SBATCH --array=0

# Relevant paths:
BULK_DIR="/tudelft.net/staff-bulk/ewi/insy/PRLab/Students/learning_curve"
PROJECT_DIR="${BULK_DIR}/genlc"
CACHE_DIR="/tudelft.net/staff-bulk/ewi/insy/PRLab/Staff/tjviering/lcdb11/openml_cache"
CONTAINER="${PROJECT_DIR}/lcdb11container.sif"
CONFIG_DIR="${HOME}/.config/openml"

ulimit -n 8000


# Run the container with the specified command
apptainer exec -c --bind ${CACHE_DIR}:/openml_cache,${PROJECT_DIR}:/mnt ${CONTAINER} /bin/bash -c "
  mkdir -p ${CONFIG_DIR} && \
  echo 'cachedir=/openml_cache' > ${CONFIG_DIR}/config && \
  cd /mnt/lcdb_function && \
  python compute_lc_anchor.py $SLURM_ARRAY_TASK_ID
"
