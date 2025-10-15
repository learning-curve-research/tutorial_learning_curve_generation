#!/bin/sh
#SBATCH --partition=general --qos=short
#SBATCH --time=1:00:00
#SBATCH --mincpus=2
#SBATCH --mem=8000
#SBATCH --output=jobs_resubmit_out.txt
#SBATCH --error=jobs_resubmit_error.txt


# Relevant paths:
PROJECT_DIR="/tudelft.net/staff-bulk/ewi/insy/PRLab/Students/learning_curve/tutorial_learning_curve_generation"
CACHE_DIR="/tudelft.net/staff-bulk/ewi/insy/PRLab/Staff/tjviering/lcdb11/openml_cache"
CONTAINER="${PROJECT_DIR}/lcdb11container.sif"
CONFIG_DIR="${HOME}/.config/openml"


# Run the container with the specified command
apptainer exec -c \
  --bind ${CACHE_DIR}:/openml_cache \
  --bind ${PROJECT_DIR}:/mnt \
  ${CONTAINER} /bin/bash -c "
  mkdir -p ${CONFIG_DIR} && \
  echo 'cachedir=/openml_cache' > ${CONFIG_DIR}/config && \
  cd /mnt/ && \
  python jobs_resubmit.py
"