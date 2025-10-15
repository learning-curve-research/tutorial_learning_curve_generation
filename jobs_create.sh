#!/bin/sh
#SBATCH --partition=general --qos=short
#SBATCH --time=4:00:00
#SBATCH --mincpus=4
#SBATCH --mem=20000
#SBATCH --output=jobs_create_out.txt
#SBATCH --error=jobs_create_error.txt


# Relevant paths:
PROJECT_DIR="/tudelft.net/staff-bulk/ewi/insy/PRLab/Students/learning_curve/genlc"
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
  python jobs_create.py
"