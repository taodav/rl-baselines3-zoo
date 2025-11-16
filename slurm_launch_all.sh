#!/bin/bash
BUFFERS_DIR='/users/rtao7/projects/rl-baselines3-zoo/buffers'
RAN_JOBS=''

for item in $BUFFERS_DIR/*/;
do
  printf "\n"
  FILE_NAME=$item"buffer.npz"
  JOB_NAME=$(basename $item)
  echo "---------LAUNCHING $FILE_NAME---------"
  echo "---------JOB NAME: $JOB_NAME---------"
  sbatch slurm_launch.sh $FILE_NAME -j $JOB_NAME
done