#!/usr/bin/env sh
# Compute the mean image from the imagenet training leveldb
# N.B. this is available in data/ilsvrc12

TOOLS=../../build/tools
DATA=../../data/controlpoint/$TASK_NAME

$TOOLS/compute_image_mean.bin $TASK_NAME_finetune_train_leveldb $DATA/$TASK_NAME_finetune_mean.binaryproto

echo "Done."
