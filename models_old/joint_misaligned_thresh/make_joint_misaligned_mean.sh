#!/usr/bin/env sh
# Compute the mean image from the imagenet training leveldb
# N.B. this is available in data/ilsvrc12

TOOLS=../../build/tools
DATA=../../data/joint_misaligned

$TOOLS/compute_image_mean.bin joint_misaligned_train_leveldb $DATA/joint_misaligned_mean.binaryproto

echo "Done."
