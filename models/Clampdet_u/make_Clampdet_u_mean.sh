#!/usr/bin/env sh
# Compute the mean image from the imagenet training leveldb
# N.B. this is available in data/ilsvrc12

TOOLS=../../build/tools
DATA=../../data/Clampdet_u

$TOOLS/compute_image_mean.bin Clampdet_u_train_leveldb $DATA/Clampdet_u_mean.binaryproto

echo "Done."
