#!/usr/bin/env sh
# Compute the mean image from the imagenet training leveldb
# N.B. this is available in data/ilsvrc12

TOOLS=../../build/tools
DATA=../../data/contamination

$TOOLS/compute_image_mean.bin contamination_train_leveldb $DATA/contamination_mean.binaryproto

echo "Done."
