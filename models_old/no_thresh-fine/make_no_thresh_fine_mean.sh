#!/usr/bin/env sh
# Compute the mean image from the imagenet training leveldb
# N.B. this is available in data/ilsvrc12

TOOLS=../../build/tools
DATA=../../data/controlpoint/no_thresh

$TOOLS/compute_image_mean.bin no_thresh_fine_train_leveldb $DATA/no_thresh_fine_mean.binaryproto

echo "Done."
