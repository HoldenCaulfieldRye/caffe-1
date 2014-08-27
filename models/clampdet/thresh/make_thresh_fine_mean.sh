#!/usr/bin/env sh
# Compute the mean image from the imagenet training leveldb
# N.B. this is available in data/ilsvrc12

TOOLS=../../build/tools
DATA=../../data/controlpoint/thresh

$TOOLS/compute_image_mean.bin thresh_fine_train_leveldb $DATA/thresh_fine_mean.binaryproto

echo "Done."
