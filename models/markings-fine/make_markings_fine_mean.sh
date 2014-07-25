#!/usr/bin/env sh
# Compute the mean image from the imagenet training leveldb
# N.B. this is available in data/ilsvrc12

TOOLS=../../build/tools
DATA=../../data/controlpoint/markings

$TOOLS/compute_image_mean.bin markings_fine_train_leveldb $DATA/markings_fine_mean.binaryproto

echo "Done."
