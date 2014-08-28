#!/usr/bin/env sh
# Compute the mean image from the imagenet training leveldb
# N.B. this is available in data/ilsvrc12

TOOLS=../../build/tools
DATA=../../data/fitting_proximity

$TOOLS/compute_image_mean.bin fitting_proximity_train_leveldb $DATA/fitting_proximity_mean.binaryproto

echo "Done."
