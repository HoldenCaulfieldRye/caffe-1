#!/usr/bin/env sh
# Compute the mean image from the imagenet training leveldb
# N.B. this is available in data/ilsvrc12

TOOLS=../../build/tools
DATA=../../data/insertion_markings

$TOOLS/compute_image_mean.bin insertion_markings_train_leveldb $DATA/insertion_markings_mean.binaryproto

echo "Done."
