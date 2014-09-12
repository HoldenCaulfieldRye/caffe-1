#!/usr/bin/env sh
# Compute the mean image from the imagenet training leveldb
# N.B. this is available in data/ilsvrc12

TOOLS=../../build/tools
DATA=../../data/bigger

$TOOLS/compute_image_mean.bin bigger_train_leveldb $DATA/bigger_mean.binaryproto

echo "Done."
