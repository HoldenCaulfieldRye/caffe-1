#!/usr/bin/env sh
# Compute the mean image from the imagenet training leveldb
# N.B. this is available in data/ilsvrc12

TOOLS=../../build/tools
DATA=../../data/hatch_markings

$TOOLS/compute_image_mean.bin hatch_markings_train_leveldb $DATA/hatch_markings_mean.binaryproto

echo "Done."
