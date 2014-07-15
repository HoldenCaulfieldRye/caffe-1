#!/usr/bin/env sh
# Compute the mean image from the imagenet training leveldb
# N.B. this is available in data/ilsvrc12

TOOLS=../../build/tools
DATA=../../data/controlpoint/clampdet

$TOOLS/compute_image_mean.bin clampdet_train_leveldb $DATA/clampdet_mean.binaryproto

echo "Done."
