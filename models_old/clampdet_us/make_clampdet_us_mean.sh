#!/usr/bin/env sh
# Compute the mean image from the imagenet training leveldb
# N.B. this is available in data/ilsvrc12

TOOLS=../../build/tools
DATA=../../data/clampdet_us

$TOOLS/compute_image_mean.bin clampdet_us_train_leveldb $DATA/clampdet_us_mean.binaryproto

echo "Done."
