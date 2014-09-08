#!/usr/bin/env sh
# Compute the mean image from the imagenet training leveldb
# N.B. this is available in data/ilsvrc12

TOOLS=../../build/tools
DATA=../../data/clampdetCI98_us

$TOOLS/compute_image_mean.bin clampdetCI98_us_train_leveldb $DATA/clampdetCI98_us_mean.binaryproto

echo "Done."
