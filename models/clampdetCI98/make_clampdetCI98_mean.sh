#!/usr/bin/env sh
# Compute the mean image from the imagenet training leveldb
# N.B. this is available in data/ilsvrc12

TOOLS=../../build/tools
DATA=../../data/clampdetCI98

$TOOLS/compute_image_mean.bin clampdetCI98_train_leveldb $DATA/clampdetCI98_mean.binaryproto

echo "Done."
