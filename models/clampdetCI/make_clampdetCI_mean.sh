#!/usr/bin/env sh
# Compute the mean image from the imagenet training leveldb
# N.B. this is available in data/ilsvrc12

TOOLS=../../build/tools
DATA=../../data/clampdetCI

$TOOLS/compute_image_mean.bin clampdetCI_train_leveldb $DATA/clampdetCI_mean.binaryproto

echo "Done."
