#!/usr/bin/env sh
# Compute the mean image from the imagenet training leveldb
# N.B. this is available in data/ilsvrc12

TOOLS=../../build/tools
DATA=../../data/soil_contam_us95

$TOOLS/compute_image_mean.bin soil_contam_us95_train_leveldb $DATA/soil_contam_us95_mean.binaryproto

echo "Done."
