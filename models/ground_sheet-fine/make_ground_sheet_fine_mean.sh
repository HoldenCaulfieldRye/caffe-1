#!/usr/bin/env sh
# Compute the mean image from the imagenet training leveldb
# N.B. this is available in data/ilsvrc12

TOOLS=../../build/tools
DATA=../../data/controlpoint/ground_sheet

$TOOLS/compute_image_mean.bin ground_sheet_fine_train_leveldb $DATA/ground_sheet_fine_mean.binaryproto

echo "Done."
