#!/usr/bin/env sh
# Compute the mean image from the imagenet training leveldb
# N.B. this is available in data/ilsvrc12

TOOLS=../../build/tools
DATA=../../data/scrape_zones

$TOOLS/compute_image_mean.bin scrape_zones_train_leveldb $DATA/scrape_zones_mean.binaryproto

echo "Done."
