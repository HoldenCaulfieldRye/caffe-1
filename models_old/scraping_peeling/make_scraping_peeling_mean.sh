#!/usr/bin/env sh
# Compute the mean image from the imagenet training leveldb
# N.B. this is available in data/ilsvrc12

TOOLS=../../build/tools
DATA=../../data/scraping_peeling

$TOOLS/compute_image_mean.bin scraping_peeling_train_leveldb $DATA/scraping_peeling_mean.binaryproto

echo "Done."
