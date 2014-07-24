#!/usr/bin/env sh
# Compute the mean image from the imagenet training leveldb
# N.B. this is available in data/ilsvrc12

TOOLS=../../build/tools
DATA=../../data/controlpoint/soil_risk

$TOOLS/compute_image_mean.bin soil_risk_finetune_train_leveldb $DATA/soil_risk_finetune_mean.binaryproto

echo "Done."
