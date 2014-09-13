#!/usr/bin/env sh

TOOLS=../../build/tools

GLOG_logtostderr=1 $TOOLS/train_net.bin clampdet_solver.prototxt

echo "Done training."
echo "Deleting leveldbs..."

rm -rf *leveldb

echo "Done."

