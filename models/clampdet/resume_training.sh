#!/usr/bin/env sh

TOOLS=../../build/tools

GLOG_logtostderr=1 $TOOLS/train_net.bin \
    clampdet_solver.prototxt clampdet_fine_train_iter_4000.solverstate

echo "Done training."
echo "Deleting leveldbs..."

rm -rf *leveldb

echo "Done."

