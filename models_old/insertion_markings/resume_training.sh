#!/usr/bin/env sh

TOOLS=../../build/tools

GLOG_logtostderr=1 $TOOLS/train_net.bin \
    insertion_markings_solver.prototxt insertion_markings_fine_train_iter_4000.solverstate

echo "Done."
