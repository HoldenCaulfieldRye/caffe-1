#!/usr/bin/env sh

TOOLS=../../build/tools

GLOG_logtostderr=1 $TOOLS/train_net.bin \
    soil_contam_solver.prototxt soil_contam_fine_train_iter_2000.solverstate

echo "Done."
