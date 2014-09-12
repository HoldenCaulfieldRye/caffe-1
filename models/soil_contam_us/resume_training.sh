#!/usr/bin/env sh

TOOLS=../../build/tools

GLOG_logtostderr=1 $TOOLS/train_net.bin \
    soil_contam_us_solver.prototxt soil_contam_us_fine_train_iter_4000.solverstate

echo "Done."
