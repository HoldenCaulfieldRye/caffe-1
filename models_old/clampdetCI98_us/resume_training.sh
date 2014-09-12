#!/usr/bin/env sh

TOOLS=../../build/tools

GLOG_logtostderr=1 $TOOLS/train_net.bin \
    clampdetCI98_us_solver.prototxt clampdetCI98_us_fine_train_iter_4000.solverstate

echo "Done."
