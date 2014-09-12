#!/usr/bin/env sh

TOOLS=../../build/tools

GLOG_logtostderr=1 $TOOLS/train_net.bin \
    bigger_solver.prototxt bigger_fine_train_iter_4000.solverstate

echo "Done."
