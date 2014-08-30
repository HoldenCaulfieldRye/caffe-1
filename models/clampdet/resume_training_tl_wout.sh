#!/usr/bin/env sh

TOOLS=../../build/tools

GLOG_logtostderr=1 $TOOLS/train_net.bin \
    clampdet_solver.prototxt logs/tl_wout/clampdet_fine_train_iter_50000.solverstate

echo "Done."
