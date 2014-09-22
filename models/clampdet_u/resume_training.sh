#!/usr/bin/env sh

TOOLS=../../build/tools

GLOG_logtostderr=1 $TOOLS/train_net.bin \
    clampdet_u_solver.prototxt clampdet_u_fine_train_iter_4000.solverstate

echo "Done."
