#!/usr/bin/env sh

TOOLS=../../build/tools

GLOG_logtostderr=1 $TOOLS/train_net.bin \
    ground_sheet_fine_solver.prototxt clampfine_train_iter_4900.solverstate

echo "Done."
