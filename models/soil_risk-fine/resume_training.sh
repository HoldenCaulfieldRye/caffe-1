#!/usr/bin/env sh

TOOLS=../../build/tools

GLOG_logtostderr=1 $TOOLS/train_net.bin \
    soil_risk_finetune_solver.prototxt clampfine_train_iter_4900.solverstate

echo "Done."
