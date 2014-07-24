#!/usr/bin/env sh

TOOLS=../../build/tools

GLOG_logtostderr=1 $TOOLS/fine_net.bin soil_risk_fine_solver.prototxt ../alexnet/caffe_alexnet_model

echo "Done."
