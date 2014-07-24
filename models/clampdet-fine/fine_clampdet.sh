#!/usr/bin/env sh

TOOLS=../../build/tools

GLOG_logtostderr=1 $TOOLS/fine_net.bin clampdet_fine_solver.prototxt ../alexnet/caffe_alexnet_model

echo "Done."
