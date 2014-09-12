#!/usr/bin/env sh

TOOLS=../../build/tools

GLOG_logtostderr=1 $TOOLS/finetune_net.bin joint_misaligned_solver.prototxt ../alexnet/caffe_alexnet_model

echo "Done."
