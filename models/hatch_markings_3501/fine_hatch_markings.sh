#!/usr/bin/env sh

TOOLS=../../build/tools

GLOG_logtostderr=1 $TOOLS/finetune_net.bin hatch_markings_fine_solver.prototxt ../alexnet/caffe_alexnet_model

echo "Done."