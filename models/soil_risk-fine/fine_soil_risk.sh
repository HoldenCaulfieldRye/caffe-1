#!/usr/bin/env sh

TOOLS=../../build/tools

GLOG_logtostderr=1 $TOOLS/finetune_net.bin $TASK_NAME_finetune_solver.prototxt ../alexnet/caffe_alexnet_model

echo "Done."
