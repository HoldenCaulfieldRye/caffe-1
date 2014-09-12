#!/usr/bin/env sh

TOOLS=../../build/tools

GLOG_logtostderr=1 $TOOLS/finetune_net.bin hatch_markings_solver.prototxt /homes/ad6813/net-saves/clampdet/none/clampdet_6000

echo "Done."
