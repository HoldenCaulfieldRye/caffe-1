#!/usr/bin/env sh

TOOLS=../../build/tools

GLOG_logtostderr=1 $TOOLS/train_net.bin \
    scrape_zones_solver.prototxt scrape_zones_fine_train_iter_4000.solverstate

echo "Done."
