#!/usr/bin/env sh
# CREATE THE LEVELDB INPUTS
# n.B. set the path to the imagenet train + val data dirs
CAFFE=/data/ad6813/caffe
TOOLS=$CAFFE/build/tools
DATA=$CAFFE/data/clampdet
DATA_INFO=$CAFFE/data_info/clampdet

echo "deleting any previous leveldb inputs..."
rm -rf *leveldb
echo "Creating leveldb..."

GLOG_logtostderr=1 $TOOLS/convert_imageset.bin \
    $DATA/train/ \
    $DATA_INFO/train.txt \
    clampdet_finetrain_leveldb 1

GLOG_logtostderr=1 $TOOLS/convert_imageset.bin \
    $DATA/val/ \
    $DATA_INFO/val.txt \
    clampdet_fineval_leveldb 1

GLOG_logtostderr=1 $TOOLS/convert_imageset.bin \
    $DATA/test/ \
    $DATA_INFO/test.txt \
    clampdet_finetest_leveldb 1

echo "Done."
