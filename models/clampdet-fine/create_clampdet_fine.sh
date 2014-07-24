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
    clampdet_finetune_train_leveldb 1

GLOG_logtostderr=1 $TOOLS/convert_imageset.bin \
    $DATA/val/ \
    $DATA_INFO/val.txt \
    clampdet_finetune_val_leveldb 1

GLOG_logtostderr=1 $TOOLS/convert_imageset.bin \
    $DATA/test/ \
    $DATA_INFO/test.txt \
    clampdet_finetune_test_leveldb 1

echo "Done."
