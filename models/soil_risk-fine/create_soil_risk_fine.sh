#!/usr/bin/env sh
# CREATE THE LEVELDB INPUTS
# n.B. set the path to the imagenet train + val data dirs
CAFFE=/data/ad6813/caffe
TOOLS=$CAFFE/build/tools
DATA=$CAFFE/data/$TASK_NAME
DATA_INFO=$CAFFE/data_info/$TASK_NAME

echo "deleting any previous leveldb inputs..."
rm -rf *leveldb
echo "Creating leveldb..."

for TYPE in train val test;
do    
    if [ ! -f $DATA_INFO'/'$TYPE'.txt'  ]
    then
	echo $DATA_INFO'/'$TYPE'.txt not found'
	exit
    else
	if [ ! -d $DATA'/'$TYPE ]
	then
	    echo $DATA'/'$TYPE' not found'
	    exit
	fi
    fi

GLOG_logtostderr=1 $TOOLS/convert_imageset.bin \
    $DATA/train/ \
    $DATA_INFO/train.txt \
    $TASK_NAME_finetune_train_leveldb 1

GLOG_logtostderr=1 $TOOLS/convert_imageset.bin \
    $DATA/val/ \
    $DATA_INFO/val.txt \
    $TASK_NAME_finetune_val_leveldb 1

GLOG_logtostderr=1 $TOOLS/convert_imageset.bin \
    $DATA/test/ \
    $DATA_INFO/test.txt \
    $TASK_NAME_finetune_test_leveldb 1

echo "Done."
