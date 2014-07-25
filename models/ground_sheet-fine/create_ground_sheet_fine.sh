#!/usr/bin/env sh
# CREATE THE LEVELDB INPUTS
# n.B. set the path to the imagenet train + val data dirs
CAFFE=/data/ad6813/caffe
TOOLS=$CAFFE/build/tools
DATA=$CAFFE/data/ground_sheet
DATA_INFO=$CAFFE/data_info/ground_sheet

for TYPE in train val test;
do
    if [ ! -f $DATA_INFO'/'$TYPE'.txt' ]
    then
	echo 'file '$DATA_INFO'/'$TYPE'.txt not found'
	exit
    else
	if [ ! -d $DATA'/'$TYPE ]
	then
	    echo 'directory '$DATA'/'$TYPE' not found'
	    exit
	fi
    fi
done


echo "deleting any previous leveldb inputs..."
rm -rf *leveldb
echo "Creating leveldb..."

GLOG_logtostderr=1 $TOOLS/convert_imageset.bin \
    $DATA/train/ \
    $DATA_INFO/train.txt \
    ground_sheet_fine_train_leveldb 1

GLOG_logtostderr=1 $TOOLS/convert_imageset.bin \
    $DATA/val/ \
    $DATA_INFO/val.txt \
    ground_sheet_fine_val_leveldb 1

GLOG_logtostderr=1 $TOOLS/convert_imageset.bin \
    $DATA/test/ \
    $DATA_INFO/test.txt \
    ground_sheet_fine_test_leveldb 1

echo "Done."
