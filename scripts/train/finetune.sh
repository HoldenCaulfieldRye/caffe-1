#!/bin/bash

# NOTE!
# selecting which labels to learn, whether to merge or rename is not
# specified at command line, so this script is not fully automatic.
# however, these prompts will occur at the beginning only, so as long
# as you have a single task ie a single string in TASK_NAME, this
# script is still useful, all you have to do is reply to prompts.


# with 4, bad minimum provides 80% classification accuracy
IMBALANCE_RATIO=4

for TASK_NAME in soil_risk; do

    # 1. get labels and choose which ones to learn
    source /data/ad6813/caffe/python/venv/bin/activate

    cd /data/ad6813/caffe/scripts/data_preparation
    echo "create_lookup_txtfiles..."
    python create_lookup_txtfiles_2.py --data-dir=/data/ad6813/pipe-data/Bluebox/raw_data/dump --to-dir=/data/ad6813/caffe/data_info/$TASK_NAME --imbalance-ratio=$IMBALANCE_RATIO

    
    # 2. move data (symlinks?) to train/ val/ test/ dirs
    echo "move_to_dirs..."
    python move_to_dirs.py /data/ad6813/pipe-data/Bluebox/raw_data/dump /data/ad6813/caffe/data/controlpoint/$TASK_NAME /data/ad6813/caffe/data_info/$TASK_NAME

    # 3. resize images
    cd /data/ad6813/caffe/data/$TASK_NAME
    CMD=$(convert $(ls train | tail -1) -print "%wx%h") 
    if [ "$CMD" != "256x256" ]
    then
	echo "downsizing all images to 256x256..."
	for name in */*.jpg; do convert -resize 256x256\! $name $name; done
    else
	echo "images already downsized"
    fi

    
    # 4. download alexnet
    if [ -f /data/ad6813/caffe/data/models/alexnet/caffe_alexnet_model ];
    then echo "alexnet already downloaded"
    else
	cd /data/ad6813/caffe/examples/imagenet
	./get_caffe_alexnet_model.sh
	echo "repeat to check correct download:"
	./get_caffe_alexnet_model.sh
	echo "attention: look just above to check correct download"
    fi


    # 5. create leveldb inputs
    cd /data/ad6813/caffe/data/models
    
    # first make sure exists reference dir from which to cp and sed
    if [ -d clampdet-fine ]
    then
	mkdir $TASK_NAME"-fine"
	cd clampdet-fine
	NEEDED_FILES="clampdet_fine_solver.prototxt create_clampdet_fine.sh clampdet_fine_test.prototxt fine_clampdet.sh clampdet_fine_train.prototxt make_clampdet_fine_mean.sh clampdet_fine_val.prototxt resume_training.sh"
	for file in $NEEDED_FILES;
	do
	    if [ ! -f $file ]
	    then
		echo "$file not found"
		echo "need it to create leveldb inputs for $TASK_NAME"
		exit
	    else
		cp $file "../"$TASK_NAME"-fine/"
	    fi
	done
    else
	echo "directory clampdet-fine not found"
	echo "need it to create leveldb inputs for $TASK_NAME"
    fi
    
       
