#!/bin/bash
set -e
# any subsequent command that fails will exit the script

# NOTE!
# selecting which labels to learn, whether to merge or rename is not
# specified at command line, so this script is not fully automatic.
# however, these prompts will occur at the beginning only, so as long
# as you have a single task ie a single string in TASK_NAME, this
# script is still useful, all you have to do is reply to prompts.



for TASK_NAME in soil_risk; do

    # with 4, bad minimum provides 80% classification accuracy
    read -p "Target bad min? (e.g. 0.8 for class imbalance such that 80% a bad/fake minimum yields 80% accuracy) " BAD_MIN

    read -p "Max num minibatch passes for training? (20000, cos 10500 was optimal for clampdet-finetune) " MAX_ITER

    # delete this one once you have cuda-convnet style snapshotting
    read -p "Network snapshot frequency? (2000) " SNAPSHOT
    
    
    # 1. get labels and choose which ones to learn
    source /data/ad6813/caffe/python/venv/bin/activate

    if [ ! -d /data/ad6813/caffe/data_info/$TASK_NAME ]
    then
	mkdir /data/ad6813/caffe/data_info/$TASK_NAME
    fi
    
    cd ../data_preparation
    echo "create_lookup_txtfiles..."
    # NUM_OUTPUT is number of classes to learn
    NUM_OUTPUT=$(python -i create_lookup_txtfiles.py data-dir=/data/ad6813/pipe-data/Bluebox/raw_data/dump to-dir=/data/ad6813/caffe/data_info/$TASK_NAME bad-min=$BAD_MIN)

    echo "number of output neurons: "$NUM_OUPUT

    
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
	"mkdir "$TASK_NAME"-fine"
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
		"cp $file ../"$TASK_NAME"-fine/"
	    fi
	done
    else
	echo "directory clampdet-fine not found"
	echo "need it to create leveldb inputs for $TASK_NAME"
	exit
    fi
    
    # now adapt files to taskname
    cd "../"$TASK_NAME"-fine"
    "sed -i 's/clampdet/"$TASK_NAME"/g' *"
    echo "deleting any previous leveldb inputs..."
    rm -rf *leveldb
    echo "creating new leveldb inputs..."
    "./create_"$TASK_NAME"_fine.sh"


    # 6. compute mean image
    echo "computing mean image..."
    "./make_"$TASK_NAME"_fine_mean.sh"

    
    # 7. network definition
    # keeping batchsize 50
    for TYPE in fine val test;
    do
	# change net name and num neurons in output layers
	"sed -i "$TASK_NAME"_fine_"TYPE".prototxt -e '1s!Clamp!"$TASK_NAME"!' -e '300s!2!"$NUM_OUTPUT"!'";
    done
    

    # 8. solver
    "sed -i "$TASK_NAME"_fine_solver.prototxt -e '10s!20000!"$MAX_ITER"!' -e '13s!2000!"$SNAPSHOT"!'"
    
    
    # 9. go!
    "nohup ./finetune_"$TASK_NAME".sh >> train_output.txt 2>&1 &"
    done

