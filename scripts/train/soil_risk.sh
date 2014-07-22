#!/bin/bash

for taskname in soil_risk; do

    # 1. get labels and choose which ones to learn
    source /data/ad6813/caffe/python/venv/bin/activate

    cd /data/ad6813/caffe/scripts/data_preparation
    echo "create_lookup_txtfiles..."
    python create_lookup_txtfiles_2.py /data/ad6813/pipe-data/Bluebox/raw_data/dump /data/ad6813/caffe/data_info/$taskname

    
    # 2. move data (symlinks?) to train/ val/ test/ dirs
    echo "move_to_dirs..."
    python move_to_dirs.py /data/ad6813/pipe-data/Bluebox/raw_data/dump /data/ad6813/caffe/data/controlpoint/$taskname /data/ad6813/caffe/data_info/$taskname

    # 3. resize images
    cd /data/ad6813/caffe/data/$taskname
    CMD=$(convert $(ls train | tail -1) -print "%wx%h") 
    if [ "$CMD" != "256x256" ]
    then
	echo "downsizing all images to 256x256..."
	for name in */*.jpg; do convert -resize 256x256\! $name $name; done
    else
	echo "images already downsized"
    fi

    
