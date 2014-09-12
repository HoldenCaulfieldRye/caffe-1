
1. get labels and choose which ones to learn
============================================

cd caffe/scripts/data_preparation
python create_lookup_txtfiles.py path/to/raw/data path/to/caffe/data_info/yourtask


BE IN PYTHON VENV!


2. move data (symlinks?) to train/ val/ test/ dirs
==================================================

# taskdata is where train/ val/ test/ dirs will be

python move_to_dirs.py /data/ad6813/pipe-data/Bluebox/raw_data/dump/ /data/ad6813/caffe/data/controlpoint/clampdet /data/ad6813/caffe/data_info/clampdet


3. resize images
================
cd /data/ad6813/caffe/data/controlpoint/clampdet
for name in */*.jpg; do convert -resize 256x256\! $name $name; done

# troubleshoot:
convert.im6: unable to open image `98411.jpg': No such file or directory @ error/blob.c/OpenBlob/2638.
convert.im6: option requires an argument `-print' @
error/convert.c/ConvertImageCommand/2220.
# you forgot the /dev/null


# check
cd train
convert 102003.jpg -print "Size: %wx%h\n" /dev/null


4. download alexnet
===================
cd /data/ad6813/caffe/examples/imagenet
./get_caffe_alexnet_model.sh
./get_caffe_alexnet_model.sh   # repeat to check correct download


Executables are in:
===================

caffe/build/tools


5. batch & create leveldb inputs
================================

cd /data/ad6813/caffe/examples/imagenet

NOTE: train_leveldb val_leveldb test_leveldb should not exist before this execution!

./create_clampdet.sh

'Opening leveldb imagenet_val_leveldb'
# leveldb is an open source database implementation by google

# Symlinks were not followed
# modified caffe/src/caffe/util/io.cpp to accomodate
# forked caffe and cleanly implemented it


6. compute mean image
=====================

./make_clampdet_mean.sh


7. network definition
=====================

clampdet_train.prototxt
clampdet_val.prototxt
clampdet_test.prototxt

# last layer 'loss' or 'accuracy' based on backprop or measure
# performance

# num_output is number of neurons per layer

# make batch_size for val (as large as possible?)

# the number of blobs_lr's in a layer "should be 0 or same as num of
  layer's parameter blobs" - (??)

# blobs_lr, there are usually 2: one for weight lr, other for bias.
# it is to be multiplied by lr given in solver.

# name and top of a layer need not have same name, makes no diff really.


8. solver
=========

this is like options.cfg for cuda-convnet. contains parameters for
how to carry out training.

see solvers in clampdet and cifar10 for useful comments.


9. deploy
=========

not sure what deploy.prototxt file is for. looks like it's an
alternative to train.prototxt.
but it has more info:
input_dim, weight_filler, bias_filler.
and it's for running, not training:
last layer is not of type softmax_loss, but softmax.

ok I'm guessing it's not for training, but literally for deploying,
as in for a product.

not sure why data is 4-dimensional.
alexnet and imagenet have 1st input_dim set to 10, whereas cifar10
is set to 1. the latter only makes sense to me.


10. env vars
============

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/homes/ad6813/.local/lib:/usr/local/cuda/lib64:/usr/local/cuda/lib



11.1 fine-tune
=============

cf mezN's answer: bit.ly/1siq8Wz

# need a dedicated source, even if dataset same as non finetune,
# because eg batchsize can be different, and resource can be locked
# for parallel training. make sure leveldb names are unique:
./create_clampdet_finetune.sh

# need a dedicated mean image
./make_clampdet_finetune_mean.sh

# go
./finetune_clampdet.sh

# save the nets to /data/ad6813/my-nets/saves/caffe for neatness

# set blobs_lr AND weight_decay to 0 if you want to freeze backprop on them

# change layer name not to match alexnet's to re-initialize weights


11.2 train
=========

./train_clampdet.sh

solver.cpp:106:]  Iteration 0, Testing net
syncedmem.cpp:47] Check failed: error == cudaSuccess (30 vs. 0)  unknown error

look at issues raised online.
maybe need to "rename the last fully connected layer from the imagenet_train/val.prototxt and reduced its number of outputs to 2"

try fine-tuning, see if you get same error.

try training cifar10, see if you get same error.


11.3 resume
===========

if paused/crashed and need to resume training:

./resume_training.sh

should load a snapshot, eg caffe_imagenet_train_1000.solverstate.


12. test
========
???



DEBUG TROUBLESHOOT
==================
see TROUBLESHOOT.md


CURRENTLY
=========

training:
- 05: ERROR importing numpy
- 06: clampdet/conv4
- 07: soil_contam_us/none_lr5_sbl (97%)
- 08: scrape_zones/none_lr5_sbl (0.88)
- 09: 

queued:
-> clampdetCI98/none_bs256_lr4
-> 
-> 
-> TL where bad min deeper than good min
-> 
-> 



classifiers:
- soil contam       74.8
- water contam      50.0
- scraping peeling  67.3
- clamp             89.2
- scrape zones      74.7
- joint misaligned  50.0
- fitting proximity 50.0
--
- ground sheet      88.6
- insertion mkings  78.1
- hatch mkings      76.9


next steps:
- controlpoint extra metadata
- Redbox data
  -> best date
  -> best guys
- impose best possible class balance at every batch
- new caffe with separate train val for higher batchsize with
  higher res images
- data augmentation: rotations & PCA
  -> McCormac's code
  -> bit.ly/UZ3p3E
  -> scikit-image
  -> Razvan/preproc.py
- multi-image query issue:
  -> concatenate images, or
  -> cross-image pooling after conv5
- save net only if performance gain
  -> solver.cpp l.114 approx
- different batch contents at every epoch
- preprocess: divide images by the image standard deviation and apply
  "subtractive/divisive normalization"
- ensemble network, bagging
- 'CNN features off-the-shelf', optimising CNN features for specific
  tasks: 29, 15, 51, 43, 41
- import solverstate shortly before val min
  larger minibatch, smaller lr
- partial re-initialisation of conv layer: bit.ly/1BfKVwL
- extract features with caffe
- switch to OverFeat
- cropping component:
  -> can it be trained with backprop
     -> read LeNet StreetView papers
  -> how does it learn where to crop
     -> localised clampdet
     -> crop conditional on clampdet location & joint type
- MSE
  -> Brebisson reports different predictions
  -> just to show off improvement
  

to see droplets and soil marks on joint
  


Image preprocessing:
- scraping peeling:
  -> Colours, Colour Balance, Shadows, Magenta -100?
  ->
- Water Contamination:
  -> Colours, Colour Balance: 
     -> Midtones, Cyan 100
     -> Highlights, Magenta 100
  -> Hue-Saturation
     -> Hue, -20
  -> Colorise
     -> Hue, 0
     -> Saturation, 0
  -> Brightness-Contrast
     -> Contrast, 10
  -> 
     



