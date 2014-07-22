
1. get labels and choose which ones to learn
============================================

cd caffe/scripts/data_preparation
python create_lookup_txtfiles.py path/to/rawdata path/to/caffe/data_info/yourtask


BE IN PYTHON VENV!


2. move data (symlinks?) to train/ val/ test/ dirs
==================================================

# taskdata is where train/ val/ test/ dirs will be

python move_to_dirs.py /data/ad6813/pipe-data/Bluebox/raw_data/dump/ /data/ad6813/caffe/data/controlpoint/clampdet /data/ad6813/caffe/data_info/controlpoint/clampdet


3. resize images
================
cd /data/ad6813/caffe/data/controlpoint/clampdet
for name in */*.jpg; do convert -resize 256x256\! $name $name; done

# check
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

clampdet_train.prototext
clampdet_val.prototext
clampdet_test.prototext

# last layer 'loss' or 'accuracy' based on backprop or measure
# performance

# num_output is number of neurons per layer

# make batch_size for val (as large as possible?)


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



next steps:
- USA visa!
- test rebalance (esp contents of dump files for no repetition and
  correct numbers)
- save net only if performance gain (cf pylearn2)
- train all tasks
- F measure
- data augmentation
- impose best possible class balance at every batch
- Razvan: different batch contents at every epoch
- Razvan: pre-process image, select patch of interest
- look at new data fields sent by ControlPoint

