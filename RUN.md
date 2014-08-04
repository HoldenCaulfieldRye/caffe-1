
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




DEBUG TROUBLESHOOT
==================
# during make all:
/usr/bin/ld: cannot find -lcblas
/usr/bin/ld: cannot find -latlas
# solution:
scp graphic06.doc.ic.ac.uk:/etc/alternatives/lib*las* ~/.local/lib

# create image mean:
Check failed: proto.SerializeToOStream(&output)
# solution:
just a sufficiently similar, previously computed image mean

# threshold layer:
Check failed: (*top)[0]->num() == (*top)[1]->num() (0 vs. 50) The data and label should have the same number.
# solution:
you'd scp -r 'ed the data from another graphic machine, symlinks were
followed, and actual images were in the data dir. that's not really
supposed to be a pb though.





CURRENTLY
=========

- 06: soil_risk, soil_risk2, test_layer
- 07: scrape_zone_peel
- 08:
- 09:
- 10:

classifiers:
- soil contamination risk
  -> classes:   SoilContamination{Low,High,}Risk
  -> trainsize: 2275
  -> badmin:    0.8
  -> val*:      0.83
  -> iter:      100     (bad min?)
- soil contamination risk
  -> trainsize: 910
  -> badmin:    0.5
  -> val*:      0.66
  -> iter:      3100
- ground sheet
  -> trainsize: 11031
  -> badmin:    0.65    (natural)
  -> val*:      0.87
  -> iter:      15200
- scrape_zone_peel
  -> classes:   union CantSeeScrapeZones, UnsuitableScrapingPeeling, NoEvidenceScrapingPeeling
  -> trainsize: 11031   (??groundsheet)
  -> badmin:    0.71    
  -> val*:      0.73    (bad min?)
  -> val_100:   0.71
  -> iter:      2100
- (insertion depth) markings
  -> trainsize: 3587
  -> badmin:    0.8
  -> val*:      0.89
  -> val_100:   0.85    (wow! lower learning rate always?)
  -> iter:      11200
- hatch markings
  -> trainsize: 6068
  -> badmin:    0.5     (use more data, 0.5 maybe too strict)
  -> val*:      0.78
  -> val_100:   0.65
  -> iter:      16500


next steps:
- thoughts above
- are those really bad_mins: look at val error at every iteration
- why do scrape_zone_peel and ground_sheet have same trainsize
- need test error for true performance
- larger minibatch for less noisy gradient
- lower learning rate 
- Redbox data
- controlpoint extra metadata
- google doc specified next steps
- USA VISA!models/clampdet-fine/
- script integrate python 
- save net only if performance gain (cf pylearn2)
- F measure
- data augmentation
- impose best possible class balance at every batch
- Razvan: different batch contents at every epoch
- Razvan: pre-process image, select patch of interest
- look at new data fields sent by ControlPoint
- actually break CAPTCHAs by reproducing Goodfellow's work, open
  source everything and say you are better than Vicarious
- prove that we can scale (have developed a magic formula for doing
  any kind of image classification) by publishing results on as many
  Kaggle image classification challenges as possible


  
Razvan recommends next:
- main question is: is it working at all?
  -> test interval 1 to see whether bad min all along (
- what is the imbalance in each case?
  -> if equal to accuracy, then doing nothing
- if doing nothing, need to understand why
  -> figure out how everything is working
  -> properly evaluate accuracy (per class)
