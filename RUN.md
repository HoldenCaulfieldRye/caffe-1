
1. move data (symlinks) to data/<taskName>/{train,val,test} dirs
=====================================================

cd scripts/data_preparation
python setup_data.py data-dir=/data/ad6813/pipe-data/Bluebox/raw_data/dump data-info=/data/ad6813/caffe/data_info/<taskName> to-dir=/data/ad6813/caffe/data/<taskName> bad-min=<under_sample_to_this_level_of_imbalance>(or N for no undersampling)


2. get a script to do the rest
==============================

set variables inside setup_rest.sh



##################################################################
### troubleshoot section
##################################################################

10. env vars
============

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/homes/ad6813/.local/lib:/usr/local/cuda/lib64:/usr/local/cuda/lib


11.1 fine-tune
=============

cf mezNs answer: bit.ly/1siq8Wz

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


train
=========

./train_clampdet.sh

solver.cpp:106:]  Iteration 0, Testing net
syncedmem.cpp:47] Check failed: error == cudaSuccess (30 vs. 0)  unknown error

look at issues raised online.
maybe need to "rename the last fully connected layer from the imagenet_train/val.prototxt and reduced its number of outputs to 2"

try fine-tuning, see if you get same error.

try training cifar10, see if you get same error.


resume
===========

if paused/crashed and need to resume training:

./resume_training.sh

should load a snapshot, eg caffe_imagenet_train_1000.solverstate.


for more troubleshoots, see TROUBLESHOOT.md in current directory.

==========

clampdet_u: (9 17), (11 14), (.) (53 duplicates)

clampdet: (9 11 14 17), (.)





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
- 'Spatial Pyramid Pooling'
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
<<<<<<< HEAD
- crop at test time too
=======
>>>>>>> c3db24c936d6992862600fa7d0f0c3130539c3df
  

=====

NEXT:
- why performance decreasing with more data?
  -> unsuitable images 
  -> train only on 1st query in cases of multi-query
  -> look at mis-classifications:
     -> visually obvious outliers?
     -> dense in a client, employee, time?
- Redbox data which usable 
- leveldb fixed mini batch proportions     
- to see droplets and soil marks on joint
- download ImageNet, store on terabyte drives

  


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
     

Caffe vs Torch: github.com/BVLC/caffe/issues/642

