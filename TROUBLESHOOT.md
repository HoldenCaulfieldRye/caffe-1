TROUBLESHOOT
============

# during make all:
/usr/bin/ld: cannot find -lcblas
/usr/bin/ld: cannot find -latlas
# solution:
scp graphic06.doc.ic.ac.uk:/etc/alternatives/lib*las* ~/.local/lib

# create image mean:
Check failed: proto.SerializeToOStream(&output)
# hack solution:
use a sufficiently similar, previously computed image mean
# solution:
paths specified in make_*_image_mean.sh do not exist, fix

# threshold layer:
Check failed: (*top)[0]->num() == (*top)[1]->num() (0 vs. 50) The data and label should have the same number.
# solution:
you'd scp -r 'ed the data from another graphic machine, symlinks were
followed, and actual images were in the data dir. that's not really
supposed to be a pb though.


# plot.py: list index out of range
look at log.{train,test} and see if last line pathogenic


# python wrappers:
ImportError: No module named _caffe
# solution
make pycaffe


# leveldb locked:
IO error: lock *_leveldb/LOCK: already held by process
# solution 1
rm -rf *leveldb
./create
# solution 2
{train,val}.prototxt data_param { source: reference correct? }


# inf nan consec Test Score in log:
Test score #32: 0.146749
Test score #33: -0.214647
Test score #34: 0.0004478
Test score #35: -0.312895
[...]
Iteration 1, lr = 9.995e-05
Iteration 1, loss = nan
# solution:
# you have a layer L linking to layer L+1 and L+2 or smth like that


# adding new layer
# need to modify this crazy
"\0011\"\351\001\n\024ConvolutionParameter\"
stuff in build/src/caffe/proto/caffe.pb.cc ?


# ImportError: No module named _caffe
# _<module> usually stands for <module>.so written in C(++)!
cd /data/add6813/caffe
make pycaffe

# pycaffe::_Net_set_mean()
# ValueError: axes don't match array
when it works:
shape of data blob (10, 3, 227, 227)
shape of mean file:  (3, 256, 256)
but for some reason we want mean to have shape:  (3, 227, 227)
when it doesn't:
shape of data blob (10, 3, 227, 227)
shape of mean file:  (1, 3, 256, 256)
but for some reason we want mean to have shape:  (3, 227, 227)
# solution:
mean_f = mean_f[0]


# DEVELOPMENT

attrib                      |  varname       |  meaning
---------------------------------------------------------
prob_.num()                 |  num           |  batchSize
prob_.count()               |                |
prob_.cpu_data()            |  prob_data     |

bottom[1]                   |  
bottom[1]->count()          |

labels_                     |
labels_.count()             |

bottom_diff[case*dimensionality+neuron]
---------------------------------------------------------

the main functions from which net is trained:
":Solve("  	       in src/caffe/solver.cpp
":Forward("            in src/caffe/net.cpp
":Backward("           in src/caffe/net.cpp
":Backward(const"      in src/caffe/layer.hpp
":ComputeUpdateValue(" in src/caffe/solver.cpp
":Update(" 	       in src/caffe/
":Update("             in src/caffe/blob.cpp     (crux)
"void caffe_cpu_axpby(" in src/caffe/util/math_functions.cpp


# conv1
params_[0] dimensions:
num: 96
channels: 3
height: 11
width: 11
count: 34848

params_[1] dimensions:
num: 1
channels: 1
height: 1
width: 96
count: 96

# conv2?
params_[2] dimensions:
num: 256
channels: 48
height: 5
width: 5
count: 307200

params_[3] dimensions:
num: 1
channels: 1
height: 1
width: 256
count: 256

# conv3?
params_[4] dimensions:
num: 384
channels: 256
height: 3
width: 3
count: 884736

params_[5] dimensions:
num: 1
channels: 1
height: 1
width: 384
count: 384

# conv4?
params_[6] dimensions:
num: 384
channels: 192
height: 3
width: 3
count: 663552

params_[7] dimensions:
num: 1
channels: 1
height: 1
width: 384
count: 384

# conv5?
params_[8] dimensions:
num: 256
channels: 192
height: 3
width: 3
count: 442368

params_[9] dimensions:
num: 1
channels: 1
height: 1
width: 256
count: 256

# fc6?
params_[10] dimensions:
num: 1
channels: 1
height: 4096
width: 9216
count: 37748736

params_[11] dimensions:
num: 1
channels: 1
height: 1
width: 4096
count: 4096

# fc7?
params_[12] dimensions:
num: 1
channels: 1
height: 4096
width: 4096
count: 16777216

params_[13] dimensions:
num: 1
channels: 1
height: 1
width: 4096
count: 4096

# fc8?
# softmax weights!
params_[14] dimensions:
num: 1
channels: 1
height: 2   # one for each softmax neuron
width: 4096 # one for each neuron below
count: 8192

params_[15] dimensions:
num: 1
channels: 1
height: 1
width: 2
count: 2



STEP 1
======

solved. you were wrong, fwd pass not ok. move to step 2

debug SBL:
- fwd pass: OK
- bwd pass: OK
- update: PROB
  -> solver.cpp l.250:
     caffe_cpu_axpby(net_params[param_id]->count(), local_rate,
          net_params[param_id]->cpu_diff(), momentum,
          history_[param_id]->mutable_cpu_data());
	  
     -> cpu_diff() might be where PROB is
     	-> only for param_id = {14,15} do we have nonzero diff, why??
	   -> because backprop accidentally active on fc8 only
	-> woah! exploding/vanishing cpu_diff() with SBL
	   -> which stage outscales the cpu_diff()? none!
	      what happens to cpu_diff() b4/after bwd pass?
	      add couts in net.cpp l.269
	      -> net.hpp::ForwardBackward calls
	          net::Backward calls
		   layer::Backward calls
		    specific_layer::Backward_cpu
		 solver::ComputeUpdateValue
    		 solver::net_->Update()
	   -> compared w/ benchmark throughout an iteration, similar
	      values (also outscale for benchmark)
	      
   -> actual parameter values might be where PROB is
      -> solver::net_->Update() calls
	     net::Update calls
	       blob::Update
      -> compare logs
	   -> PROB1: layer[10] is 0 for sbl only
	      -> should be ..?
	   -> loss just after ForwardBackward:
	   already fucked up
	   -> cpu_diff just after ForwardBackward: 
		     sbl max       benchmark max
	   net_15    e+34          e+31
	   net_14    0.92          0.99
	   net_13    0.015         0.009
	   net_12    0    !        0.013 (but no neg values!)
	   net_11    0.0055        0.0027
	   -> cpu_diff just after ComputeUpdateValue():
	   fine
	   -> current params, diff, new params:
	   fine
	   -> cpu_diff just after Update():
	   fine
	   
  -> so the fucked up stuff occurs inside ForwardBackward()
     -> occurs inside net.hpp::Forward() or before
     -> 


STEP 2
======

issue with the update. after 1 iteration, next loss is 14
for SBL, 73 for SL. bottom_diff takes on rubbish values.

so:
- bwd pass is wrong
- weight update is wrong

-> after 1 iteration, net only outputs 1s or 0s!
   -> so z = <x,w> + b can easily = 0 ? how?

-> ok, cost function seems to be working now. no nans or infs,
   and trianing error gets minimised.


   
STEP 3
======

Why is accuracy so weird?
-> find out whether same net loaded in by printing out param values
   -> have identical train and val files with just 128 cases
      compare output probs

Examine outputs
-> is this harsh error preventing the net from learning anything?
   -> ie all outputs are around 0.5, it's very confused
   -> find out by comparing benchmarks
      -> ground sheet outputs
      	 -> min class is 1, so other way around
      	 -> uh oh, forgot to get SL to print them out
      	 -> 23-08-2014 has them, 22-08-2014 doesnt
      	 -> 22-08-2014 is from old build, you can compare train time
      	    series with 23* to make sure new build isn't doing
	    anything different or wrong
      -> scrape zones outputs
      	 -> min class is 0
   -> if so, make it less harsh?
      -> less extreme renormalisation
      -> only penalise if output <=0.5 ie introduce kink in cost
      	 function
	 -> formula?
   -> implement under-sampling like the paper says

   
Test if correctly implemented:
-> graphic06: on a dataset of 6 images, perfectly balanced, batchsize 6, 
   train and val sets the same
   -> prob outputs not same for val as for train
      -> calling bottom[0] in SBL, prob_ in PCA
         is one of them wrong?
	 maybe SBL is wrong, hence bad results below?
   -> loss same for sbl and sl at iter_1, but not afterwards
   -> bottom_diff not same for sbl as for sl
   -> CAREFUL! after debugging, get back data/ground_sheet/temp

Read the paper threshold paper properly!   


IDENTIFIED PROBS & SOLS:
-> what if prior is (1,0)
-> must implement under- and over-sampling as well
   shit that will be hard



=====

Fuck it, that is too hard. And it might not even work. Threshold
works, is easier to implement, and probably has more powerful results.

currently:
-> graphic07 writing python wrappers for running the net
-> idea is to get the prediction probs, and assign flags based on them
& threshold
-> debugging run_classifier.py
   -> done

=====


Need to:
- train nets
  -> use optimal backprop freeze
- use Redbox data
  -> script to use Redbox data from a certain date
     -> graphic07 meta.zip
  -> try multiple threshold dates
  -> use best performing network so far
     -> clampdet 94%, what arch was that?
     	-> clampdet                           0.2 
	-> no_thresh-fine                     0.12
	-> thresh                             0.12
	-> thresh_freeze_backprop5/13         0.7
	-> thresh_freeze_backprop5/14         0.7
	-> thresh_freeze_backprop5.5/11       0.15
	-> thresh_freeze_backprop5.5/12       0.15
	-> thresh_freeze_backprop5.5/13       0.39
	-> thresh_freeze_backprop5.5/14       0.4     
	-> thresh_freeze_backprop5.5/15       0.18
	-> thresh_freeze_backprop6/11         0.17   
	-> thresh_freeze_backprop6/13         0.4   
	-> thresh_freeze_backprop7/11         0.17
     ok seems perf driven by:
     - expressiveness 
     - whether lr_policy fucked up
     still space for optimising both
	
     -> better than optimal backprop freeze?
- write up threshold
- write up sbl


=====


screw the Redbox data. focus on running experiments
from below.


-> ReLU maths
   -> neat writeup
   
-> Early stopping maths
   -> draft
   -> neat writeup
   
-> Generic clamp
   -> restructure
   -> neat writeup

-> Transfer learning
   -> freezing backprop
   -> initialising weights

-> Class imbalance
   -> under-sampling
   -> in-net threshold
   -> SBL
   -> test-time threshold

-> Final Results


=====

What nets do I still need to train?

- Generic Clamp:
  -> mis-labelling, how to show?

- Transfer Learning
  -> test run
     -> with:  clampdet/08                                
     -> w/out:                                TRAINING tl_wout
  -> clampdet, freeze backprop on:
     -> none:  clampdet/08
     -> conv1:                                
     -> conv2:                                
     -> conv3:                                
     -> conv4:                                
     -> conv5:                                TRAINING 
     -> fc6:                                  TRAINING
     -> fc7:   thresh_freeze6/11              TODO? weight_decay
     -> fc8:   thresh_freeze7/11              TODO? weight_decay
  -> weight initialisation?
     -> reinit best net from above            TODO
     -> PROPER best net from above            TODO
  -> parametric vs non parametric
     -> linear SVM			      TRAINING
     -> best net from above
  
- Class Imbalance
  fitting proximity
  -> test run                                 TRAINING
  -> under-sampling                           TODO need diff leveldb
  -> over-sampling                            TODO need diff leveldb 
  -> within-net threshold                     TODO             
  -> SBL                                      TODO
  -> test-time threshold                      TODO          
  

- Final Results
  what is the best arch?
  -> clampdet
  -> ground sheet
  -> hatch markings
  -> insertion depth markings
  -> scrape zones
  -> joint misaligned
  -> contamination
  -> fitting proximity
  -> scraping peeling

  
What do I still need to write (from scratch)?
- Background:
  -> why neural nets so good?
     because they generalise so well
     why do we care about generalising well?
     because of curse of dimensionality
     bit.ly/1pEOuYV
     how does neural net generalise so well?
     with distributed representation
     ie hierarchical representation
     ie compositionality of parameters
     ie exponential compactness
     
  -> AlexNet in detail, Rob Fergus tutorial
  
- Transfer Learning:
  -> conv vs fc, intriguing properties


  

NEXT:
- other nets to train:
  -> write class imbalance fitting proximity prototxts & leveldbs
  
- finished nets:
  -> run_classifier.py on them
  -> plots
  -> write up:
     -> comments
     -> plot
     -> table from run_classifier
     


=====

Just realised:
- all recent clampdets in bad min
- clampdet/08 got 94% accuracy
  -> how was it trained??
     -> under-sampling

Retrain all clampdets, with undersampling     




