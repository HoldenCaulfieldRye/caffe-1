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
"::Solve("  	   	in src/caffe/solver.cpp
"::ComputeUpdateValue(" in src/caffe/solver.cpp
"::Update(" 		in src/caffe/
"void caffe_cpu_axpby(" in src/caffe/util/math_functions.cpp



