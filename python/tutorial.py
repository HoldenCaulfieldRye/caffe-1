import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline

# Make sure that caffe is on the python path:
caffe_root = '../'  # this file is expected to be in {caffe_root}/examples
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe

# Set the right path to your model definition file, pretrained model weights,
# and the image you would like to classify.
MODEL_FILE = 'imagenet/imagenet_deploy.prototxt'
PRETRAINED = 'imagenet/caffe_reference_imagenet_model'
IMAGE_FILE = 'images/cat.jpg'
