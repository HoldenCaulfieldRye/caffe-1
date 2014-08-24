import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
import os, sys
import caffe
from os.path import join as ojoin
from subprocess import call

sys.path.insert(0, caffe_root + 'python')

# usage:
# python run_classifier.py classifier-dir=.. images=..


if __name__ == '__main__':
  # Make sure that caffe is on the python path:
  caffe_root = '../'  # this file is expected to be in {caffe_root}/examples

  classifier_dir, images = None, None
  for arg in sys.argv:
    if "classifier-dir=" in arg:
      classifier_dir = os.abspath(arg.split('=')[-1])
      classifier_name = classifier_dir.split('/')[-1]
    elif "images=" in arg:
      images = os.path.abspath(arg.split('=')[-1])
  
  # Set the right path to your model definition file, pretrained model 
  # weights, and the image you would like to classify
  MODEL_FILE = ojoin(classifier_dir, classifier_name+'_deploy.prototxt')
  PRETRAINED = ojoin(classifier_dir, 'caffe_reference_imagenet_model')
  MEAN_FILE = ojoin(caffe_root, 'python/caffe/imagenet/ilsvrc_2012_mean.npy')
  IMAGE_FILE = ojoin(caffe_root, 'examples/images/cat.jpg')


# get PRETRAINED
# if not os.path.isfile(PRETRAINED):
#   call(['./get_caffe_reference_imagenet_model.sh'])
  
# load network
print os.getcwd()
net = caffe.Classifier(MODEL_FILE, PRETRAINED,
                       image_dims=(256, 256), input_scale=255,
                       mean_file=MEAN_FILE, channel_swap=(2,1,0))

# set phase to test since we are doing testing
net.set_phase_test()

# use CPU for the computation
# so you can run this on your laptop!
net.set_mode_cpu()

# load image
input_image = caffe.io.load_image(IMAGE_FILE)
# plt.imshow(input_image)

# classify image
# this prints out a ton of numbers, why?
prediction = net.predict([input_image])

# print prediction bar chart
# print 'prediction shape:', prediction[0].shape
# plt.plot(prediction[0])

# print top 5 classes
print 'predicted class:'
class_ = prediction[0].argmax()
print class_#, prediction[class_]

  
# for faster prediction, turn off oversampling
# for even faster prediciton, use GPU mode

# Not as fast as you expected? Indeed, in this python demo you are seeing only 4 times speedup. But remember - the GPU code is actually very fast, and the data loading, transformation and interfacing actually start to take more time than the actual conv. net computation itself!

# To fully utilize the power of GPUs, you really want to:

# Use larger batches, and minimize python call and data transfer overheads.
# Pipeline data load operations, like using a subprocess.
# Code in C++. A little inconvenient, but maybe worth it if your dataset is really, really large.



