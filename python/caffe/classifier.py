#!/usr/bin/env python
"""
Classifier is an image classifier specialization of Net.
"""

import numpy as np

import caffe


class Classifier(caffe.Net):
  """
  Classifier extends Net for image class prediction
  by scaling, center cropping, or oversampling.
  """
  # input_scale aka raw_scale
  def __init__(self, model_file, pretrained_file,
               image_dims=None, input_scale=None, 
               mean_file=None, channel_swap=None,
               gpu=False):
    """
    Take
    image_dims: dimensions to scale input for cropping/sampling.
                Default is to scale to net input size for whole-image crop.
    gpu, mean_file, input_scale, channel_swap: convenience params for
        setting mode, mean, input scale, and channel order.
    """
    # call __init__ on the superclass
    caffe.Net.__init__(self, model_file, pretrained_file)
    print 'caffe.Net.__init__() successful'
    self.set_phase_test()
    # print 'a'
    
    if gpu:
      self.set_mode_gpu()
    else:
      self.set_mode_cpu()

    # print 'b'
    if mean_file:
      # print 'self.inputs[0]: %s'%(self.inputs[0])
      # self.inputs is array of blob names from the net that we want
      # in current case, just data, ie first blob
      self.set_mean(self.inputs[0], mean_file) # bug here; wtf is self.inputs ??
      # print 'c'
      if input_scale:
        # print 'd'
        self.set_input_scale(self.inputs[0], input_scale)
        if channel_swap:
          # print 'e'
          self.set_channel_swap(self.inputs[0], channel_swap)

    # print 'f'
    self.crop_dims = np.array(self.blobs[self.inputs[0]].data.shape[2:])
    # print 'g'
    if not image_dims:
      image_dims = self.crop_dims
    self.image_dims = image_dims


  def predict(self, inputs, oversample=True):
    """
    Predict classification probabilities of inputs.

    Take
    inputs: iterable of (H x W x K) input ndarrays.
    oversample: average predictions across center, corners, and mirrors
                when True (default). Center-only prediction when False.

    Give
    predictions: (N x C) ndarray of class probabilities
                 for N images and C classes.
    """
    # Scale to standardize input dimensions.
    inputs = np.asarray([caffe.io.resize_image(im, self.image_dims)
                         for im in inputs])

    if oversample:
      # Generate center, corner, and mirrored crops.
      inputs = caffe.io.oversample(inputs, self.crop_dims)
    else:
      # Take center crop.
      center = np.array(self.image_dims) / 2.0
      crop = np.tile(center, (1, 2))[0] + np.concatenate([
        -self.crop_dims / 2.0,
        self.crop_dims / 2.0
      ])
      inputs = inputs[:, crop[0]:crop[2], crop[1]:crop[3], :]

    # Classify
    print 'classifier::predict: preprocessing images...'
    caffe_in = np.asarray([self.preprocess(self.inputs[0], in_)
                           for in_ in inputs])
    print 'finished preprocessing images.'
    out = self.forward_all(**{self.inputs[0]: caffe_in})
    predictions = out[self.outputs[0]].squeeze(axis=(2,3))

    # For oversampling, average predictions across crops.
    if oversample:
      predictions = predictions.reshape((len(predictions) / 10, 10, -1))
      predictions = predictions.mean(1)

    return predictions
