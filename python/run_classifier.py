import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
import os, sys
import caffe
from caffe.proto import caffe_pb2
from os.path import join as ojoin
from subprocess import call

caffe_root = '../'  # this file is expected to be in {caffe_root}/examples
sys.path.insert(0, caffe_root + 'python')

# usage:
# python run_classifier.py classifier-dir=.. data-dir=..

# Note! data-dir should be data/<name>, not data/<name>/test


def get_pretrained_model(classifier_dir):
  suggest = os.listdir(classifier_dir)
  print suggest
  suggest = [fname for fname in suggest if 'iter' in fname]
  for elem in enumerate(suggest): print elem
  idx = int(raw_input("\nWhich model? "))
  return ojoin(classifier_dir,suggest[idx])

#  ojoin(, 'caffe_reference_imagenet_model')


def get_np_mean_fname(data_dir):
  for fname in os.listdir(data_dir):
    if fname.endswith('mean.npy'): return ojoin(data_dir,fname)
  proto_img_fname = ''
  for fname in os.listdir(data_dir):
    if fname.endswith('mean.binaryproto'):
      proto_img_fname = fname
      break
  if proto_img_fname == '':
    print 'ERROR: no *mean.npy nor *mean.binaryproto found in %s'%(data_dir)
    sys.exit()

  # er wait how does it know where the proto img file is?
  blob_img = caffe_pb2.BlobProto() # ojoin(data_dir,proto_img_fname))
  npy_mean = caffe.io.blobproto_to_array(blob_img)
  npy_mean_fname = (proto_img_fname.split('_mean.binaryproto')[0]).split('-fine')[0]
  npy_mean_file = open(ojoin(data_dir,npy_mean_fname),'w')
  np.save(npy_mean_file, npy_mean)
  return ojoin(data_dir, npy_mean_fname)

          
def load_all_images_from_dir(test_dir):
  batch = []
  img_fnames = os.listdir(test_dir)
  for fname in img_fnames:
    batch.append(caffe.io.load_image(ojoin(test_dir,fname)))
  return batch, img_fnames
    


if __name__ == '__main__':
  print 'Warning: make sure that caffe is on the python path!'

  classifier_dir, images = None, None
  for arg in sys.argv:
    if "classifier-dir=" in arg:
      classifier_dir = os.path.abspath(arg.split('=')[-1])
      classifier_name = classifier_dir.split('/')[-1]
    elif "data-dir=" in arg:
      data_dir = os.path.abspath(arg.split('=')[-1])
    # elif "train-iter=" in arg:
    #   train_iter = os.path.abspath(arg.split('=')[-1])
  
  # Set the right path to your model definition file, pretrained model 
  # weights, and the image you would like to classify
  MODEL_FILE = ojoin(classifier_dir, classifier_name.split('-fine')[0]+'_deploy.prototxt')
  PRETRAINED = get_pretrained_model(classifier_dir)
  MEAN_FILE = get_np_mean_fname(data_dir)


  # get PRETRAINED
  # if not os.path.isfile(PRETRAINED):
  #   call(['./get_caffe_reference_imagenet_model.sh'])

  # load network
  net = caffe.Classifier(MODEL_FILE, PRETRAINED,
                         image_dims=(256, 256), input_scale=255,
                         mean_file=MEAN_FILE, channel_swap=(2,1,0))

  # set phase to test since we are doing testing
  net.set_phase_test()

  # use CPU for the computation
  # so you can run this on your laptop!
  # net.set_mode_cpu()

  # use GPU for the computation
  # so you can run on entire test set
  net.set_mode_gpu()
  
  # load image
  # input_image = caffe.io.load_image(images)
  # plt.imshow(input_image)

  # load images
  # parallelise this? use cudaconvnet code
  img_batch,img_fnames = load_all_images_from_dir(ojoin(data_dir,test))

  # classify images
  prediction = net.predict(img_batch)

  # print prediction bar chart
  # print 'prediction shape:', prediction[0].shape
  # plt.plot(prediction[0])

  # print top 5 classes
  print 'predictions:'
  for idx,img in enumerate(img_fnames):
    print '%s: %s'(img, prediction[idx])
  


  # for faster prediction, turn off oversampling
  # for even faster prediciton, use GPU mode

  # Not as fast as you expected? Indeed, in this python demo you are seeing only 4 times speedup. But remember - the GPU code is actually very fast, and the data loading, transformation and interfacing actually start to take more time than the actual conv. net computation itself!

  # To fully utilize the power of GPUs, you really want to:

  # Use larger batches, and minimize python call and data transfer overheads.
  # Pipeline data load operations, like using a subprocess.
  # Code in C++. A little inconvenient, but maybe worth it if your dataset is really, really large.



