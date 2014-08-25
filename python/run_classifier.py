import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
import os, sys
import caffe
from caffe.proto import caffe_pb2
from os.path import join as ojoin
from subprocess import call
from create_deploy_prototxt import *

caffe_root = '../'  # this file is expected to be in {caffe_root}/exampless
sys.path.insert(0, caffe_root + 'python')

# usage:
# python run_classifier.py classifier-dir=../models/scrape_zone_peel-fine/ data-dir=../data/scrape_zone_peel/ data-info=../data_info/scrape_zone_peel/
# python run_classifier.py classifier-dir=.. data-dir=.. data-info=..

# Note! data-dir should be data/<name>, not data/<name>/test

def get_flag_value(data_info):
  print open(oj(data_info,'read.txt'),'r').readlines())
  return int(raw_input('1 or 0 corresponds to flag? '))

def get_pretrained_model(classifier_dir):
  suggest = os.listdir(classifier_dir)
  suggest = [fname for fname in suggest
             if 'iter' in fname and 'solverstate' not in fname]
  for elem in enumerate(suggest): print elem
  idx = int(raw_input("\nWhich model? "))
  return ojoin(classifier_dir,suggest[idx])

#  ojoin(, 'caffe_reference_imagenet_model')


def get_np_mean_fname(data_dir):
  # for fname in os.listdir(data_dir):
    # if fname.endswith('mean.npy'): return ojoin(data_dir,fname)
  proto_img_fname = ''
  for fname in os.listdir(data_dir):
    if fname.endswith('mean.binaryproto'):
      print 'found binaryproto: %s'%(fname)
      proto_img_fname = fname
      break
  if proto_img_fname == '':
    print 'ERROR: no *mean.npy nor *mean.binaryproto found in %s'%(data_dir)
    sys.exit()

  # er wait how does it know where the proto img file is?
  blob = caffe_pb2.BlobProto()
  data = open(ojoin(data_dir,proto_img_fname), "rb").read()
  blob.ParseFromString(data)
  nparray = caffe.io.blobproto_to_array(blob)[0]
  npy_mean_fname = (proto_img_fname.split('_mean.binaryproto')[0]).split('_fine')[0]+'_mean2.npy'
  npy_mean_file = file(ojoin(data_dir,npy_mean_fname),"wb")
  np.save(npy_mean_file, nparray)
  npy_mean_file.close()
  
  # blob_img = caffe_pb2.BlobProto() # ojoin(data_dir,proto_img_fname))
  # npy_mean = caffe.io.blobproto_to_array(blob_img)
  # npy_mean_fname = (proto_img_fname.split('_mean.binaryproto')[0]).split('_fine')[0]+'_mean.npy'
  # npy_mean_file = open(,'w')
  # np.save(npy_mean_file, npy_mean)
  # npy_mean_file.close()
  # print 'closed file %s'%(npy_mean_fname)
  return ojoin(data_dir, npy_mean_fname)

          
def load_all_images_from_dir(test_dir):
  batch = []
  img_fnames = os.listdir(test_dir)
  print 'loading images...'
  for fname in img_fnames:
    batch.append(caffe.io.load_image(ojoin(test_dir,fname)))
  print 'finished loading images.'
  return batch, img_fnames
    

def assign_flags(d, data_info, sig_level, flag_val):
  # get data_info test file
  label_data = open(oj(data_info,'test.txt'),'r').readlines()
  assert label_data[:,0] == d['fnames']
  d['label'] = label_f.readlines()[:,1]
  for idx in range(len(label_data)):
    if d['pred'][idx][flag_val] >= sig_level:
      d['flags'].append(flag_val)


if __name__ == '__main__':
  print 'Warning: make sure that caffe is on the python path!'

  # this is the test batch size
  # you could set it up as a command line arg if turn out useful
  N = 96

  # this is the sig level
  # you could set it up as a command line arg if turn out useful
  sig_level = 0.1

  flag_val = get_flag_value(data_info)
  
  classifier_dir, images = None, None
  for arg in sys.argv:
    if "classifier-dir=" in arg:
      classifier_dir = os.path.abspath(arg.split('=')[-1])
      classifier_name = classifier_dir.split('/')[-1]
    elif "data-dir=" in arg:
      data_dir = os.path.abspath(arg.split('=')[-1])
    elif "data-info=" in arg:
      data_info = os.path.abspath(arg.split('=')[-1])
    # elif "train-iter=" in arg:
    #   train_iter = os.path.abspath(arg.split('=')[-1])

  # create deploy prototxt
  train_file = get_train_file(classifier_dir)
  # num_imgs = len(os.listdir(ojoin(data_dir,'test')))
  content = train_file.readlines()
  content = edit_train_content_for_deploy(content)
  write_content_to_deploy_file(classifier_dir, content)
    
  # Set the right path to your model definition file, pretrained model 
  # weights, and the image you would like to classify
  MODEL_FILE = ojoin(classifier_dir, classifier_name.split('-fine')[0]+'_deploy.prototxt')
  PRETRAINED = get_pretrained_model(classifier_dir)
  MEAN_FILE = get_np_mean_fname(data_dir)


  # get PRETRAINED
  # if not os.path.isfile(PRETRAINED):
  #   call(['./get_caffe_reference_imagenet_model.sh'])

  # load network
  print 'loading network...'
  net = caffe.Classifier(MODEL_FILE, PRETRAINED,
                         image_dims=(256, 256), input_scale=255,
                         mean_file=MEAN_FILE, channel_swap=(2,1,0))
  # flow of control:
  #   classifier::__init__(
  #   classifier::caffe.Net.__init__()
  print 'network loaded successfully'

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
  img_batch,img_fnames = load_all_images_from_dir(ojoin(data_dir,'test'))

  # classify images
  pred = net.predict(img_batch[:N])
  # print pred
  for i in range(1,len(img_batch)/N):
    pred = np.append(pred, net.predict(img_batch[i*N:(i+1)*N]),axis=0)
  pred=np.append(pred, net.predict(img_batch[-(len(img_batch)%N):]),axis=0)

  
  # print pred bar chart
  # print 'pred shape:', pred[0].shape
  # plt.plot(pred[0])

  # print top 5 classes
  assert len(preds) == len(img_batch)
  # for idx,img_name in enumerate(img_fnames):
  #   print '%s: %s'(img_fnames[idx], pred[idx])
  d = {'fname': img_fnames,
       'pred': pred,
       'label': [],
       'flag': [],}
  assign_flags(d,data_info, sig_level, flag_val)

  # for faster prediction, turn off oversampling BUT!
  # you need to set oversampling in edit_train_content_for_deploy to
  # False... so you should probs merge the script into this one
  
  # Not as fast as you expected? Indeed, in this python demo you are seeing only 4 times speedup. But remember - the GPU code is actually very fast, and the data loading, transformation and interfacing actually start to take more time than the actual conv. net computation itself!

  # To fully utilize the power of GPUs, you really want to:

  # Use larger batches, and minimize python call and data transfer overheads.
  # Pipeline data load operations, like using a subprocess.
  # Code in C++. A little inconvenient, but maybe worth it if your dataset is really, really large.



