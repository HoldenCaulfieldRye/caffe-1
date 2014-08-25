import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
import os, sys
import caffe
from caffe.proto import caffe_pb2
from os.path import join as oj
from subprocess import call
from create_deploy_prototxt import *

caffe_root = '../'  # this file is expected to be in {caffe_root}/exampless
sys.path.insert(0, caffe_root + 'python')

# usage:
# python run_classifier.py classifier-dir=../models/scrape_zone_peel-fine/ data-dir=../data/scrape_zone_peel/
# python run_classifier.py classifier-dir=.. data-dir=..

# Note! data-dir should be data/<name>, not data/<name>/test


def get_pretrained_model(classifier_dir):
  suggest = os.listdir(classifier_dir)
  suggest = [fname for fname in suggest
             if 'iter' in fname and 'solverstate' not in fname]
  for elem in enumerate(suggest): print elem
  idx = int(raw_input("\nWhich model? "))
  return oj(classifier_dir,suggest[idx])

#  oj(, 'caffe_reference_imagenet_model')


def get_np_mean_fname(data_dir):
  # for fname in os.listdir(data_dir):
  #   if fname.endswith('mean.npy'): return oj(data_dir,fname)
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
  data = open(oj(data_dir,proto_img_fname), "rb").read()
  blob.ParseFromString(data)
  nparray = caffe.io.blobproto_to_array(blob)[0]
  npy_mean_fname = (proto_img_fname.split('_mean.binaryproto')[0]).split('_fine')[0]+'_mean.npy'
  npy_mean_file = file(oj(data_dir,npy_mean_fname),"wb")
  np.save(npy_mean_file, nparray)
  npy_mean_file.close()
  return oj(data_dir, npy_mean_fname)

          
def initialise_img_dict(test_dir):
  d = {'fnames': os.listdir(test_dir), 'data': []}
  print 'loading images...'
  print '(this is slow, runs on a single core, could be paralellised, or even run on a GPU, it\'s just matrix operations)'
  print 'batching up images...'

  for i in range(len(d['fnames'])/128):
    print 'batch %i:'%(i)
    print d['fnames'][i*128:(i+1)*128]
  print 'last batch:'
  print d['fnames'][-(len(d['fnames'])%128):]
  
  for i in range(len(d['fnames'])/128):
    d['data'].append(np.array([caffe.io.load_image(oj(test_dir,d['fnames'][idx])) for idx in range(i*128,(i+1)*128)]))
    
  d['data'].append(np.array([caffe.io.load_image(oj(test_dir,d['fnames'][-1*(len(d['fnames'])%128):]))]))
  return d
  
  # data = np.array([caffe.io.load_image(oj(test_dir,fname)) for fname in d['fnames']])
  # print 'batching up images...'
  # for i in range(data.size%128):
  #   d['data'].append([data[i*128:(i+1)*128]])
  # d['data'].append([data[(data.size%128)*128:]])
  # return d

  # # noccn code
  # print 'Generating data_batch_%i'%(batch_num)
  # rows = Parallel(n_jobs=self.n_jobs)(
  #   delayed(_process_item)(self, name, symlink)
  #   for name, label in names_and_labels)

  # batch_idx, img_idx = 0, 0
  # while img_idx < len(d['fnames']):
  #   np.append(d['data'], np.array([]))
  #   while img_idx < 128:
  #     np.append(d['data'][batch_idx],caffe.io.load_image(oj(test_dir,d['fnames'][img_idx])))
  #     # d['data'][batch_idx].append(caffe.io.load_image(oj(test_dir,d['fnames'][img_idx])))
  #     img_idx += 1
  #   batch_idx += 1      
  # # for fname in img_fnames:
  # #   d['data'][i].append(caffe.io.load_image(oj(test_dir,fname)))
  # print '%i batches of 128 imgs made to test data'%(len(d['data']))
  # return d
    

def get_predictions(net, img_dict):
  net.predict(batch)

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

  # create deploy prototxt
  train_file = get_train_file(classifier_dir)
  # num_imgs = len(os.listdir(oj(data_dir,'test')))
  content = train_file.readlines()
  content = edit_train_content_for_deploy(content)
  write_content_to_deploy_file(classifier_dir, content)
    
  # Set the right path to your model definition file, pretrained model 
  # weights, and the image you would like to classify
  MODEL_FILE = oj(classifier_dir, classifier_name.split('-fine')[0]+'_deploy.prototxt')
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

  # load image
  # input_image = caffe.io.load_image(images)
  # plt.imshow(input_image)

  # load images
  # parallelise this? use cudaconvnet code
  img_dict = initialise_img_dict(oj(data_dir,'test'))

  # use GPU for the computation
  # so you can run on entire test set
  net.set_mode_gpu()
  print 'Warning: in GPU mode; if your GPU is not CUDA-enabled, this will not work!'
  
  # classify images
  print 'computing predictions...'
  preds = get_predictions(net, img_dict)

  # print prediction bar chart
  # print 'prediction shape:', prediction[0].shape
  # plt.plot(prediction[0])

  # print top 5 classes
  print 'predictions:'
  for idx,img in enumerate(img_fnames):
    print '%s: %s'(img, prediction[idx])
  
    

  # for faster prediction, turn off oversampling BUT!
  # you need to set oversampling in edit_train_content_for_deploy to
  # False... so you should probs merge the script into this one
  
  # Not as fast as you expected? Indeed, in this python demo you are seeing only 4 times speedup. But remember - the GPU code is actually very fast, and the data loading, transformation and interfacing actually start to take more time than the actual conv. net computation itself!

  # To fully utilize the power of GPUs, you really want to:

  # Use larger batches, and minimize python call and data transfer overheads.
  # Pipeline data load operations, like using a subprocess.
  # Code in C++. A little inconvenient, but maybe worth it if your dataset is really, really large.



