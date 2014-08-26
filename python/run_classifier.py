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
# python run_classifier.py classifier-dir=../models/scrape_zone_peel-fine/ data-dir=../data/scrape_zone_peel/ data-info=../data_info/scrape_zone_peel/
# python run_classifier.py classifier-dir=.. data-dir=.. data-info=..

# Note! data-dir should be data/<name>, not data/<name>/test

def main(classifier_dir, data_dir, data_info):
  # this is the test batch size
  # you could set it up as a command line arg if turn out useful
  N = 96

  classifier_name = classifier_dir.split('/')[-1]
  
  # create deploy prototxt
  train_file = get_train_file(classifier_dir)
  # num_imgs = len(os.listdir(oj(data_dir,'test')))
  content = train_file.readlines()
  content = edit_train_content_for_deploy(content)
  write_content_to_deploy_file(classifier_dir, content)
    
  # Set the right path to your model definition file, pretrained model 
  # weights, and the image you would like to classify
  MODEL_FILE = oj(classifier_dir, classifier_name.split('-fine')[0]+'_deploy.prototxt')
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
  imgs,img_fnames = load_all_images_from_dir(oj(data_dir,'test'))

  # classify images
  num_imgs = len(imgs)
  pred = net.predict(imgs[:N])
  # print pred
  for i in range(1,num_imgs/N):
    pred = np.append(pred, net.predict(imgs[i*N:(i+1)*N]),axis=0)
  pred=np.append(pred, net.predict(imgs[-(len(imgs)%N):]),axis=0)

  assert len(pred) == num_imgs

  d = {'fname': img_fnames,
       'pred': pred,
       'label': [],
       'pred_lab': [],
       'pot_mislab': []}
  
  # save predictions to data_info
  # HEY! move this to bottom once fully operational
  np.savez(oj(data_info, PRETRAINED.split('/')[-1]+'_pred.npz'), d)

  # print pred bar chart
  # print 'pred shape:', pred[0].shape
  # plt.plot(pred[0])
  return d


def get_flag_and_thresh(data_info):
  flag_val, thresh = 0, 0.5
  rl = open(oj(data_info,'read.txt'),'r').readlines()
  rl = [l.split() for l in rl]
  for l in rl[2:]:
    if l == ['1','flag_val']: flag_val = 1
    elif l[1] == 'threshold': thresh = float(l[0])
  # if got no thresh to return, means read.txt needs be filled in
  return flag_val, thresh

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
    # if fname.endswith('mean.npy'): return oj(data_dir,fname)
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
  npy_mean_fname = (proto_img_fname.split('_mean.binaryproto')[0]).split('_fine')[0]+'_mean2.npy'
  npy_mean_file = file(oj(data_dir,npy_mean_fname),"wb")
  np.save(npy_mean_file, nparray)
  npy_mean_file.close()
  
  # blob_img = caffe_pb2.BlobProto() # oj(data_dir,proto_img_fname))
  # npy_mean = caffe.io.blobproto_to_array(blob_img)
  # npy_mean_fname = (proto_img_fname.split('_mean.binaryproto')[0]).split('_fine')[0]+'_mean.npy'
  # npy_mean_file = open(,'w')
  # np.save(npy_mean_file, npy_mean)
  # npy_mean_file.close()
  # print 'closed file %s'%(npy_mean_fname)
  return oj(data_dir, npy_mean_fname)

          
def load_all_images_from_dir(test_dir):
  batch = []
  img_fnames = os.listdir(test_dir)
  print 'loading images...'
  for fname in img_fnames:
    batch.append(caffe.io.load_image(oj(test_dir,fname)))
  print 'finished loading images.'
  return batch, img_fnames
    

def fill_dict(d, data_info):
  # this comes early because flag_val prompts user
  flag_val, threshold = get_flag_and_thresh(data_info)

  # get data_info test file
  label_data = open(oj(data_info,'test.txt'),'r').readlines()
  label_data = [line.split() for line in label_data]
  label_data = sorted(label_data, key= lambda x:x[0])
  print 'label_data[] is like', label_data[:3]
  print 'label_data[:,0] is like', label_data[:3][0]
  print 'd[\'fnames\'] is like', d['fname'][:3]
  assert label_data[:][0] == d['fname']
  num_imgs = len(label_data)

  # fill with true labels
  d['label'] = label_f.readlines()[:,1]
  # get threshold
  threshold = float()

  # fill in predicted labels and flag if potentially mislab
  false_pos, num_pos, false_neg, num_neg = 0, 0, 0, 0
  for idx in range(num_imgs):
    # assign predicted label
    if d['pred'][idx][flag_val] >= threshold:
      d['pred_lab'].append(flag_val) 
    else: d['pred_lab'].append(-(flag_val-1))

    # correct classification or not 
    if d['pred_lab'][idx] != d['label'][idx]:
      d['pot_mislab'].append(d['fname'][idx])
      if d['label'][idx] == flag_val:
        false_neg += 1
        num_pos += 1
      else:
        false_pos += 1
        num_neg += 1
    else:
      if d['label'][idx] == flag_val: num_pos += 1
      else: num_neg += 1

  # compute accuracies
  print 'num_pos:', num_pos
  d['accuracy'] = {'total': float((false_neg+false_pos)/num_imgs),
                   'pos': float(false_neg/num_pos),
                   'neg': float(false_pos/num_neg)}
  return d


def compute_kpi(d):
  num_imgs = len(d['fname'])
  flag = get_flag_and_thresh(data_info)[0]
  # create array (idx,prob(pos)) of all positives
  pos = [(idx,d['pred'][flag]) for idx in range(num_imgs)
         if d['label'][idx] == flag]
  print 'check same with above! num_pos:', len(pos)
  
  # sort array descending prob(pos)
  pos = sorted(pos, key=lambda x: x[1])
  
  # Sig_level is prob(pos) for i-th entry where float(i/len) = 0.95
  Sig_level = pos[0.95*num_imgs][1]
  
  # pct_auto is, for all imgs:
  # (num imgs with prob(pos) < Sig_level) / (num imgs)
  automated = [idx for idx in range(num_imgs)
               if d['pred'][idx] < Sig_level]

  return Sig_level, float(len(automated)/num_imgs)



if __name__ == '__main__':
  print 'Warning: make sure that caffe is on the python path!'

  # don't need sig_level, using class imbalance threshold
  # # this is the sig level
  # # you could set it up as a command line arg if turn out useful
  # sig_level = 0.1

  # classifier_dir, images = None, None
  for arg in sys.argv:
    if "classifier-dir=" in arg:
      classifier_dir = os.path.abspath(arg.split('=')[-1])
    elif "data-dir=" in arg:
      data_dir = os.path.abspath(arg.split('=')[-1])
    elif "data-info=" in arg:
      data_info = os.path.abspath(arg.split('=')[-1])
    # elif "train-iter=" in arg:
    #   train_iter = os.path.abspath(arg.split('=')[-1])

  PRETRAINED = get_pretrained_model(classifier_dir)
  already_pred = oj(data_info, PRETRAINED.split('/')[-1]+'_pred.npz')
  if os.path.isfile(already_pred) and raw_input('found %s; use? ([Y]/N) '%(already_pred)) != 'N':
    d = np.load(already_pred)
  else:
    d = main(classifier_dir, data_dir, data_info)

  # HEY! this should go in main as well
  # get true labels, assign predicted labels, get metrics
  d = fill_dict(d, data_info)

  # accuracies
  print 'with threshold at test only:'
  print 'accuracy overall: ', d['accuracy']['total']
  print 'accuracy on positives: ', d['accuracy']['pos']
  print 'overall on negatives: ', d['accuracy']['neg']

  # find highest sig_level that raises >=95% of true positives,
  # and compute % workload that is automated
  Sig_level, pct_auto = compute_kpi(d)

  # for faster prediction, turn off oversampling BUT!
  # you need to set oversampling in edit_train_content_for_deploy to
  # False... so you should probs merge the script into this one
  
  # Not as fast as you expected? Indeed, in this python demo you are seeing only 4 times speedup. But remember - the GPU code is actually very fast, and the data loading, transformation and interfacing actually start to take more time than the actual conv. net computation itself!

  # To fully utilize the power of GPUs, you really want to:

  # Use larger batches, and minimize python call and data transfer overheads.
  # Pipeline data load operations, like using a subprocess.
  # Code in C++. A little inconvenient, but maybe worth it if your dataset is really, really large.
  

