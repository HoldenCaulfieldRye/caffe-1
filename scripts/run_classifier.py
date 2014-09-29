import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
import os, sys, shutil
import caffe
import check
import yaml
from caffe.proto import caffe_pb2
from os.path import join as oj
from subprocess import call
from create_deploy_prototxt import *

caffe_root = '../'  # this file is expected to be in {caffe_root}/exampless
sys.path.insert(0, caffe_root + 'python')

# usage:
# python run_classifier.py classifier-dir=../models/ground_sheet-fine data-dir=../data/ground_sheet_3501 data-info=../data_info/ground_sheet_3501

# Note! data-dir should be data/<name>, not data/<name>/test

def main(classifier_dir, data_dir, data_info):
  N = 96
  classifier_name = classifier_dir.split('/')[-1]  
  if len([fname for fname in os.listdir(classifier_dir) 
          if fname == classifier_name.split('-fine')[0]+'_deploy.prototxt']) == 0:
    train_file = get_train_file(classifier_dir)
    content = train_file.readlines()
    content = edit_train_content_for_deploy(content)
    write_content_to_deploy_file(classifier_dir, content)
    
  MODEL_FILE = oj(classifier_dir, classifier_name.split('-fine')[0]+'_deploy.prototxt')
  MEAN_FILE = get_np_mean_fname(data_dir)
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
  net.set_mode_gpu()
  d = {'fname': [],
       'pred': [],
       'time': [],
       'dude': [],
       'label': [],
       'pred_lab_thresh': [],
       'pred_lab_std': [],
       'pot_mislab': []}
  # load images
  imgs,d['fname'],d['time'],d['dude'] =  load_all_images_from_dir(oj(data_dir,'test'))
  suggest = os.listdir(classifier_dir)
  suggest = [fname for fname in suggest
             if 'iter' in fname and 'solverstate' not in fname]
  for elem in enumerate(suggest): print elem
  idx = int(raw_input("\nWhich model? "))
  return oj(classifier_dir,suggest[idx])


def get_np_mean_fname(data_dir):
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
  return oj(data_dir, npy_mean_fname)

          
def load_all_images_from_dir(test_dir):
  batch, times, dudes = [], [], []
  img_fnames = os.listdir(test_dir)
  print 'loading images...'
  # d_multJoints is a dict: fname -> joint_name
  d_multJoints = create_dict_jname()
  for fname in img_fnames:
    full_fname = oj(test_dir, fname)
    batch.append(caffe.io.load_image(full_fname))
    time,dude = get_(fname,['CreatedTime','InspectedBy'])
    times.append(time)
    dudes.append(dude)
  print 'finished loading images.'
  return batch, img_fnames

def create_dict_jname():
  file_multJoints = '/data/ad6813/pipe-data/Redbox/multJoints.txt'
  data_dir = '/data/ad6813/pipe-data/Redbox/raw_data/dump'
  multJoints = {}
  for line in open(file_multJoints,'r').readlines():
    for img in line.split()[1:]:
      multJoints[img+'.jpg'] = line.split()[0]
  return multJoints
  
def get_(fname,what):
  ret = []
  meta_name = fname.split('.')[0] + '.met'
  data_dir = '/data/ad6813/pipe-data/Redbox/raw_data/dump'
  for f in os.listdir(data_dir):
    if f == meta_name:
      for line in open(f,'r').readlines():
        for field in what:
          if line.startswith(field):
            ret.append(line.split(field+'=')[-1].split()[0])
  return ret
  
                 
def fill_dict(d, data_info):
  # this comes early because flag_val prompts user
  flag_val, threshold = get_flag_and_thresh(data_info)

  # get data_info test file
  label_data = open(oj(data_info,'test.txt'),'r').readlines()
  label_data = [line.split() for line in label_data]
  label_data = sorted(label_data, key= lambda x:x[0])
  assert d['fname'] == [el[0] for el in label_data]
  num_imgs = len(label_data)
  # fill with true labels
  d['label'] = [int(el[1]) for el in label_data]
  # fill in predicted labels and flag if potentially mislab
  false_pos_thresh, num_pos, false_neg_thresh, num_neg, false_neg_std, false_pos_std = 0, 0, 0, 0, 0, 0
  for idx in range(num_imgs):
    # assign predicted label wrt threshold
    if d['pred'][idx][flag_val] >= threshold:
      d['pred_lab_thresh'].append(flag_val) 
    else: d['pred_lab_thresh'].append(-(flag_val-1))
    # assign predicted label in std way
    if d['pred'][idx][flag_val] >= 0.5:
      d['pred_lab_std'].append(flag_val) 
    else: d['pred_lab_std'].append(-(flag_val-1))
    # correct thresh classification or not 
    if d['pred_lab_thresh'][idx] != d['label'][idx]:
      if d['label'][idx] == flag_val:
        false_neg_thresh += 1
        num_pos += 1
      else:
        false_pos_thresh += 1
        num_neg += 1
    else:
      if d['label'][idx] == flag_val: num_pos += 1
      else: num_neg += 1

    # correct std classification or not 
    if d['pred_lab_std'][idx] != d['label'][idx]:
      d['pot_mislab'].append(idx)
      if d['label'][idx] == flag_val: false_neg_std += 1
      else: false_pos_std += 1

  print 'false_neg_thresh: %i, false_pos_thresh: %i'%(false_neg_thresh,false_pos_thresh)
  print 'false_neg_std: %i, false_pos_std: %i'%(false_neg_std,false_pos_std)
  print 'num_neg: %i, num_pos: %i'%(num_neg,num_pos)
  # compute accuracies
  d['accuracy']= {}
  d['accuracy']['total_thresh'] = 1-(false_neg_thresh+false_pos_thresh)/float(num_imgs)
  d['accuracy']['pos_thresh'] = 1-false_neg_thresh/float(num_pos)
  d['accuracy']['neg_thresh'] = 1-false_pos_thresh/float(num_neg)
  d['accuracy']['total_std'] = 1-(false_neg_std+false_pos_std)/float(num_imgs)
  d['accuracy']['pos_std'] = 1-false_neg_std/float(num_pos)
  d['accuracy']['neg_std'] = 1-false_pos_std/float(num_neg)
  print "d['accuracy']", d['accuracy']
  return d


def compute_kpi(d):
  num_imgs = len(d['fname'])
  flag = get_flag_and_thresh(data_info)[0]
  # create array (idx,prob(pos)) of all positives
  pos = [(idx,float(d['pred'][idx][flag]))
         for idx in range(num_imgs)
         if d['label'][idx] == flag]
  print 'check same with above! num_pos:', len(pos)
  # sort array descending prob(pos)
  pos = sorted(pos, key=lambda x: x[1], reverse=True)
  # Sig_level is prob(pos) for i-th entry where float(i/len) = 0.95
  print 'sig levels required for following accuracy on positives:'
  print '70\%:',pos[int(0.7*len(pos))][1]
  print '80\%:',pos[int(0.8*len(pos))][1]
  print '90\%:',pos[int(0.9*len(pos))][1]
  print '95\%:',pos[int(0.95*len(pos))][1]
  Sig_level = float(pos[int(0.95*len(pos))][1])
  automated = [idx for idx in range(num_imgs)
               if float(d['pred'][idx][flag]) < Sig_level]
  return Sig_level, len(automated)/float(num_imgs)


if __name__ == '__main__':
  print 'Warning: make sure that caffe is on the python path!'
  for arg in sys.argv:
    if "classifier-dir=" in arg:
      classifier_dir = os.path.abspath(arg.split('=')[-1])
    elif "data-dir=" in arg:
      data_dir = os.path.abspath(arg.split('=')[-1])
    elif "data-info=" in arg:
      data_info = os.path.abspath(arg.split('=')[-1])

  if check.check(data_dir, data_info) != [0,0]:
    print 'ERROR: mismatch between test files in data_dir and data_info'
    sys.exit()

  PRETRAINED = get_pretrained_model(classifier_dir)
  already_pred = oj(data_info, PRETRAINED.split('/')[-1]+'_pred.npy')
  if os.path.isfile(already_pred) and raw_input('found %s; use? ([Y]/N) '%(already_pred)) != 'N':
    d = (np.load(already_pred)).item()
  else:
    d = main(classifier_dir, data_dir, data_info)

  # this should go in main as well?
  # get true labels, assign predicted labels, get metrics
  d = fill_dict(d, data_info)

  # potential mislabels
  mislab_dir = oj(data_info,'potential_mislabels_'+PRETRAINED.split('/')[-1])
  try: os.mkdir(mislab_dir)
  except:
    shutil.rmtree(mislab_dir)
    os.mkdir(mislab_dir)
  for idx in d['pot_mislab']:
    shutil.copy(oj(data_dir,'test',d['fname'][idx]), mislab_dir)
  print "saving potential mislabels to %s"%(mislab_dir)

  # accuracies
  print 'with threshold at test only:'
  print 'accuracy overall: ', d['accuracy']['total_thresh']
  print 'accuracy on positives: ', d['accuracy']['pos_thresh']
  print 'accuracy on negatives: ', d['accuracy']['neg_thresh']

  print 'with standard 0.5 classification:'
  print 'accuracy overall: ', d['accuracy']['total_std']
  print 'accuracy on positives: ', d['accuracy']['pos_std']
  print 'accuracy on negatives: ', d['accuracy']['neg_std']
  
  # find highest sig_level that raises >=95% of true positives,
  # and compute % workload that is automated
  Sig_level, pct_auto = compute_kpi(d)
  print 'sig level required for 95% accuracy on positives:', Sig_level
  print 'this enables', pct_auto, 'automation'

  
  # for faster prediction, turn off oversampling BUT!
  # you need to set oversampling in edit_train_content_for_deploy to
  # False... so you should probs merge the script into this one
  
  # To fully utilize the power of GPUs, you really want to:

  # Use larger batches, and minimize python call and data transfer overheads.
  # Pipeline data load operations, like using a subprocess.
  # Code in C++. A little inconvenient, but maybe worth it if your dataset is really, really large.
  

