import numpy as np
import os
from os.path import join as ojoin
from PIL import Image
from operator import itemgetter as ig
import json, random


def get_label_dict(data_dir):
  path = data_dir
  d = {'Perfect': []}
  print 'generating dict of label:files from %s...'%(data_dir)
  for filename in os.listdir(path):
    if not filename.endswith('.dat'): continue
    fullname = os.path.join(path, filename)
    with open(fullname) as f:
      content = f.readlines()
      if content == []:
        d['Perfect'].append(filename.split('.')[0]+'.jpg')
      else:
        for label in content:
          if label not in d.keys(): d[label] = []
          d[label].append(filename.split('.')[0]+'.jpg')
  return d


def create_lookup_txtfiles(data_dir, to_dir=None):
  ''' data_dir: where raw data is
      to_dir: where to store .txt files. '''

  case_count = 0  # number of training cases
  tagless_count = 0 # n
  badcase_count = 0 # num of images with multiple flags to train on
  All = get_label_dict(data_dir)
  count = {}

  Keep = classes_to_learn(All)
  # merge_classes only after default label entry created
  Keep = default_class(All, Keep)
  Keep = merge_classes(Keep)
  Keep = rebalance(Keep)
    
  if to_dir is not None:
    train_file = open(ojoin(to_dir,'train.txt'), 'w')
    val_file = open(ojoin(to_dir,'val.txt'), 'w')
    test_file = open(ojoin(to_dir,'test.txt'), 'w')
    read_file = open(ojoin(to_dir,'read.txt'), 'w')


def rebalance(Keep):
  '''prompts user for a new imbalance ratio and implements it. '''
  s = [(key,len(Keep[key])) for key in Keep.keys()]
  minc, maxc = min(s,ig(1))[0], max(s,ig(1))[0]
  target_ratio = raw_input("you have imbalance ratio %.2f, what's your target? [num/N] "%(float(len(Keep[maxc])/len(Keep[minc]))))
  if target_ratio is not 'N':
    minlen = len(Key[minc])
    for key in Keep.keys() if key is not minc:
      print '%s has %i images so %i will be randomly removed'%(key, len(Keep[key]), len(Keep[key])-minlen*target_ratio)
      random.shuffle(Keep[key])
      del Keep[key][minlen*target_ratio:]
  return Keep


def default_class(All, Keep):
  ''' all images without retained labels go to default class. '''
  label_default = raw_input("Default label for all images not containing any of given labels? (name/N) ")
  if label_default is not 'N':
    Keep[label_default] = []
    # below is why need to wait for merge_classes
    for key in All.keys() if key not in Keep.keys():
      # computationally inefficient. but so much more flexible to
      # have this dict.
      Keep[label_default] += [fname in All[key]
                              if fname not in All.values()]
  return Keep

def merge_classes(Keep):
  more = 'Y'
  while more == 'Y':
    print '%s' % (', '.join(map(str,Keep.keys())))
    if raw_input('Merge (more) classes? (Y/N) ') == 'Y':
      merge = [-1]
      while not all([idx < len(Keep.keys()) for idx in merge]):
        for elem in enumerate(Keep.keys()): print elem
        merge = [int(elem) for elem in raw_input("Name two class numbers from above, separated by ' ': ").split()]
      merge.sort()
      merge = [Keep.keys()[i] for i in merge]
      merge_label = raw_input("Name of merged class: ")
      Keep[merge_label] = Keep.pop(merge[1]) + Keep.pop(merge[0])
      count_duplicates = len(Keep[merge_label])-len(set(Keep[merge_label]))
      if count_duplicates > 0:
        print "WARNING! merging these classes has made %i duplicates! Removing them." % (count_duplicates)
        Keep[merge_label] = set(Keep[merge_label])
    else: more = False
  return Keep
  
  
def classes_to_learn(All):    
  ''' prompts user for which labels to use as classes, returns dict
  that is like All but with only required labels . '''
  Keep = {}
  for elem in enumerate(All.keys()): print elem
  read_labels = [All.keys()[int(num)] for num in raw_input("Numbers of labels to learn, separated by ' ': ").split()]
  for label in read_labels:
    Keep[label] = All[label]
  return Keep
  
#### STEP 5.1: SETUP IMBALANCE EXPERIMENT ############################

def rebalance(D, min_ratio, max_ratio, num_nets):
  ''' given a dict containing fnames for each class, a 
  range of imbalance ratios to cover, and a number of nets to train,
  creates num_nets directories, each holding a subdir for each class
  , with max_ratio as imbalance for net_0, ..., min_ratio as 
  imbalance for net_num_nets. '''

  if min_ratio < 1 or max_ratio < 1: 
    print 'Error: ratios must be >=1.'
    exit

  # using cool log calculus, compute la raison de la suite 
  # geometrique donnant les ratios a obtenir pour chaque net.
  step = compute_step(min_ratio, max_ratio, num_nets)

  # move contents of data_dir to a new subdir, 'all'
  if os.path.isdir(ojoin(data_dir,'all')):
    shutil.rmtree(ojoin(data_dir,'all'))
  all_names = os.listdir(data_dir)
  for name in all_names:
    shutil.move(ojoin(data_dir,name), ojoin(data_dir,'all',name))

  # recursively make subdirs for each net, preserving strict set 
  # inclusion from net[i] to net[i+1]
  nets = ['all'] + ['net_'+str(i) for i in range(num_nets)]
  random_delete_recursive(data_dir, step, nets, ratio=2, i=0)
  print 'NOTE: net_0 has highest imbalance ratio.'


def random_delete_recursive(data_dir, step, nets, ratio, i):
#  os.mkdir(ojoin(data_dir,nets[i+1]))
  if os.path.isdir(ojoin(data_dir,nets[i+1])):
    shutil.rmtree(ojoin(data_dir,nets[i+1]))
  shutil.copytree(ojoin(data_dir, nets[i]), 
                  ojoin(data_dir, nets[i+1]), symlinks=True)
  random_delete_aux(ojoin(data_dir, nets[i+1]), ratio)
  if i+2 in range(len(nets)):
    random_delete_recursive(data_dir, step, nets, float(ratio)/step, i+1)


# careful! if you deleted links and now wish to add some back, make 
# sure json dump gets updated/overwritten correctly 
def random_delete_aux(data_dir, ratio):
  ''' randomly deletes as few images from outnumbering class dirs 
      as possible such that #biggest/#smallest == ratio. '''

  data_dir = os.path.abspath(data_dir)
  dump = raw_input('Do you want a json dump in %s of which files were randomly deleted?(Y/any) '%(data_dir))
    
  # D is for dict, d is for directory
  D = {}
  os.chdir(data_dir)
  dirs = [d for d in os.listdir(data_dir) if os.path.isdir(ojoin(data_dir,d))]
  
  print 'the directories are: %s'%(dirs)

  for d in dirs:
    D[d] = {}
    D[d]['total'] = len(os.listdir(ojoin(data_dir,d)))

  dirs = [(d,D[d]['total']) for d in D.keys()]
  dirs = sorted(dirs, key = lambda x: x[1])

  print '%s is smallest class with %i images'%(dirs[0][0],dirs[0][1])
  for d in D.keys():
    D[d]['remove'] = max(0,int(D[d]['total']-(ratio*dirs[0][1])))
    print '%s has %i images so %i will be randomly removed'%(d, D[d]['total'], D[d]['remove'])
    if D[d]['remove'] > 0 :
      D = random_delete_aux2(data_dir,d,D)

  if dump == 'Y': json.dump(D, open(data_dir+'/random_remove_dict.txt','w'))
  return D


# D is for dict, d is for directory
def random_delete_aux2(data_dir,d,D,delete_hard=False):
  D[d]['deleted'] = random.sample(os.listdir(ojoin(data_dir,d)),D[d]['remove'])
  print 'successfully condemned images from %s'%(d)
  back = os.getcwd()
  os.chdir(ojoin(data_dir,d))
  for link in D[d]['deleted']: os.remove(link)
  os.chdir(back)
  return D


def compute_step(min_ratio, max_ratio, num_nets):
  ''' calculates step such that ratio[i+1] = ratio[i]*step for all i,   and such that '''
  return pow(max_ratio/float(min_ratio), 1/float(num_nets-1))

