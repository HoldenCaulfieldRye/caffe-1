import numpy as np
import os
from os.path import join as ojoin
from PIL import Image
from operator import itemgetter as ig
from itertools import chain
from datetime import date
import json, yaml, random


def get_label_dict(data_dir):
  total_num_images = 0
  path = data_dir
  for fname in os.listdir(os.getcwd()):
    if not fname.startswith('label_dict'): continue
    else:
      if raw_input('found %s; use as label_dict? ([Y]/N) '%(fname)) in ['','Y']:
        return yaml.load(open(fname,'r'))
  d = {'Perfect': []}
  print 'generating dict of label:files from %s...'%(data_dir)
  for filename in os.listdir(path):
    if not filename.endswith('.dat'): continue
    total_num_images += 1
    fullname = os.path.join(path, filename)
    with open(fullname) as f:
      content = [line.strip() for line in f.readlines()] 
      if content == []:
        d['Perfect'].append(filename.split('.')[0]+'.jpg')
      else:
        for label in content:
          if label not in d.keys(): d[label] = []
          d[label].append(filename.split('.')[0]+'.jpg')
  d['total_num_images'] = total_num_images
  json.dump(d, open('label_dict_'+str(date.today()),'w'))
  return d


def create_lookup_txtfiles(data_dir, target_bad_min=None, to_dir=None):
  '''data_dir: where raw data is. to_dir: where to store .txt 
  files. '''
  All = get_label_dict(data_dir)
  total_num_images = All.pop('total_num_images')
  Keep = classes_to_learn(All)
  # merge_classes only after default label entry created
  Keep = default_class(All, Keep)
  total_num_check = sum([len(Keep[key]) for key in Keep.keys()])
  if total_num_images <> total_num_check:
    print "WARNING! started off with %i images, now have %i distinct training cases"%(total_num_images, total_num_check)
  Keep, num_output = merge_classes(Keep)
  Keep = rebalance(Keep, total_num_images, target_bad_min)
  print 'finished rebalancing'
  Keep = within_class_shuffle(Keep)
  print 'finished shuffling'
  if to_dir is not None:
    dump_to_files(Keep, to_dir)
  print 'num_output:', num_output
  return num_output


def dump_to_files(Keep, to_dir):
  dump_fnames = ['train.txt','val.txt','test.txt']
  part = [0, 0.8, 0.87, 1] # partition into train val test
  for i in xrange(3):
    dfile = open(ojoin(to_dir,dump_fnames[i]),'w')
    dump = []
    for (num,key) in enumerate(Keep.keys()):
      l = len(Keep[key])
      dump += ["%s %i\n" % (f,num) for f in
               Keep[key][int(part[i]*l):int(part[i+1]*l)]]
    random.shuffle(dump)
    dfile.writelines(dump)
    dfile.close()
    print 'closed', dump_fnames[i]
    
  # write to read file how to interpret values as classes      
  read_file = open(ojoin(to_dir,'read.txt'), 'w')    
  read_file.writelines(["%i %s\n" % (num,label) for (num, label)
                         in enumerate(Keep.keys())])
  read_file.close()

    
def rebalance(Keep, total_num_images, target_bad_min=None):
  '''if target_bad_min not given, prompts user for one; 
  and implements it. Note that with >2 classes, this can be 
  implemented either by downsizing all non-minority classes by the
  same factor in order to maintain their relative proportions, or 
  by downsizing as few majority classes as possible until
  target_bad_min achieved. We can assume that we care mostly about 
  having as few small classes as possible, so the latter is 
  implemented.'''
  # minc is class with minimum number of training cases
  ascending_classes = sorted([(key,len(Keep[key]))
                              for key in Keep.keys()],
                             key=lambda x:x[1])
  maxc, len_maxc = ascending_classes[-1][0], ascending_classes[-1][1]
  print "max class is %s with %i images"%(maxc, len_maxc)
  print "total num images: %i"%(total_num_images)
  maxc_proportion = float(len_maxc)/total_num_images
  if target_bad_min is None:
    target_bad_min = raw_input("max class currently takes up %.2f, what's your target? [num/N] "%(maxc_proportion))
  if target_bad_min is not 'N':
    target_bad_min = float(target_bad_min)
    print 'maxc_proportion: %.2f, target_bad_min: %.2f'%(maxc_proportion, target_bad_min)
    if maxc_proportion > target_bad_min:
      delete_size = len_maxc - int(target_bad_min*total_num_images/(1+target_bad_min))
      random.shuffle(Keep[maxc])
      print '%s has %i images so %i will be randomly removed'%(maxc, len_maxc, delete_size)
      del Keep[maxc][:delete_size]
  return Keep


def default_class(All, Keep):
  ''' all images without retained labels go to default class. '''
  label_default = raw_input("Default label for all images not containing any of given labels? (name/N) ")
  if label_default is not 'N':
    Keep[label_default] = All['Perfect']
    # no need to check for overlap between Perfect and Keep's other
    # labels because Perfect overlaps with no other label by def
    # below is why need to wait for merge_classes
    for key in All.keys():
      if key in Keep.keys()+['Perfect']: continue
      else:
        # computationally inefficient. but so much more flexible to
        # have this dict.
        # add fname if not in any
        # ---
        # updating 'already' is the expensive bit. must do it no less
        # than after every key iteration because mutual exclusiveness
        # between keys not guaranteed. no need to do it more freq
        # because a key contains no duplicates
        already = set(chain(*Keep.values()))
        print "%s getting images from %s..."%(label_default,key)
        Keep[label_default] += [fname for fname in All[key] if fname
                                not in already]
  return Keep


def merge_classes(Keep):
  more = 'Y'
  while more == 'Y':
    print '%s' % (', '.join(map(str,Keep.keys())))
    if raw_input('Merge (more) classes? (Y/N) ') == 'Y':
      merge = [10000]
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
  return Keep, len(Keep.keys())
  
  
def classes_to_learn(All):    
  ''' prompts user for which labels to use as classes, returns dict
  that is like All but with only required labels . '''
  Keep = {}
  for elem in enumerate(All.keys()): print elem
  read_labels = [All.keys()[int(num)] for num in raw_input("Numbers of labels to learn, separated by ' ': ").split()]
  # if 'Perfect' in All.keys():
  #   Keep['Perfect'] = All['Perfect']
  for label in read_labels:
    Keep[label] = All[label]
  return Keep


def within_class_shuffle(Keep):
  ''' randomly shuffles the ordering of Keep[key] for each key. '''
  for key in Keep.keys():
    random.shuffle(Keep[key])
  return Keep


if __name__ == '__main__':
  import sys
  
  target_bad_min, data_dir, to_dir = None, None, None
  for arg in sys.argv:
    if "bad-min=" in arg:
      target_bad_min = float(arg.split('=')[-1])
    elif "data-dir=" in arg:
      data_dir = arg.split('=')[-1]
    elif "to-dir=" in arg:
      to_dir = arg.split('=')[-1]

  if data_dir is None:
    print "ERROR: data_dir not given"
    exit
      
  num_output = create_lookup_txtfiles(data_dir, target_bad_min, to_dir)
  print "It's going to say 'An exception has occured etc' but don't worry, that's just information for the training shell script to use"
  sys.exit(num_output)
