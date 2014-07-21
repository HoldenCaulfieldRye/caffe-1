import numpy as np
import os
from os.path import join as ojoin
from PIL import Image
from operator import itemgetter as ig
from itertools import chain
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
  All = get_label_dict(data_dir)
  Keep = classes_to_learn(All)
  # merge_classes only after default label entry created
  Keep = default_class(All, Keep)
  Keep = merge_classes(Keep)
  Keep = shuffle_and_rebalance(Keep)
  if to_dir is not None:
    dump_to_files(Keep)
  return Keep


def dump_to_files(Keep, to_dir):
  dump_fnames = ['train.txt','val.txt','test.txt']
  part = [0, 0.8, 0.87, 1] # partition into train val test
  for i in xrange(3):
    dfile = open(ojoin(to_dir,dump_fnames[i]),'w')
    dump = []
    for (num,key) in enumerate(Keep.keys()):
      l = len(Keep[key])
      dump += ["%s %i\n" % (f,num) for f in
               Keep[key][part[i]*l:part[i+1]*l]]
    random.shuffle(dump)
    dfile.writelines(dump)
    dfile.close()
    
  # write to read file how to interpret values as classes      
  read_file = open(ojoin(to_dir,'read.txt'), 'w')    
  read_file.writelines(["%i %s\n" % (num,label) for (num, label)
                         in enumerate(Keep.keys())])
  read_file.close()

    
def shuffle_and_rebalance(Keep):
  '''prompts user for a new imbalance ratio and implements it. '''
  s = [(key,len(Keep[key])) for key in Keep.keys()]
  minc, maxc = min(s,ig(1))[0], max(s,ig(1))[0]
  target_ratio = raw_input("you have imbalance ratio %.2f, what's your target? [num/N] "%(float(len(Keep[maxc])/len(Keep[minc]))))
  if target_ratio is not 'N':
    minlen = len(Key[minc])
    for key in Keep.keys():
      random.shuffle(Keep[key])
      if key is not minc:
        print '%s has %i images so %i will be randomly removed'%(key, len(Keep[key]), len(Keep[key])-minlen*target_ratio)
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
      Keep[label_default] += [fname in All[key] if fname not in
                              set(chain(All.values()))]
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
  
if __name__ == '__main__':
  import sys
  create_lookup_txtfiles(sys.argv[1],sys.argv[2])

