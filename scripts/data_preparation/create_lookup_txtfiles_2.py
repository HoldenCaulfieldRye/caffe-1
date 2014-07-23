import numpy as np
import os
from os.path import join as ojoin
from PIL import Image
from operator import itemgetter as ig
from itertools import chain
from datetime import date
import json, random


def get_label_dict(data_dir):
  path = data_dir
  for fname in os.listdir(os.getcwd()):
    if not fname.startswith('label_dict'): continue
    else:
      if raw_input('found %s; use as label_dict? ([Y]/N) '%(fname)) in ['','Y']:
        return json.load(open(fname,'r'))
  d = {'Perfect': []}
  print 'generating dict of label:files from %s...'%(data_dir)
  for filename in os.listdir(path):
    if not filename.endswith('.dat'): continue
    fullname = os.path.join(path, filename)
    with open(fullname) as f:
      content = [line.strip() for line in f.readlines()] 
      if content == []:
        d['Perfect'].append(filename.split('.')[0]+'.jpg')
      else:
        for label in content:
          if label not in d.keys(): d[label] = []
          d[label].append(filename.split('.')[0]+'.jpg')
  json.dump(d, open('label_dict_'+str(date.today()),'w'))        
  return d


def create_lookup_txtfiles(data_dir, target_ratio=None, to_dir=None):
  ''' data_dir: where raw data is
      to_dir: where to store .txt files. '''
  All = get_label_dict(data_dir)
  Keep = classes_to_learn(All)
  # merge_classes only after default label entry created
  Keep = default_class(All, Keep)
  Keep, num_output = merge_classes(Keep)
  Keep = shuffle_and_rebalance(Keep, target_ratio)
  if to_dir is not None:
    dump_to_files(Keep, to_dir)
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
    
  # write to read file how to interpret values as classes      
  read_file = open(ojoin(to_dir,'read.txt'), 'w')    
  read_file.writelines(["%i %s\n" % (num,label) for (num, label)
                         in enumerate(Keep.keys())])
  read_file.close()

    
def shuffle_and_rebalance(Keep, target_ratio=None):
  '''if target_ratio not given, prompts user for a new imbalance 
  ratio; and implements it. '''
  s = [(key,len(Keep[key])) for key in Keep.keys()]
  ordered = min(s,ig(1))
  # minc is class with minimum number of training cases
  minc, maxc = ordered[-1][0], ordered[0][0]
  if target_ratio is None:
    target_ratio = raw_input("you have imbalance ratio %.2f, what's your target? [num/N] "%(float(len(Keep[maxc])/len(Keep[minc]))))
  if target_ratio is not 'N':
    target_ratio = float(target_ratio)
    minlen = len(Keep[minc])
    print "min class is %s with %i images, so other classes will have at most %i images, max class is %s"%(minc, minlen, int(minlen*target_ratio), maxc)
    for key in Keep.keys():
      random.shuffle(Keep[key])
      if key is not minc:
        delete_size = max(0, len(Keep[key])-int(minlen*target_ratio))
        print '%s has %i images so %i will be randomly removed'%(key, len(Keep[key]), delete_size)
        del Keep[key][:delete_size]
  return Keep


def default_class(All, Keep):
  ''' all images without retained labels go to default class. '''
  label_default = raw_input("Default label for all images not containing any of given labels? (name/N) ")
  if label_default is not 'N':
    Keep[label_default] = All['Perfect']
    # no need to check for overlap between Perfect and Keep's other
    # labels because Perfect overlaps with no other label by def
    already = set(chain(*Keep.values()))
    # below is why need to wait for merge_classes
    for key in All.keys():
      if key in Keep.keys()+['Perfect']: continue
      else:
        # computationally inefficient. but so much more flexible to
        # have this dict.
        # add fname if not in any
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


if __name__ == '__main__':
  import sys
  
  target_ratio, data_dir, to_dir = None, None, None
  for arg in sys.argv:
    if "imbalance-ratio=" in arg:
      target_ratio = float(arg.split('=')[-1])
    elif "data-dir=" in arg:
      data_dir = arg.split('=')[-1]
    elif "to-dir=" in arg:
      to_dir = arg.split('=')[-1]

  if data_dir is None:
    print "ERROR: data_dir not given"
    exit
      
  num_output = create_lookup_txtfiles(data_dir, target_ratio, to_dir)
  sys.exit(num_output)
