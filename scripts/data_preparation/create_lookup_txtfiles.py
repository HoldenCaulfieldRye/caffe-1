
# n01440764/n01440764_10290.JPEG 0
# n01440764/n01440764_10293.JPEG 0

import numpy as np
import os
from os.path import join as ojoin
from PIL import Image
import json, random


#### STEP 1: GET LABELS #############################################

def get_all_pipe_labels(data_dir,save=True):
  ''' looks into all .dat files in data_dir, and if find a new 
  label, add it to the list. stores final list as binary pickle 
  file.'''
  path = data_dir
  whichBox = data_dir.split('/')[-1]
  d = {'labels': []}
  print 'getting all existing labels...'
  for filename in os.listdir(path):
    if not filename.endswith('.dat'): continue
    fullname = os.path.join(path, filename)
    with open(fullname) as f:
      content = [line.strip() for line in f.readlines()] 
      for label in content:
        if label not in d['labels']:
          # print label
          d['labels'].append(label)
  if save==True:
    d['labels'].sort()
    d['no_labels'] = len(d['labels'])
    pickle.dump(d, open('labels_'+whichBox+'.pickle', 'wb'))
    print 'saved pickle file in', os.getcwd()
  return d

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



#### STEP 2: CREATE train.txt val.txt  #############################

def create_lookup_txtfiles(data_dir, to_dir=None):
  ''' data_dir: where raw data is
      to_dir: where to store .txt files. '''
  
  list_dir = os.listdir(data_dir) # names of all elements in dir
  lastLabelIsDefault = False
  img_labels = [] # image's labels to train on
  dump = []       # contain text to write to .txt files
  case_count = 0 # number of training cases
  tagless_count = 0 # n
  badcase_count = 0 # num of images with multiple flags to train on

  if to_dir is not None:
    train_file = open(ojoin(to_dir,'train.txt'), 'w')
    val_file = open(ojoin(to_dir,'val.txt'), 'w')
    test_file = open(ojoin(to_dir,'test.txt'), 'w')
    read_file = open(ojoin(to_dir,'read.txt'), 'w')
  
  # get labels of classes to learn
  read_labels = get_all_pipe_labels(data_dir,save=False)['labels']
  lookup = {}
  # count has numbers rather than label names as lookup values, in
  # order to deal with diffs betw read_labels and write_labels
  count = {}
  for num,label in enumerate(read_labels):
    lookup[label] = num
  for elem in enumerate(read_labels): print elem
  read_labels = [read_labels[int(num)] for num in raw_input("Numbers of labels to learn, separated by ' ': ").split()]
  write_labels = read_labels[:]
    
  lookup, write_labels = merge_classes(lookup, write_labels)

  label_default = raw_input("Default label for all images not containing any of given labels? (name/N) ")
  if label_default is not 'N':
    lastLabelIsDefault = True
    lookup[label_default] = len(write_labels)
    write_labels.append(label_default)
            
  print 'sorting images by class label...'
  for fname in list_dir:
    if not fname.endswith('.dat'): continue
    case_count += 1    
    fullname_dat = os.path.join(data_dir, fname)
    rootname = os.path.splitext(fname)[0]
    with open(fullname_dat) as f:
      content = [line.strip() for line in f.readlines()] 
      img_labels = [label for label in read_labels
                    if label in content]

      # last label is a normal label iif images with no labels will
      # not be batched
      if not img_labels: 
        if lastLabelIsDefault:
          dump.append((fname.split('.')[0]+'.jpg',
                       lookup[label_default]))
          # len(write_labels) is lookup val for default label
          count[lookup[label_default]] += 1 
        else: tagless_count += 1
      else:
        # if image has multiple flags, it will appear in each flag
        # subdir, each time with only one label. this is very bad for
        # training, so hopefully such cases are very rare.'
        if len(img_labels)>1: 
          badcase_count += len(img_labels)-1
          case_count += len(img_labels)-1
        for label in img_labels:
          dump.append((fname.split('.')[0]+'.jpg',lookup[label]))
          count[lookup[label]] += 1

  print "dump has %i elements, looking like %s and %s"%(len(dump),dump[0], dump[300])

  dump = rebalance(dump, count, case_count)

  # write dump to train and val files
  # randomise!!
  # 80% of dataset for training, 7% for validation, 14% for testing
  non_train_dump_size = int(0.2*len(dump))
  relative_val_size = int(0.34*non_train_dump_size)
  non_train_dump = random.sample(dump, non_train_dump_size)
  val_dump = random.sample(non_train_dump, relative_val_size)
  test_dump = [elem for elem in non_train_dump
               if elem not in val_dump]
  train_dump = [elem for elem in dump if elem not in non_train_dump]
  random.shuffle(train_dump)

  if to_dir is not None:
    train_file.writelines(["%s %i\n" % (fname,num)
                           for (fname,num) in train_dump])
    val_file.writelines(["%s %i\n" % (fname,num)
                         for (fname,num) in val_dump])
    test_file.writelines(["%s %i\n" % (fname,num)
                         for (fname,num) in test_dump])

    # write to read file how to interpret values as classes
    read_file.writelines(["%i %s\n" % (lookup[label],label,)
                           for label in write_labels])
    train_file.close()
    val_file.close()
    test_file.close()
    read_file.close()


  print 'create_lookup_txtfiles complete. summary stats:'
  print 'badcase_freq: %0.2f' % (float(badcase_count) / case_count)
  print 'tagless_freq: %0.2f' % (float(tagless_count) / case_count)

  return train_dump, val_dump, test_dump


# imbalance needs to be calculated from write_labels, not read_labels
def rebalance(dump, count, case_count):
  
  return dump


def update_labels(write_labels, merge, new_label):
  write_labels = [label for label in write_labels if label not in [write_labels[i] for i in merge]]
  write_labels.append(new_label)
  return write_labels

def merge_classes(lookup, write_labels):
  more = 'Y'
  while more == 'Y':
    print '%s' % (', '.join(map(str,write_labels)))
    if raw_input('Merge (more) classes? (Y/N) ') == 'Y':
      merge = [-1]
      while not all([idx in range(len(write_labels)) for idx in merge]):
        for elem in enumerate(write_labels): print elem
        merge = [int(elem) for elem in raw_input("Name two class numbers from above, separated by ' ': ").split()]
      merge.sort()
      merge_label = raw_input("Name of merged class: ")
      for label in [merge_label, write_labels[merge[0]], write_labels[merge[1]]]:
        lookup[label] = merge[0]        
      write_labels = update_labels(write_labels, merge, merge_label)
    else: more = False
  return lookup, write_labels 


if __name__ == '__main__':
  import sys
  create_lookup_txtfiles(sys.argv[1],sys.argv[2])

  # else: print 'arg not recognised'

  
