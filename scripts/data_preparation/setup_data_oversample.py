import numpy as np
import os
from os.path import join as ojoin
from PIL import Image
from operator import itemgetter as ig
from itertools import chain
from datetime import date
from shutil import rmtree
import copy
import json, yaml, random
import setup_data


def main(data_dir, data_info, to_dir, target_bad_min):
  ''' This is the master function. 
  data_dir: where raw data is. data_info: where to store .txt files. '''
  All = setup_data.get_label_dict(data_dir)
  total_num_images = All.pop('total_num_images')
  Keep = setup_data.classes_to_learn(All)
  # merge_classes only after default label entry created
  Keep = setup_data.default_class(All, Keep)
  total_num_check = sum([len(Keep[key]) for key in Keep.keys()])
  if total_num_images != total_num_check:
    print "\nWARNING! started off with %i images, now have %i distinct training cases"%(total_num_images, total_num_check)
  Keep, num_output = setup_data.merge_classes(Keep)
  Keep, num_output = setup_data.check_mutual_exclusion(Keep, num_output)
  print "target bad min: %s" %(target_bad_min)
  Keep = rebalance_oversample(Keep, total_num_images, target_bad_min)
  print 'finished rebalancing'
  Keep = setup_data.within_class_shuffle(Keep)
  print 'finished shuffling'
  dump = setup_data.symlink_dataset(Keep, data_dir, to_dir)
  if data_info is not None:
    setup_data.dump_to_files(Keep, dump, data_info)
  return num_output, dump


def rebalance_oversample(Keep, total_num_images, target_bad_min):
  '''if target_bad_min not given, prompts user for one; 
  and implements it. Note that with >2 classes, this can be 
  implemented either by downsizing all non-minority classes by the
  same factor in order to maintain their relative proportions, or 
  by downsizing as few majority classes as possible until
  target_bad_min achieved. We can assume that we care mostly about 
  having as few small classes as possible, so the latter is 
  implemented.'''
  if target_bad_min == 'N': return Keep
  else: target_bad_min = float(target_bad_min)
  # minc is class with minimum number of training cases
  ascending_classes = sorted([(key,len(Keep[key]))
                              for key in Keep.keys()],
                             key=lambda x:x[1])
  maxc, len_maxc = ascending_classes[-1][0], ascending_classes[-1][1]
  minc, len_minc = ascending_classes[0][0], ascending_classes[0][1]
  # print ascending_classes
  # print "\ntotal num images: %i"%(total_num_images)
  minc_proportion = float(len_minc)/total_num_images
  maxc_proportion = float(len_maxc)/total_num_images
  if target_bad_min is None:
    target_bad_min = raw_input("\nmax class currently takes up %.2f, what's your target? [num/N] "%(maxc_proportion))
  if target_bad_min is not 'N':
    target_bad_min = float(target_bad_min)
    print 'minc_proportion: %.2f, target_bad_min: %.2f'%(minc_proportion, target_bad_min)
    if maxc_proportion > target_bad_min:
      copy_size = ( target_bad_min*(total_num_images) - len_minc ) / (1 - target_bad_min)
      copy_size = int(copy_size)
      random.shuffle(Keep[minc])
      print '%s has %i images so %i copies will be made'%(minc, len_maxc,copy_size)
      min_imgs_copy = copy.copy(Keep[minc])
      print 'min_imgs_copy has %i images'%(len(min_imgs_copy))
      # number_of_copies = 
      # number_of_copies = int(number_of_copies)
      for i in range((copy_size/len_minc)):
        Keep[minc] += min_imgs_copy
      Keep[minc] += min_imgs_copy[:(copy_size % len_minc)]
      random.shuffle(Keep[minc])
      total_num_images = len(Keep[minc]) + len(Keep[maxc])
      print 'minc now has %i images, compared to %i for maxc'%(len(Keep[minc]), len(Keep[maxc]))
      assert len(Keep[maxc])/float(total_num_images) == target_bad_min
  return Keep


if __name__ == '__main__':
  import sys
  
  target_bad_min, data_dir, data_info = None, None, None
  for arg in sys.argv:
    if "bad-min=" in arg:
      target_bad_min = arg.split('=')[-1]
      print "target bad min: %s" %(target_bad_min)
    elif "data-dir=" in arg:
      data_dir = os.path.abspath(arg.split('=')[-1])
    elif "to-dir=" in arg:
      to_dir = os.path.abspath(arg.split('=')[-1])
    elif "data-info=" in arg:
      data_info = os.path.abspath(arg.split('=')[-1])

  # data_dir = /data/ad6813/pipe-data/Bluebox/raw_data/dump
  # data_info = /data/ad6813/caffe/data_info
  if data_dir is None:
    print "\nERROR: data_dir not given"
    exit
      
  num_output,dump = main(data_dir, data_info, to_dir, target_bad_min)
  print "\nIt's going to say 'An exception has occured etc'"
  print "but don't worry, that's num_output info for the training shell script to use\n"
  sys.exit(num_output)

