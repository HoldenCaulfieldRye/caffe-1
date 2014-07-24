import numpy as np
import os
from os.path import join as ojoin
from PIL import Image
from shutil import rmtree
import json, random

from create_lookup_txtfiles import main


# image-label mapping in data_info_dir/{train,val,test}.txt
def move_to_dirs(data_src_dir, data_dest_dir, data_info_dir, target_bad_min=None):
  data_src_dir = os.path.abspath(data_src_dir)
  if os.path.exists(data_dest_dir): rmtree(data_dest_dir)
  os.mkdir(data_dest_dir)
  data_dest_dir = os.path.abspath(data_dest_dir)
  # task_dir = os.path.abspath(task_dir)

  num_output, dump = create_lookup_txtfiles(data_src_dir, data_info_dir, target_bad_min)

  assert len(dump) == 3
  
  cross_val = [np.array(d) for d in dump]

  print "np.array(d).shape:", np.array(dump[0]).shape
  print "cross_val[0].shape:", cross_val[0].shape
  print "cross_val length: %i, each elem with shape %s"%(len(cross_val), cross_val[0].shape)

  for d,dname in zip(cross_val,['train','val','test']):
    dddir = ojoin(data_dest_dir,dname)
    if os.path.isdir(dddir): rmtree(dddir)
    os.mkdir(dddir)
    print 'symlinking %s with shape %s'%(dname, d.shape)
    for fname in d[:,0]:
      os.symlink(ojoin(data_src_dir,fname),ojoin(dddir,fname))

  return num_output



if __name__ == '__main__':
  import sys
  
  data_dir, to_dir, data_info, target_bad_min = None, None, None, None
  for arg in sys.argv:
    if "bad-min=" in arg:
      target_bad_min = float(arg.split('=')[-1])
    elif "data-dir=" in arg:
      data_dir = arg.split('=')[-1]
    elif "to-dir=" in arg:
      to_dir = arg.split('=')[-1]
    elif "data-info=" in arg:
      data_info = arg.split('=')[-1]

  for ddir, varname in zip([data_dir, to_dir, data_info],
                           ['data_dir', 'to_dir', 'data_info']):
    if ddir is None:
      print "\nERROR: %s not given"%(varname)
      exit
      
  num_output = move_to_dirs(data_dir, to_dir, data_info,
                            target_bad_min)
  
  print "\nIt's going to say 'An exception has occured etc'"
  print "but don't worry, that's just information for the training shell script to use\n"
  sys.exit(num_output)
