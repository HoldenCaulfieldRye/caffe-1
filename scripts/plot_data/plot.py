import cPickle as pickle
import sys, os, shutil, re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from math import ceil
from subprocess import call
from os.path import join as ojoin


def matplot(model_dir, error, start=-1, end=-1):
  
  if end == start == -1:
    start, end = 0, len(error)
    print 'plotting entire training data'
    
  elif start == -1:
    start = 0
    print 'plotting from epoch %i to %i'%(start,end)
    end *= 800
    
  elif end == -1:
    print 'plotting from epoch %i to the end'%(start)
    start, end = start*800, len(error)

  else:
    print 'plotting from epoch %i to %i'%(start,end)
    start, end = start*800, end*800
    
  x = np.array(range(len(error[start:end])))
  ytrain = np.array([train for (train,test) in error[start:end]])
  # ytest = np.array([test for (train,test) in error[start:end]])
  plt.plot(x, ytrain, label='training error')
  # plt.plot(x, ytest, label='validation error')
  plt.legend(loc='upper left')
  plt.xlabel('Iters')
  plt.ylabel('TrainingLoss')
  # plt.title('Go on choose one')
  plt.grid(True)
  plt.savefig(ojoin(model_dir,"/plot_train.png"))
  # plt.show()




if __name__ == '__main__':

  print('Usage: python plot.py path/to/model [--start-epoch=..] [--end-epoch==..]')

  try: 
    os.environ['DISPLAY']
  except: 
    print 'ERROR: X11 forwarding not enabled, cannot run script'
    sys.exit()

  start,end = -1,-1
  for arg in sys.argv:
    if arg.startswith("--start-epoch="):
      start = int(arg.split('=')[-1])
    if arg.startswith("--end-epoch="):
      end = int(arg.split('=')[-1])

      
  model_dir = sys.argv[1]
  # to use absolute path of model_dir, not relative
  back = os.getcwd()
  os.chdir(model_dir)
  model_dir = os.getcwd()
  os.chdir(back)

  content = open(ojoin(model_dir,'train_output.log.train'),'r').readlines()
  content = [line.replace('  ',' ').split(' ') for line in content
           if not line.startswith('#')]
  content = [(line[0],line[2]) for line in content]
  
  matplot(model_dir, content, start, end)

  # ideal would be get layer names from cfg, and prompt for which ones
