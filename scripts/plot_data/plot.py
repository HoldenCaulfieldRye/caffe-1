#!/usr/bin/env python
import cPickle as pickle
import sys, os, shutil, re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from math import ceil
from subprocess import call
# from imp import load_path
# ccn = load_path('ccn', '../../noccn/noccn/ccn.py')  
# from ccn import shownet

def write_data_to_txt(error, num_epochs=-1):
  if num_epochs == -1: end = len(error)
  else: end = num_epochs*800
  data = open(os.getcwd()+'/time_series.txt','w')
  data.writelines(["%i\t%.5f \t%.5f\n" % (x,train,test)
                   for x,(train,test) in enumerate(error[0:end])])
  print "wrote txt data to %s"%(os.getcwd()+'/time_series.txt')
  data.close()

  
def parse(content):
  error = []
  test_error = 1 # blank would be better for matplotlib
  for idx in xrange(len(content)):

    if content[idx].startswith("Testing "):

      if content[idx].startswith("Testing frequency"):
        test_freq = int(content[idx].strip().split(' ')[-1])
        print "test_freq changed to %i"%(test_freq)

      else:
        for i in range(idx-test_freq-1,idx-1):
          try:
            assert content[i].strip().split(' ')[-1] == 'sec)'
            error.append((float(content[i].strip().split(' ')[3].split(',')[0]),test_error))
          except:
            # print 'ERROR: the following line was supposed to contain a train error:\n',content[i]
            # print 'So assume training was interrupted, so breaking'
            break
            
    if "===Test output===" in content[idx]:
      test_error = float(content[idx+1].split(',')[0].split('  ')[-1])

  for idx in xrange(len(content)):
    if content[idx].startswith("Saved checkpoint"):
      saved_net = '/'+content[idx].split('/',1)[1]

  return error, saved_net


def matplot(cfg_dir, error, start=-1, end=-1):
  
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
  plt.savefig(cfg_dir+"/plot_train.png")
  # plt.show()




if __name__ == '__main__':

  print('Usage: ./plot.py path/to/model [--start-epoch=..] [--end-epoch==..]')

  try: 
    os.environ['DISPLAY']
  except: 
    print 'ERROR: X11 forwarding not enabled, cannot run script'
    exit

  train_path = sys.argv[1]

  # to get absolute path of cfg_dir, not relative
  back = os.getcwd()
  os.chdir(cfg_dir)
  cfg_dir = os.getcwd()
  os.chdir(back)

  with open(train_path) as f:
    content = f.readlines()
    error, saved_net = parse(content)

  # can specify
  start,end = -1,-1
  for arg in sys.argv:
    if arg.startswith("--start-epoch="):
      start = int(arg.split('=')[-1])
    if arg.startswith("--end-epoch="):
      end = int(arg.split('=')[-1])

  error = open('train_output.log.train','r').readlines()
  error = [line.replace('  ',' ').split(' ') for line in error
           if not line.startswith('#')]
  error = [(line[0],line[2]) for line in error]
  
  matplot(os.getcwd(), error, start, end)

  # ideal would be get layer names from cfg, and prompt for which ones
