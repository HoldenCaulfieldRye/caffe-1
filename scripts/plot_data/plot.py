import sys, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from os.path import join as ojoin


def matplot(model_dir, train, val, start=-1, end=-1):
  
  # if end == start == -1:
  start, end = 0, len(error)
  print 'plotting entire training data'
    
  # elif start == -1:
  #   start = 0
  #   print 'plotting from epoch %i to %i'%(start,end)
  #   end *= 800
    
  # elif end == -1:
  #   print 'plotting from epoch %i to the end'%(start)
  #   start, end = start*800, len(train)

  # else:
  #   print 'plotting from epoch %i to %i'%(start,end)
  #   start, end = start*800, end*800
    
  x = np.array(range(len(train[start:end])))
  ytrain = np.array([el[1] for el in train[start:end]])
  ytest = np.array([el[1] for el in val[start:end]])
  plt.plot(x, ytrain, label='training train')
  plt.plot(x, ytest, label='validation train')
  plt.legend(loc='upper left')
  plt.xlabel('Iters')
  plt.ylabel('TrainingLoss')
  # plt.title('Go on choose one')
  plt.grid(True)
  plt.savefig(ojoin(model_dir,'plot.png'))
  # plt.show()


def get_caffe_train_errors():
  get_caffe_errors(ojoin(model_dir,'train_output.log.train'))

def get_caffe_val_errors():
  get_caffe_errors(ojoin(model_dir,'train_output.log.test'))

def get_caffe_errors(error_file):
  content = open(error_file,'r').readlines()
  content = [' '.join(line.split()).split(' ') for line in content
             if not line.startswith('#')]
  print 'content looks like %s and %s'%(content[0], content[-1])
  content = [(line[0],line[2]) for line in content]
  print 'content looks like %s and %s'%(content[0], content[-1])
  return content



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

      
  model_dir = os.path.abspath(model_dir)

  train, val = get_caffe_train_errors(model_dir), get_caffe_val_errors(model_dir)
  
  matplot(model_dir, train, val, start, end)

  # ideal would be get layer names from cfg, and prompt for which ones
