import os, sys
from os.path import join as ojoin

# useage: python create_deploy_prototxt.py <model_dir>

def get_train_file(model_dir):
  train_file = ''
  for fname in os.listdir(model_dir):
    if fname.endswith('train.prototxt'):
      return open(ojoin(model_dir,fname),'r')
  print 'no train prototxt found'
  sys.exit()

def edit_train_content_for_deploy(content):
  idx = 0
  while idx < len(content):
    if 'blobs_lr' in content[idx]:
      del content[idx]
      # idx += 1
    elif 'weight_decay' in content[idx]:
      del content[idx]
      # idx += 1
    elif 'weight_filler' in content[idx]:
      del content[idx:idx+7]
      idx += 7
    elif 'accuracy' in content[idx]:
      del content[idx-1:idx+5]
      idx += 5
    else:
      idx += 1
  return content

def write_content_to_deploy_file(model_dir, content):
  model_name = model_dir.split('/')[-1]
  model_name = model_name.split('-fine')[0]
o  fname = ojoin(model_dir,model_name+'_deploy.prototxt')
  print "fname: %s"%(fname)
  deploy_file = open(fname,'w')
  deploy_file.writelines(content)
  deploy_file.close()

if __name__ == '__main__':
  model_dir = os.path.abspath(sys.argv[1])
  train_file = get_train_file(model_dir)

  content = train_file.readlines()

  content = edit_train_content_for_deploy(content)

  write_content_to_deploy_file(model_dir, content)
    



