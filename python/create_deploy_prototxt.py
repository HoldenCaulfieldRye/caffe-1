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
  for idx in range(len(content)):
    if 'blobs_lr' in line: del content[idx]
    elif 'weight_decay' in line: del content[idx]
    elif 'weight_filler' in line: del content[idx:idx+7]
    elif 'accuracy' in line: del content[idx-1:idx+5]
  return content

def write_content_to_deploy_file(model_dir, content):
  model_name = model_dir.split('/')[-1]
  deploy_file = open(ojoin(model_dir,model_name+'_deploy.prototxt'),'w')
  deploy_file.writelines(content)
  deploy_file.close()

if __name__ == '__main__':
  model_dir = os.path.abspath(sys.argv[2])
  train_file = get_train_file(model_dir)

  content = train_file.readlines()

  content = edit_train_content_for_deploy(content)

  write_content_to_deploy_file(model_dir, content)
    



