import os, json, shutil
from os.path import join as oj

# this is the one you used to get Water contam examples from new
# jointmanager.com file structure


def all_labels(dirlist):
  Queries = {'perfect':[]}
  for dirname in dirlist:
    with open(oj(dirname,'inspection.txt')) as f:
      lines = f.readlines()
      lines = [line.strip() for line in lines]
      if lines == []:
        Queries['perfect'].append(dirname)
      for line in lines:
        if line not in Queries.keys():
          Queries[line] = []
        Queries[line].append(dirname)
  return Queries


# this function was originally the only call from main of this script
def find_them():
  back = os.getcwd()
  img_dir = '*'
  while not os.path.exists(img_dir):
    img_dir = 'data2/ad6813/pipe-data/Redbox/raw_data/dump'  # raw_input('path to Queries? ')
  os.chdir(img_dir)
  dirlist = os.listdir(os.getcwd())
  Queries = all_labels(dirlist)
  os.chdir(back)
  json.dump(Queries, open('Queries.txt','w'))
  return Queries
    

def sample_from_label(Queries, data_dir):
  for elem in enumerate(Queries.keys()): print elem
  lab, length = -1, -1
  while lab not in range(len(Queries.keys())):
    merge = int(raw_input("\nName 1 class number you wish to sample from: "))
  lab = Queries.keys()[lab]
  while length > len(Queries[lab]):
    length = int(raw_input("\nSample how many? "))
  try:
    os.mkdir(label)
  except:
    if not raw_input('Sample for that class already exists, ok to overwrite? [Y]/N ') == 'N':
      shutil.rmtree(label)
      os.mkdir(label)
  for directory in Queries[label][:length]:
    for f in os.listdir(oj(data_dir,directory)):
      if f.endswith('.jpg'): 
        shutil.copy(ojoin(data_dir,f),ojoin(ojoin(label,f)))
  
  

if __name__ == '__main__':
  Queries = all_labels(data_dir)
  sample_from_label(Queries, data_dir)
