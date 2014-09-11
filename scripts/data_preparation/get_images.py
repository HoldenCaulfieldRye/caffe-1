import os, json
from os.path import join as oj

def all_labels(dirlist):
  queries = {'perfect':[]}
  for dirname in dirlist:
    with open(oj(dirname,'inspection.txt')) as f:
      lines = f.readlines()
      lines = [line.strip() for line in lines]
      if lines == []:
        queries['perfect'].append(dirname)
      for line in lines:
        if line not in queries.keys():
          queries[line] = []
        queries[line].append(dirname)
  return queries

def find_them():

  back = os.getcwd()
  img_dir = '*'
  while not os.path.exists(img_dir):
    img_dir = 'data2/ad6813/pipe-data/Redbox/raw_data/dump'  # raw_input('path to queries? ')

  os.chdir(img_dir)
  dirlist = os.listdir(os.getcwd())

  queries = all_labels(dirlist)

  os.chdir(back)
  json.dump(queries, open('queries.txt','w'))

  return queries
    

if __name__ == '__main__':

  find_them()
