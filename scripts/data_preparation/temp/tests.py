import numpy as np
import os
from os.path import join as ojoin
from PIL import Image
from operator import itemgetter as ig
from itertools import chain
import json, random
from create_lookup_txtfiles_2 import *


def test_create_lookup_txtfiles(data_dir, to_dir):
  create_lookup_txtfiles(data_dir, to_dir)

if __name__ == '__main__':
  import sys
  test_create_lookup_txtfiles(sys.argv[1],sys.argv[2])

