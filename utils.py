import math
import os
from tqdm import tqdm

from torch import nn
from torch.utils.data import Dataset

import numpy as np
from scipy.spatial import distance_matrix
import random

from motion_primitives_no_azi import *


##Dataloader class


def get_hash():
	pass

def populate_traj_lib():
	
	# note position of motion prim library text files
	lib_path = os.getcwd() + '/traj_lib_0SI.txt'
	index_path = os.getcwd() + '/traj_index_0SI.txt'

	## obtain a 3d matrix of each trajectory's (x, y, z) positions into an array
	file1 = open(lib_path, 'r',newline='\r\n')
	traj_no = 252 # note the number
	count, j = 0,0
	traj_lib = np.zeros([traj_no,3,20])

	for line in open(lib_path, 'r',newline='\n'):
		if count%4 == 1:
		traj_lib[j,0] = np.fromstring( line.strip(), dtype=float, sep=' ' )
		if count%4 == 2:
		traj_lib[j,1] = np.fromstring( line.strip(), dtype=float, sep=' ' )
		if count%4 == 3:
		traj_lib[j,2] = np.fromstring( line.strip(), dtype=float, sep=' ' )
		j+=1
		count += 1

	## obtain the details of each trajectory from the index
	file1 = open(index_path, 'r',newline='\r\n')

	j = 0
	index_lib = np.zeros([traj_no,6])

	for line in open(index_path, 'r',newline='\n'):
		index_lib[j,:] = np.fromstring( line.strip(), dtype=float, sep=' ' )
		j+=1

	return traj_lib,index_lib


