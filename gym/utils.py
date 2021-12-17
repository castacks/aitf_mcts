import math
import os
from tqdm import tqdm

from torch import nn
import torch
from torch.utils.data import Dataset
import numpy as np
import random
THRESH = 5 #KM


def populate_traj_lib():
	
    # note position of motion prim library text files
    lib_path = os.getcwd() + '/gym/traj_lib_0SI.txt'
    index_path = os.getcwd() + '/gym/traj_index_0SI.txt'
    print("Loading traj lib from", lib_path)
    ## obtain a 3d matrix of each trajectory's (x, y, z) positions into an array
    file1 = open(lib_path, 'r',newline='\r\n')
    traj_no = 252 # note the number
    count, j = 0,0
    traj_lib = np.zeros([traj_no,3,20])

    for line in open(lib_path, 'r',newline='\n'):
        if count%4 == 1:
            traj_lib[j,0] = np.fromstring( line.strip(), dtype=float, sep=' ' )/1000
        if count%4 == 2:
            traj_lib[j,1] = np.fromstring( line.strip(), dtype=float, sep=' ' )/1000
        if count%4 == 3:
            traj_lib[j,2] = np.fromstring( line.strip(), dtype=float, sep=' ' )/1000
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


    # return one-hot vector of goal position
def direction_goal_detect(pos,second_pos):
    
    dir_array = torch.zeros([10]) ## [N, NE, E, SE, S, SW, W, NW, R1, R2]
    yaw_diff = pos-second_pos

    if np.linalg.norm(pos) > THRESH:
        # print("diff",difference,'pos',pos, "input_pos",input_pos)
            planar_slope = torch.atan2(pos[1],pos[0])
            degrees_slope = planar_slope*180.0/np.pi
          

            if degrees_slope <22.5 and degrees_slope >-22.5: #east
                dir_array[2] = 1.0
            elif degrees_slope <67.5 and degrees_slope >22.5: #NE
                dir_array[1] = 1.0
            elif degrees_slope <112.5 and degrees_slope >67.5: #N
                dir_array[0] = 1.0
            elif degrees_slope <157.5 and degrees_slope >112.5: # NW
                dir_array[7] = 1.0
            elif degrees_slope <-157.5 or degrees_slope >157.5: # W
                dir_array[6] = 1.0
            elif degrees_slope <-22.5 and degrees_slope >-67.5: #SE
                dir_array[3] = 1.0
            elif degrees_slope <-67.5 and degrees_slope >-112.5: #S
                dir_array[4] = 1.0
            elif degrees_slope <-112.5 and degrees_slope >-157.5: #SW:
                dir_array[5] = 1.0
        # print("Outer pos reached",goal_enum(dir_array))
    else:
        
            yaw_diff_slope = torch.atan2(yaw_diff[1],yaw_diff[0])
            yaw_diff_slope_degrees = yaw_diff_slope*180.0/np.pi
            # print(yaw_diff_slope_degrees)
            if pos[0]<0.2 and pos[0]> -0.2 and abs(pos[1])<0.20 and pos[2] <0.3: #1
                if abs(yaw_diff_slope_degrees) <20.0:
                    dir_array[9] = 1.0
                    # print("Runway reached",goal_enum(dir_array))


            elif pos[0]<1.65 and pos[0]> 1.25 and abs(pos[1])<0.20 and pos[2] <0.3:  #2,
                if 180-abs(yaw_diff_slope_degrees) <20.0:
                    dir_array[8] = 1.0
                    # print("Runway reached",goal_enum(dir_array))

    return dir_array

def goal_enum(goal):
    msk = goal.squeeze().numpy().astype(bool)
    g = ["N","NE","E","SE","S","SW","W","NW","R1","R2"]
    return [g[i] for i in range(len(g)) if msk[i]]

def goal_eucledian_list(num_goals = 10):

    pos = []
    for goal_idx in range(num_goals): 
        ang = np.array([90,45,0,-45,-90,-135,180,135])
        
        if goal_idx < 8:
            pos.append(np.array([THRESH*np.cos(np.deg2rad(ang[goal_idx])),THRESH*np.sin(np.deg2rad(ang[goal_idx])), 1.0 ]))
        elif goal_idx == 9:
            pos.append(np.array([0.0,0.0,0.2]))
        elif goal_idx == 8:
            pos.append(np.array([1.45,0.0,0.2]))

    return pos

