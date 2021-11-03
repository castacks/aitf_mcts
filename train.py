
import argparse
import os 

import torch
import numpy as np
from gym import Gym
from mcts import MCTS
from net import Policy

class Train():

    def __init__(self,datapath,modelpath,args):
        
        self.net = Policy(args)
        self.gym = Gym(datapath,args)
        self.mcts = MCTS(self.gym,self.net,args)
        if modelpath is not None:
            self.load_hot_start(modelpath)
        self.executeEpisode()
            
    def load_hot_start(self,modelpath):
		
        checkpoint = torch.load(modelpath,map_location=torch.device('cpu'))
        miss, unex = self.net.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print(miss)
        print(unex)


    def executeEpisode(self):

        curr_position = self.gym.get_random_start_position()
        # self.gym.plot_env(curr_position)

        curr_goal = self.gym.get_random_goal_location()
        # print(curr_position,curr_goal)
        

        while True:
            pi = np.squeeze(self.mcts.getActionProbs(curr_position,curr_goal))
            action = np.random.choice(len(pi),p=pi)
            curr_position = self.gym.getNextState(curr_position,action)
            # self.gym.plot_env(curr_position)
            break

    def train(self):
        pass

if __name__ == '__main__':
    parser=argparse.ArgumentParser(description='Train MCTS model')

    parser.add_argument('--dataset_folder',type=str,default='/dataset/')
    parser.add_argument('--dataset_name',type=str,default='7days1')
    parser.add_argument('--models_folder',type=str,default='/saved_models/')
    parser.add_argument('--obs',type=int,default=20)
    parser.add_argument('--preds',type=int,default=120)
    parser.add_argument('--preds_step',type=int,default=10)
    parser.add_argument('--delim',type=str,default=' ')

    parser.add_argument('--input_size',type=int,default=3)
    parser.add_argument('--num_channels',type=int,default=2)
    parser.add_argument('--channel_size',type=int,default=512)
    parser.add_argument('--kernel_size',type=int,default=3)
    parser.add_argument('--dropout',type=float,default=0.05)

    parser.add_argument('--numMCTS',type=int,default=50)
    parser.add_argument('--cpuct',type=int,default=2)



    args=parser.parse_args()

    datapath = os.getcwd() + args.dataset_folder + args.dataset_name + "/processed_data/"
    modelpath = os.getcwd() + args.models_folder + 'goalGAIL6by_91.pt'
    Train(datapath, modelpath, args)
