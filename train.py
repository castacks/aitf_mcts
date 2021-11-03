
import argparse
import os 


import numpy as np
from gym import Gym
from mcts import MCTS
from net import Policy

class Train():

    def __init__(self,datapath,args):
        

        self.gym = Gym(datapath,args)
        self.net = Policy(args)
        self.mcts = MCTS(self.gym,self.net)
        self.executeEpisode()

    def executeEpisode(self):

        curr_position = self.gym.get_random_start_position()
        self.gym.plot_env(curr_position)

        curr_goal = self.gym.get_random_goal_location()
        # print(curr_position,curr_goal)
        

        while True:
            pi = np.squeeze(self.mcts.getActionProbs(curr_position,curr_goal))
            action = np.random.choice(len(pi),p=pi)
            print(action)
            curr_position = self.gym.getNextState(curr_position,action)
            self.gym.plot_env(curr_position)
            # break

    def train(self):
        pass

if __name__ == '__main__':
    parser=argparse.ArgumentParser(description='Train MCTS model')

    parser.add_argument('--dataset_folder',type=str,default='/dataset/')
    parser.add_argument('--dataset_name',type=str,default='7days1')
    parser.add_argument('--obs',type=int,default=20)
    parser.add_argument('--preds',type=int,default=120)
    parser.add_argument('--preds_step',type=int,default=10)
    parser.add_argument('--delim',type=str,default=' ')

    parser.add_argument('--input_size',type=int,default=3)
    parser.add_argument('--num_channels',type=int,default=2)
    parser.add_argument('--channel_size',type=int,default=256)
    parser.add_argument('--kernel_size',type=int,default=4)
    parser.add_argument('--dropout',type=float,default=0.05)



    args=parser.parse_args()

    datapath = os.getcwd() + args.dataset_folder + args.dataset_name + "/processed_data/"
    Train(datapath,args)
