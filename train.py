
import argparse
import os 


import numpy as np
from gym import Gym
from mcts import MCTS

class Train():

    def __init__(self,datapath,args):
        

        self.gym = Gym(datapath,args)
        self.mcts = MCTS(self.gym,None)
        self.executeEpisode()

    def executeEpisode(self):

        curr_position = self.gym.get_random_start_position()
        curr_goal = self.gym.get_random_goal_location()
        print(curr_position,curr_goal)
        

        # while True:
        # pi = self.mcts.getActionProbs(curr_position)
        # action = np.random.choice(len(pi),p=pi)
        # curr_position = self.gym.get_next_position(curr_position,action)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description='Train MCTS model')

    parser.add_argument('--dataset_folder',type=str,default='/dataset/')
    parser.add_argument('--dataset_name',type=str,default='7days1')
    parser.add_argument('--obs',type=int,default=20)
    parser.add_argument('--preds',type=int,default=120)
    parser.add_argument('--preds_step',type=int,default=10)
    parser.add_argument('--delim',type=str,default=' ')

    args=parser.parse_args()

    datapath = os.getcwd() + args.dataset_folder + args.dataset_name + "/processed_data/"
    actionpath = os.getcwd()
    Train(datapath,args)
