import argparse
import copy
import os
from numpy.lib import utils
from tqdm import tqdm
import torch
import numpy as np
from gym import Gym
from train import Net
from collections import deque
from utils import goal_enum
import torch.multiprocessing as mp
import time
from play_utils import *

# torch.manual_seed(5345)
# import random
# random.seed(5345)
# np.random.seed(5345)

class Play():

    def __init__(self, datapath, args):

        self.args = args
        self.datapath = datapath
        self.net = Net(args)
        self.gym = Gym(datapath, args)
        # self.mcts = MCTS(self.gym, self.net, args)
        self.play()


    def parallel_play(self,rank,world_size):
        iterationTrainExamples = deque([])

        # net = Net(self.args)
        # net.nnet.eval()
        gym = Gym(self.datapath, self.args)
        print("Playing with Process", rank)

        for i in range(1):
            states = run_episode(i,gym,self.net,self.args)
            if states is not None:
                iterationTrainExamples += states   
        save_episodes(self.args.checkpoint,iterationTrainExamples,rank) 



    def play(self):
        
        iterationTrainExamples = deque([], maxlen=self.args.maxlenOfQueue)

        for ite in range(args.numIters):
            self.net.nnet.eval()

            t = time.time()            
            mp.spawn(self.parallel_play, args=(5,), nprocs=3, join=True)
            print(time.time() - t)

            iterationTrainExamples += load_episodes(self.args.checkpoint) 
            print("Number of training samples:",len(iterationTrainExamples))
                # print(iterationTrainExamples)
                # if ite==0:
                    # self.save_episodes(iterationTrainExamples,ep)

            print("Training....")

            self.net.train(iterationTrainExamples)
            print("Testing....")
            self.net.nnet.eval()
            self.test()


    
    def test(self):
        accuracy = 0
        for _ in tqdm(range(self.args.numEpsTest)):
            self.mcts = MCTS(self.gym, self.net, self.args)  # reset search tree
            states = run_episode(0,self.gym,self.net,self.args)
            if states is not None:
                if states[0][3]==1:
                    accuracy += 1
        print("Accuracy = ",accuracy/self.args.numEpsTest)





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train MCTS model')

    parser.add_argument('--dataset_folder', type=str, default='/dataset/')
    parser.add_argument('--dataset_name', type=str, default='7days1')
    parser.add_argument('--models_folder', type=str, default='/saved_models/')
    parser.add_argument('--model_weights', type=str, default=None)
    parser.add_argument('--checkpoint', type=str, default='/episodes/')
    parser.add_argument('--load_episodes', type=bool, default=False)

    parser.add_argument('--obs', type=int, default=20)
    parser.add_argument('--preds', type=int, default=120)
    parser.add_argument('--preds_step', type=int, default=10)
    parser.add_argument('--delim', type=str, default=' ')

    parser.add_argument('--input_size', type=int, default=3)
    parser.add_argument('--num_channels', type=int, default=1)
    parser.add_argument('--channel_size', type=int, default=[256,256,128])
    parser.add_argument('--kernel_size', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.05)
    parser.add_argument('--balance_data', type=bool, default=True)

    parser.add_argument('--numMCTS', type=int, default=50)
    parser.add_argument('--cpuct', type=int, default= 1)

    parser.add_argument('--numEpisodeSteps', type=int, default=20)
    parser.add_argument('--maxlenOfQueue', type=int, default=25600)
    parser.add_argument('--numEps', type=int, default=1000)
    parser.add_argument('--numEpsTest', type=int, default=1)

    parser.add_argument('--numIters', type=int, default=20)

    parser.add_argument('--epochs', type=int, default=15)

    parser.add_argument('--plot', type=bool, default=False)






    args = parser.parse_args()

    datapath = os.getcwd() + args.dataset_folder + args.dataset_name + "/processed_data/"
    Play(datapath, args)
