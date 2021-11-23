import argparse
import os
from numpy.lib import utils
from tqdm import tqdm
import torch
import numpy as np
from gym import Gym
from mcts import MCTS
from train import Net
from collections import deque
from utils import goal_enum

class Play():

    def __init__(self, datapath, args):

        self.args = args
        self.net = Net(args)
        self.gym = Gym(datapath, args)
        self.mcts = MCTS(self.gym, self.net, args)
        # self.train()
        self.play()


    def executeEpisode(self):

        curr_position = self.gym.get_random_start_position()
        # self.gym.plot_env(curr_position)

        curr_goal = self.gym.get_random_goal_location()
        # print("curr goal",goal_enum(curr_goal))
        
        trainExamples = []
        episodeStep = 0

        while True:
            episodeStep += 1

            pi = np.squeeze(self.mcts.getActionProbs(curr_position, curr_goal))

            trainExamples.append([curr_position, curr_goal, pi])

            action = np.random.choice(len(pi), p=pi)
            curr_position = self.gym.getNextState(curr_position, action)
            # print("Step")
            if self.args.plot: self.gym.plot_env(curr_position,'g')

            r,g = self.gym.getGameEnded(curr_position, curr_goal)
            if r != 0 and self.args.plot:
                self.gym.reset_plot()

            if r == 1:
                # print("Goal Reached; Exiting")
                return [(x[0], x[1], x[2], r) for x in trainExamples]
            if r == -1:
                # print("Other Goal Reached; Exiting",goal_enum(g))
                return [(x[0], x[1], x[2], r) for x in trainExamples]
            if episodeStep > self.args.numEpisodeSteps:
                # print("Max Steps Reached")
                if self.args.plot: self.gym.reset_plot()
                return None

    def play(self):

        for _ in range(args.numIters):
            iterationTrainExamples = deque([], maxlen=self.args.maxlenOfQueue)
            print("Playing....")
            for _ in tqdm(range(self.args.numEps)):
                self.mcts = MCTS(self.gym, self.net, self.args)  # reset search tree
                states = self.executeEpisode()
                if states is not None:
                    iterationTrainExamples += states
            print("Number of training samples:",len(iterationTrainExamples))
            # print(iterationTrainExamples)
            print("Training....")

            self.net.train(iterationTrainExamples)
            print("Testing....")

            self.test()


    
    def test(self):
        accuracy = 0
        for _ in range(self.args.numEps):
            self.mcts = MCTS(self.gym, self.net, self.args)  # reset search tree
            states = self.executeEpisode()
            if states is not None:
                if states[0][3]==1:
                    accuracy += 1
        print("Accuracy = ",accuracy/self.args.numEps)




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train MCTS model')

    parser.add_argument('--dataset_folder', type=str, default='/dataset/')
    parser.add_argument('--dataset_name', type=str, default='7days1')
    parser.add_argument('--models_folder', type=str, default='/saved_models/')
    parser.add_argument('--model_weights', type=str, default=None)

    parser.add_argument('--obs', type=int, default=20)
    parser.add_argument('--preds', type=int, default=120)
    parser.add_argument('--preds_step', type=int, default=10)
    parser.add_argument('--delim', type=str, default=' ')

    parser.add_argument('--input_size', type=int, default=3)
    parser.add_argument('--num_channels', type=int, default=1)
    parser.add_argument('--channel_size', type=int, default=[256,256,128])
    parser.add_argument('--kernel_size', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.05)

    parser.add_argument('--numMCTS', type=int, default=50)
    parser.add_argument('--cpuct', type=int, default= 1)

    parser.add_argument('--numEpisodeSteps', type=int, default=20)
    parser.add_argument('--maxlenOfQueue', type=int, default=12800)
    parser.add_argument('--numEps', type=int, default=1)
    parser.add_argument('--numIters', type=int, default=5)

    parser.add_argument('--epochs', type=int, default=5)

    parser.add_argument('--plot', type=bool, default=False)






    args = parser.parse_args()

    datapath = os.getcwd() + args.dataset_folder + args.dataset_name + "/processed_data/"
    Play(datapath, args)
