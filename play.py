import argparse
import copy
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
from pickle import Pickler, Unpickler
from glob import glob


torch.manual_seed(5345)
import random
random.seed(5345)
np.random.seed(5345)

class Play():

    def __init__(self, datapath, args):

        self.args = args
        self.net = Net(args)
        self.gym = Gym(datapath, args)
        self.mcts = MCTS(self.gym, self.net, args)
        # self.train()
        self.play()
        # self.executeEpisode()

    def executeEpisode(self):

        while True:

            curr_position = self.gym.get_random_start_position()
            # self.gym.plot_env(curr_position)
            start_position = copy.deepcopy(curr_position)
            curr_goal = self.gym.get_random_goal_location()

            r,g = self.gym.getGameEnded(curr_position, curr_goal)
            if r == 0:
                break ##make sure start is not goal
            else:
                print("No viable start")

        # print("curr goal",goal_enum(curr_goal))
        # print(curr_position,curr_goal)
        # curr_position = torch.Tensor([[-0.1239, -0.0032,  0.4420],
        # [-0.1519, -0.0048,  0.4496],
        # [-0.1855, -0.0069,  0.4572],
        # [-0.2022, -0.0066,  0.4572],
        # [-0.2343, -0.0068,  0.4572],
        # [-0.2684, -0.0050,  0.4648],
        # [-0.2864, -0.0098,  0.4648],
        # [-0.3366, -0.0101,  0.4648],
        # [-0.3617, -0.0133,  0.4724],
        # [-0.3889, -0.0152,  0.4724],
        # [-0.4345, -0.0215,  0.4724],
        # [-0.4622, -0.0232,  0.4801],
        # [-0.4913, -0.0296,  0.4801],
        # [-0.5275, -0.0344,  0.4801],
        # [-0.5557, -0.0364,  0.4801],
        # [-0.5827, -0.0427,  0.4877],
        # [-0.6207, -0.0476,  0.4877],
        # [-0.6383, -0.0526,  0.4877],
        # [-0.6673, -0.0569,  0.4902],
        # [-0.6962, -0.0613,  0.4928]])
        # curr_goal = torch.Tensor([[0., 0., 1., 0., 0., 0., 0., 0., 0., 0.]])
        trainExamples = []
        episodeStep = 0

        while True:
            episodeStep += 1

            pi = self.mcts.getActionProbs(curr_position, curr_goal)
            if pi == None:
                print(curr_position,curr_goal,start_position)
            pi = np.squeeze(pi)
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
        
        iterationTrainExamples = deque([], maxlen=self.args.maxlenOfQueue)

        for ite in range(args.numIters):
            self.net.nnet.eval()

            print("Playing....")
            if ite == 0 and self.args.load_episodes:
                iterationTrainExamples = self.load_episodes()
            else:     
                for ep in tqdm(range(self.args.numEps),"Episode"):
                    if  (ep%100) == 0:
                        print("Eps",ep/self.args.numEps)
                        if ite==0:
                            self.save_episodes(iterationTrainExamples,ep)


                    self.mcts = MCTS(self.gym, self.net, self.args)  # reset search tree
                    states = self.executeEpisode()
                    if states is not None:
                        iterationTrainExamples += states
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
        for _ in range(self.args.numEpsTest):
            self.mcts = MCTS(self.gym, self.net, self.args)  # reset search tree
            states = self.executeEpisode()
            if states is not None:
                if states[0][3]==1:
                    accuracy += 1
        print("Accuracy = ",accuracy/self.args.numEpsTest)

    def save_episodes(self,iterationTrainExamples,ep):
        print("Saving Episodes..")
        folder = os.getcwd() + self.args.checkpoint
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = os.path.join(folder, "episodes_" + str(ep) + ".examples")
        with open(filename, "wb+") as f:
            Pickler(f).dump(iterationTrainExamples)
        f.closed

    def load_episodes(self):
        # iterationTrainExamples = deque([], maxlen=self.args.maxlenOfQueue)
        folder = os.getcwd() + self.args.checkpoint

        filelist = [y for x in os.walk(folder) for y in glob(os.path.join(x[0], '*.examples'))]
        print(filelist,folder)
        for examplesFile in filelist:
            with open(examplesFile, "rb") as f:
                iterationTrainExamples = Unpickler(f).load()
                # if trainExamplesHistory is not None:
                    # iterationTrainExamples = trainExamplesHistory
        print("Loaded Episodes:", len(iterationTrainExamples))
        return iterationTrainExamples




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

    parser.add_argument('--numMCTS', type=int, default=50)
    parser.add_argument('--cpuct', type=int, default= 1)

    parser.add_argument('--numEpisodeSteps', type=int, default=20)
    parser.add_argument('--maxlenOfQueue', type=int, default=25600)
    parser.add_argument('--numEps', type=int, default=1000)
    parser.add_argument('--numEpsTest', type=int, default=100)

    parser.add_argument('--numIters', type=int, default=20)

    parser.add_argument('--epochs', type=int, default=25)

    parser.add_argument('--plot', type=bool, default=False)






    args = parser.parse_args()

    datapath = os.getcwd() + args.dataset_folder + args.dataset_name + "/processed_data/"
    Play(datapath, args)
