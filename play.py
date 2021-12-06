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


# torch.manual_seed(5345)
# import random
# random.seed(5345)
# np.random.seed(5345)

class Play():

    def __init__(self, datapath, args):

        self.args = args
        self.net = Net(args)
        self.gym = Gym(datapath, args)
        self.mcts = MCTS(self.gym, self.net, args)
        self.play()

    def executeEpisode(self):

        while True:

            curr_position = self.gym.get_random_start_position()
            # self.gym.plot_env(curr_position)
            start_position = copy.deepcopy(curr_position)
            curr_goal = self.gym.get_random_goal_location()

            r,g = self.gym.getGameEnded(curr_position, curr_goal)
            if r == 0:
                break ##make sure start is not goal
            # else:
                # print("No viable start")

        # print("curr goal",goal_enum(curr_goal))
        # print(curr_position,curr_goal)
        curr_position = torch.Tensor([[ 0.1519,  0.3398,  0.2845],
        [ 0.1379,  0.2875,  0.2859],
        [ 0.1238,  0.2352,  0.2872],
        [ 0.1098,  0.1829,  0.2885],
        [ 0.0958,  0.1306,  0.2899],
        [ 0.0818,  0.0783,  0.2912],
        [ 0.0677,  0.0260,  0.2925],
        [ 0.0537, -0.0263,  0.2939],
        [ 0.0397, -0.0785,  0.2952],
        [ 0.0257, -0.1308,  0.2965],
        [ 0.0116, -0.1831,  0.2979],
        [-0.0024, -0.2354,  0.2992],
        [-0.0164, -0.2877,  0.3005],
        [-0.0304, -0.3400,  0.3019],
        [-0.0445, -0.3923,  0.3032],
        [-0.0585, -0.4446,  0.3046],
        [-0.0725, -0.4968,  0.3059],
        [-0.0866, -0.5491,  0.3072],
        [-0.1006, -0.6014,  0.3086],
        [-0.1146, -0.6537,  0.3099]])
        curr_goal = torch.Tensor([[0., 0., 1., 0., 0., 0., 0., 0., 0., 0.]])
        print(self.gym.getGameEnded(curr_position, curr_goal))
        trainExamples = []
        episodeStep = 0

        while True:
            episodeStep += 1

            pi = self.mcts.getActionProbs(curr_position, curr_goal)
            if pi == None:
                print(curr_position,curr_goal,start_position)
                # pi = np.ones_like(counts)/self.gym.getActionSize()

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
    parser.add_argument('--load_episodes', type=bool, default=True)

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

    parser.add_argument('--epochs', type=int, default=15)

    parser.add_argument('--plot', type=bool, default=False)






    args = parser.parse_args()

    datapath = os.getcwd() + args.dataset_folder + args.dataset_name + "/processed_data/"
    Play(datapath, args)
