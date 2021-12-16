
import os
from pickle import Pickler, Unpickler
from glob import glob
from collections import deque
from mcts.mcts import MCTS
from gym.utils import goal_enum
import numpy as np

def run_episode(rank,gym,net,args):
    mcts = MCTS(gym, net, args)

    curr_position , curr_goal = gym.get_valid_start_goal()

    print("curr goal",goal_enum(curr_goal))
    # # print(curr_position,curr_goal)
    # curr_position = torch.Tensor([[ 0.1519,  0.3398,  0.2845],
    # [ 0.1379,  0.2875,  0.2859],
    # [ 0.1238,  0.2352,  0.2872],
    # [ 0.1098,  0.1829,  0.2885],
    # [ 0.0958,  0.1306,  0.2899],
    # [ 0.0818,  0.0783,  0.2912],
    # [ 0.0677,  0.0260,  0.2925],
    # [ 0.0537, -0.0263,  0.2939],
    # [ 0.0397, -0.0785,  0.2952],
    # [ 0.0257, -0.1308,  0.2965],
    # [ 0.0116, -0.1831,  0.2979],
    # [-0.0024, -0.2354,  0.2992],
    # [-0.0164, -0.2877,  0.3005],
    # [-0.0304, -0.3400,  0.3019],
    # [-0.0445, -0.3923,  0.3032],
    # [-0.0585, -0.4446,  0.3046],
    # [-0.0725, -0.4968,  0.3059],
    # [-0.0866, -0.5491,  0.3072],
    # [-0.1006, -0.6014,  0.3086],
    # [-0.1146, -0.6537,  0.3099]])
    # curr_goal = torch.Tensor([[0., 0., 1., 0., 0., 0., 0., 0., 0., 0.]])
    # print(self.gym.getGameEnded(curr_position, curr_goal))
    trainExamples = []
    episodeStep = 0

    while True:
        episodeStep += 1
        # print(episodeStep,rank)
        pi = mcts.getActionProbs(curr_position, curr_goal)
        # if pi == None:
            # print(curr_position,curr_goal,start_position)
            # pi = np.ones_like(counts)/self.gym.getActionSize()

        pi = np.squeeze(pi)
        trainExamples.append([curr_position, curr_goal, pi])

        action = np.random.choice(len(pi), p=pi)
        curr_position = gym.getNextState(curr_position, action)
        print(curr_position[-1,2]*3280.84,1000*gym.get_cost(curr_position))
        # print("Step")
        if args.plot: gym.plot_env(curr_position,'g')

        r,g = gym.getGameEnded(curr_position, curr_goal)
        if r != 0 and args.plot:
            gym.reset_plot()

        if r == 1:
            print("Goal Reached; Exiting")
            return [(x[0], x[1], x[2], r) for x in trainExamples]
        if r == -1:
            print("Other Goal Reached; Exiting",goal_enum(g))
            return [(x[0], x[1], x[2], r) for x in trainExamples]
        if episodeStep > args.numEpisodeSteps:
            print("Max Steps Reached")
            if args.plot: gym.reset_plot()
            return None
            
def save_episodes(checkpoint,iterationTrainExamples,ep):
    print("Saving Episode for: ",ep)
    folder = os.getcwd() + checkpoint
    if not os.path.exists(folder):
        os.makedirs(folder)
    filename = os.path.join(folder, "episodes_" + str(ep) + ".examples")
    with open(filename, "wb+") as f:
        Pickler(f).dump(iterationTrainExamples)
    f.closed

def load_episodes(checkpoint):
    iterationTrainExamples = deque([])
    folder = os.getcwd() + checkpoint

    filelist = [y for x in os.walk(folder) for y in glob(os.path.join(x[0], '*.examples'))]
    # print(filelist,folder)
    for examplesFile in filelist:
        with open(examplesFile, "rb") as f:
            states = Unpickler(f).load()
            if states is not None:
                iterationTrainExamples += states
    print("Loaded Episodes:", len(iterationTrainExamples))
    return iterationTrainExamples