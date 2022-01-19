import numpy as np
import scipy
import math
import torch
from tqdm import tqdm
from line_profiler import LineProfiler
import time
from matplotlib import pyplot as plt

EPS = 1e-8


class MCTS():
    """
    This class handles the MCTS tree.
    """

    def __init__(self, gym, nnet, args):

        self.gym = gym
        self.nnet = nnet
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # print(self.device)
        self.Qsa = {}  # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}  # stores #times edge s,a was visited
        self.Ns = {}  # stores #times board s was visited
        self.Ps = {}  # stores initial policy (returned by neural net)

        self.Es = {}  # stores game.getGameEnded ended for board s
        self.Vs = {}  # stores game.getValidMoves for board s

        self.Hs = {}
        self.Hsa = {}



    def getActionProbs(self, curr_position, goal_postion, temp=1, max_time = None):

        # return np.eye(self.gym.getActionSize())[np.random.choice(self.gym.getActionSize(), 1)]

        # for i in tqdm(range(self.args.numMCTS), desc="MCTS Trees"):
        start_time = time.time()
        if max_time is None:
            for i in (range(self.args.numMCTS)):
                # print("MCTS Tree #" + str(i))
                self.search(curr_position, goal_postion,0)
        else:
            while (time.time()-start_time) < max_time:
                self.search(curr_position, goal_postion,0)


        s = self.gym.get_hash(curr_position)

        counts = [self.Nsa[(s, a)] if (s, a) in self.Nsa else 0 for a in range(self.gym.getActionSize())]

        if temp == 0:
            bestAs = np.array(np.argwhere(counts == np.max(counts))).flatten()
            bestA = np.random.choice(bestAs)
            probs = [0] * len(counts)
            probs[bestA] = 1
            return probs

        counts = [x ** (1. / temp) for x in counts]
        counts_sum = float(sum(counts))
        if int(counts_sum) != 0:
            probs = [x / counts_sum for x in counts]
        else:
            # print(self.Nsa,self.Qsa)
            print("All counts zero for ",s)
            probs = np.ones_like(counts)/self.gym.getActionSize()
            # probs = None
        return probs

    # @profile
    def search(self, curr_position, goal_position,h):
        # print("hah")
        s = self.gym.get_hash(curr_position)
        # print(s)
        if s not in self.Es:
            self.Es[s],_ = self.gym.getGameEnded(curr_position, goal_position)
        if self.Es[s] != 0:
            # terminal node
            print("Terminal Node")
            return self.Es[s],h

        if s not in self.Ps:
            # leaf node
            v =  self.gym.get_cost(curr_position,goal_position)

            # v = 0.5
            # if self.args.changeh:
            self.gym.plot_env(curr_position, 'r')
            # plt.plot(curr_position[:, 0], curr_position[:, 1], color='b')

            # print(curr_position[-1,2]*3280.84,v)
            curr_position = curr_position.to(self.device)*1000 ##km to m
            goal_position = goal_position.to(self.device)

            self.Ps[s], v = self.nnet.predict(curr_position, goal_position)

            self.Ns[s] = 0
            # print("Leaf")
            return v,h

        cur_best = -float('inf')
        best_act = -1
        heu = np.zeros(self.gym.getActionSize())
        
        for a in range(self.gym.getActionSize()):
            next_state = self.gym.getNextState(curr_position,a)
            if not self.args.changeh:
                heu[a] = 1.0/self.gym.get_heuristic(next_state,goal_position)
            else:
                heu[a] = 1.0/self.gym.get_heuristic_dw(next_state,goal_position)

        heu = scipy.special.softmax(heu)
        # u = np.zeros((self.gym.getActionSize()))
        # pick the action with the highest upper confidence bound
        for a in range(self.gym.getActionSize()):
            if (s, a) in self.Qsa:
                u = self.Qsa[(s, a)]  + self.args.cpuct * self.Ps[s][a] * (math.sqrt(self.Ns[s]) / (1 + self.Nsa[(s, a)])) + self.args.huct*self.Hsa[(s, a)] 
                # print(u,a)
            else:
                u =  self.args.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s] + EPS)  + self.args.huct*heu[a] 
                # print(u,a)

                # print(0.1*h[a])

            if u > cur_best:
                cur_best = u
                best_act = a

        a = best_act
        # u = scipy.special.softmax(u)
        # print(u.shape)
        # a = np.random.choice(self.gym.getActionSize(), p=u)

        next_position = self.gym.getNextState(curr_position, a)

        v , h = self.search(next_position, goal_position,heu[a])
        # print("test")
        if (s, a) in self.Qsa:
            self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + v) / (self.Nsa[(s, a)] + 1)
            self.Nsa[(s, a)] += 1
            self.Hsa[(s, a)] = (self.Nsa[(s, a)] * self.Hsa[(s, a)] + h) / (self.Nsa[(s, a)] + 1)


        else:
            self.Qsa[(s, a)] = v
            self.Nsa[(s, a)] = 1
            self.Hsa[(s, a)] = h

        self.Ns[s] += 1
        return v,h
