import numpy as np
import scipy
import math
import torch
from tqdm import tqdm
from line_profiler import LineProfiler

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





    def getActionProbs(self, curr_position, goal_postion, temp=1):

        # return np.eye(self.gym.getActionSize())[np.random.choice(self.gym.getActionSize(), 1)]

        # for i in tqdm(range(self.args.numMCTS), desc="MCTS Trees"):
        for i in (range(self.args.numMCTS)):
            # print("MCTS Tree #" + str(i))
            self.search(curr_position, goal_postion)

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
    def search(self, curr_position, goal_position):

        s = self.gym.get_hash(curr_position)

        if s not in self.Es:
            self.Es[s],_ = self.gym.getGameEnded(curr_position, goal_position)
        if self.Es[s] != 0:
            # terminal node
            # print("Terminal Node")
            return 10.0*self.Es[s]

        if s not in self.Ps:
            # leaf node
            v =  self.gym.get_cost(curr_position)
            self.gym.plot_env(curr_position,'r',save=False)
            # print(curr_position[-1,2]*3280.84,v)
            curr_position = curr_position.to(self.device)*1000 ##km to m
            goal_position = goal_position.to(self.device)

            self.Ps[s], _ = self.nnet.predict(curr_position, goal_position)

            self.Ns[s] = 0
            # print("Leaf")
            return v

        cur_best = -float('inf')
        best_act = -1
        h = np.zeros(self.gym.getActionSize())
        
        for a in range(self.gym.getActionSize()):
            next_state = self.gym.getNextState(curr_position,a)
            h[a] = 1.0/self.gym.get_heuristic(next_state,goal_position)
        h = scipy.special.softmax(h)


        # pick the action with the highest upper confidence bound
        for a in range(self.gym.getActionSize()):
            if (s, a) in self.Qsa:
                u = self.Qsa[(s, a)] + h[a] + self.args.cpuct * self.Ps[s][a] * (math.sqrt(self.Ns[s]) / (1 + self.Nsa[(s, a)])) 
                # print(self.Qsa[(s, a)] , 0.1*h[a])
            else:
                u = h[a] + self.args.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s] + EPS)  
                # print(0.1*h[a])

            if u > cur_best:
                cur_best = u
                best_act = a

        a = best_act
        next_position = self.gym.getNextState(curr_position, a)

        v = self.search(next_position, goal_position)
        # print("test")
        if (s, a) in self.Qsa:
            self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + v) / (self.Nsa[(s, a)] + 1)
            self.Nsa[(s, a)] += 1

        else:
            self.Qsa[(s, a)] = v
            self.Nsa[(s, a)] = 1

        self.Ns[s] += 1
        return v
