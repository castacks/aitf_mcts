import numpy as np
import scipy
import math
import torch
from tqdm import tqdm
from line_profiler import LineProfiler
import time
from matplotlib import pyplot as plt
from mcts.stl_specs import monitor_R2
from collections import deque
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

        self.stack = deque()

    def getActionProbs(self, curr_position, goal_postion, temp=1, max_time = None, history=None):
        self.history = history

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
        if self.stack:
            print("stack len", len(self.stack))
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
            if self.stack:
                temp_stack_plot = np.concatenate(self.stack, axis=0)
                # print(temp_stack_plot.shape,temp_stack_plot[:,0:2])
            # v = 0.5
            # if self.args.changeh:
            self.gym.plot_env(curr_position, 'r')
            # plt.plot(curr_position[:, 0], curr_position[:, 1], color='b')

            # print(curr_position[-1,2]*3280.84,v)
            curr_position = curr_position.to(self.device) ##km to m
            goal_position = goal_position.to(self.device)

            pred = self.nnet.predict(curr_position, goal_position)
            # print(curr_position,pred)
            self.gym.plot_env(np.transpose(pred),'k')
            all_next_states = self.gym.getAllNextStates(curr_position.cpu()) # added a copy to cpu since Allnextstates also performs numpy operations
            # print(curr_position,all_next_states[0],pred)
            # for i in range(30):
            #     self.gym.plot_env(np.transpose(all_next_states[i]),'k')
            self.Ps[s] = self.gym.traj_to_action(pred[:,:2],all_next_states)
            # print(all_next_states.shape) 
            # print(np.argmax(self.Ps[s]))
            # motion = self.gym.getNextState(curr_position, np.argmax(self.Ps[s]))

            # self.gym.plot_env((motion),'c')

            self.Ns[s] = 0
            # print("Leaf")
            for _ in range(len(self.stack)):
                # print("poppin")
                self.stack.pop()
            return v,h

        cur_best = -float('inf')
        best_act = -1
        heu = np.zeros(self.gym.getActionSize())
        for a in range(self.gym.getActionSize()):

            next_state = self.gym.getNextState(curr_position,a)
            # print("next_state",next_state.shape,heu.shape)
            R1 =  torch.tensor([[0., 0., 0., 0., 0., 0., 0., 0., 1., 0.]],dtype=torch.uint8)
            R2 =  torch.tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]],dtype=torch.uint8)
            goal_position = goal_position.type(torch.uint8)
            # print(type(goal_position),type(R1))
            # if torch.equal(goal_position, R1):
            # 	# print("goals r1",goal_position)
            # 	# heurist = MCTS_STL_spec_R1.evaluate(next_state,0,0)
            # 	heurist = monitor_R1(next_state)

            if torch.equal(goal_position, R2):
                # print("goals r2",goal_position)
                if self.history:
                    temp_old_state = np.concatenate(self.history, axis=0)
                    if self.stack:
                        temp_stack_state = np.concatenate(self.stack, axis=0)
                        stack_added_state = np.concatenate((temp_old_state,temp_stack_state))

                        # print(a,"temp_old_state",temp_old_state.shape, "temp_stack_state",temp_stack_state.shape,"history",len(self.history))
                        temp_state = np.concatenate((stack_added_state,next_state))
                    else:
                        temp_state = np.concatenate((temp_old_state,next_state))
                    # print(a,"temp_state",temp_state.shape, "this temp state")
                    # heurist = MCTS_STL_spec_R2.evaluate(temp_state,0,0)
                    temp_state = temp_state[::5,:]
                    heurist = monitor_R2(temp_state)
                else:
                    # heurist = MCTS_STL_spec_R2.evaluate(next_state,0,0)
                    # print("no his")
                    if self.stack:
                        temp_stack_state = np.concatenate(self.stack, axis=0)
                        # print(a,"temp_stack_state",temp_stack_state.shape)
                        # print(temp_stack_state[0:3,0:2])
                        # print(temp_stack_state[-4:-1,0:2])
                        # print(next_state[0:3,0:2])
                        temp_state = np.concatenate((temp_stack_state,next_state))
                        # print(a, "temp_state",temp_state.shape, "this temp state")
                        heurist = monitor_R2(temp_state)
                    else:
                        heurist = monitor_R2(next_state)
                
            heu[a] = heurist
            # print("len",len(self.old_state))
            # print("temp state",temp_state.shape,"temp_old_state",temp_old_state.shape)
            goal_position = goal_position.type(torch.float)
            # print("heu",a, heu[a])
            # if heu[a]>-0.1:
                # print("heu pos",a, heurist)
            # print(heur)          
        # print(np.linalg.norm(heu))
        heu = heu / np.linalg.norm(heu)
        heu = scipy.special.softmax(heu)
        # pick the action with the highest upper confidence bound
        for a in range(self.gym.getActionSize()):
            if (s, a) in self.Qsa:
                u = self.Qsa[(s, a)]  + self.args.cpuct * self.Ps[s][a] * (math.sqrt(self.Ns[s]) / (1 + self.Nsa[(s, a)])) + self.args.huct*self.Hsa[(s, a)] 
                # print(u,a)
            else:
                u =  self.args.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s] + EPS)  + self.args.huct*heu[a] 
                # print(u,a)

                # print(0.1*h[a])
            # print(u)
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
