from utils import get_hash
import math
EPS = 1e-8

class MCTS():
    """
    This class handles the MCTS tree.
    """

    def __init__(self, gym, nnet):
        self.gym = gym
        # self.nnet = nnet
        self.Qsa = {}  # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}  # stores #times edge s,a was visited
        self.Ns = {}  # stores #times board s was visited
        self.Ps = {}  # stores initial policy (returned by neural net)

        self.Es = {}  # stores game.getGameEnded ended for board s
        self.Vs = {}  # stores game.getValidMoves for board s


    def getActionProbs(self,curr_position,goal_postion):

        self.search(curr_position,goal_postion)


    def search(self,curr_position,goal_position):
       
        s = get_hash(curr_position)

        if s not in self.Es:
            self.Es[s] = self.gym.getGameEnded(curr_position,goal_position)
        if self.Es[s] != 0:
            # terminal node
            return -self.Es[s]

        if s not in self.Ps:
            # leaf node
            # self.Ps[s], v = self.nnet.predict(canonicalBoard)
            self.Ps[s], v = self.nnet.predict(curr_position)
            self.Ns[s] = 0
            return -v

        cur_best = -float('inf')
        best_act = -1

        # pick the action with the highest upper confidence bound
        for a in range(self.gym.getActionSize()):
            if (s, a) in self.Qsa:
                u = self.Qsa[(s, a)] + self.args.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s]) / (
                        1 + self.Nsa[(s, a)])
            else:
                u = self.args.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s] + EPS)  # Q = 0 ?

            if u > cur_best:
                cur_best = u
                best_act = a

        a = best_act
        next_position = self.gym.getNextState(curr_position, a)

        v = self.search(next_position)

        if (s, a) in self.Qsa:
            self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + v) / (self.Nsa[(s, a)] + 1)
            self.Nsa[(s, a)] += 1

        else:
            self.Qsa[(s, a)] = v
            self.Nsa[(s, a)] = 1

        self.Ns[s] += 1
        return -v 