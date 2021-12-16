import argparse
import os
from matplotlib import pyplot as plt

from gym.dataset_loader import TrajectoryDataset
from gym.dataset_utils import seq_collate_old
from gym.utils import populate_traj_lib, direction_goal_detect, goal_eucledian_list
from torch.utils.data import DataLoader
import torch
import numpy as np
from scipy.spatial.transform import Rotation as R


class Gym():

    def __init__(self, datapath, args):

        self.datapath = datapath
        self.args = args
        self.load_action_space()
        self.load_trajair()
        self.goal_list = goal_eucledian_list()
        if self.args.plot:
            self.fig = plt.figure()
            self.sp = self.fig.add_subplot(111)
            self.fig.show()
            self.fig_count = 0
        

    def load_action_space(self):

        self.traj_lib, self.index_lib = populate_traj_lib()

    def load_trajair(self):

        dataset_train = TrajectoryDataset(self.datapath + "test", obs_len=self.args.obs,
                                          pred_len=self.args.preds, step=self.args.preds_step, delim=self.args.delim)
        self.loader_train = DataLoader(
            dataset_train,  batch_size=1, num_workers=1, shuffle=True, collate_fn=seq_collate_old)

    def get_random_start_position(self):

        obs_traj, pred_traj, obs_traj_rel, pred_traj_rel, context, seq_start = next(iter(self.loader_train))

        return obs_traj[:, 0, :]  ##select one agent

    def get_random_goal_location(self, num_goals=10,p = None):

        return torch.from_numpy(np.eye(num_goals)[np.random.choice(num_goals, 1, p = p)]).float()

    def get_valid_start_goal(self):
            
        while True:

            start_position = self.get_random_start_position()
            # self.gym.plot_env(curr_position)
            # start_position = copy.deepcopy(curr_position)
            curr_goal = self.get_random_goal_location(p=[0,0,0,0,0,0,0,0,1,0])

            r,g = self.getGameEnded(start_position, curr_goal)
            if r == 0:
                break ##make sure start is not goal
            # else:
                # print("No viable start")
        return start_position, curr_goal

    def getActionSize(self):

        action_space_size = self.traj_lib.shape[0]
        return action_space_size

    def getGameEnded(self, curr_position, goal_position):
        for i in range(curr_position.shape[0]):

            current_pos = curr_position[i, :]  # check shape of traj input
            second_pos = curr_position[i-3,:] if i>3 else current_pos
            dir_array = direction_goal_detect(current_pos,second_pos)
            if (dir_array == goal_position).all(): ##wanted goal
                return 1,dir_array
            if (dir_array.any()): ##unwanted goal
                return -1, dir_array
            
        return 0 ,dir_array## no goal

    def getNextState(self, curr_position, action_choice):
        # rotate and translate action choice to end of previous executed traj
        difference = curr_position[-1, :] - curr_position[-3, :]
        difference = difference.cpu().numpy()  # verify device
        angle = np.arctan2(difference[1], difference[0])
        r = R.from_euler('z', angle)
        direction_matrx_rep = np.squeeze(r.as_matrix())
        trajs = (np.dot(direction_matrx_rep,self.traj_lib[action_choice]) + (np.array(curr_position[-1,:])[:,None])).T

    
        return torch.from_numpy(trajs).float()

    def get_hash(self, curr_position):

        # return str(curr_position[-1, 0]) + str(curr_position[-1, 1])
        return "%s-%s" % (int(curr_position[-1, 0]*1000),int(curr_position[-1, 1]*1000))

    def reset_plot(self):
        plt.pause(2)
        plt.close()
        self.fig = plt.figure()
        self.sp = self.fig.add_subplot(111)
        self.fig.show()

    def get_heuristic(self, curr_position, curr_goal):

        pos = self.goal_list[np.argmax(curr_goal.numpy())]
    
        return np.linalg.norm(curr_position[-1,:2]-pos)

    def plot_env(self, curr_position,color='r',save=False):
     
        self.sp.grid(True)
        self.sp.plot(curr_position[:, 0], curr_position[:, 1], color=color)
        self.sp.scatter(curr_position[-1, 0], curr_position[-1, 1], color='b')
        self.sp.scatter(0, 0, color='k')
        self.sp.scatter(1.45, 0, color='k')
        plt.plot([0, 1.450], [0, 0], '--', color='k')
        plt.axis("equal")
        plt.grid(True)
        plt.xlim([-7, 7])
        plt.ylim([-7, 7])

        if save:
            plt.savefig("mcts_"+str(self.fig_count) + ".png")
            self.fig_count += 1
        else:
            self.fig.show()
            plt.pause(0.1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train MCTS model')

    parser.add_argument('--dataset_folder', type=str, default='/dataset/')
    parser.add_argument('--dataset_name', type=str, default='7days1')
    parser.add_argument('--obs', type=int, default=20)
    parser.add_argument('--preds', type=int, default=120)
    parser.add_argument('--preds_step', type=int, default=10)
    parser.add_argument('--delim', type=str, default=' ')
    parser.add_argument('--plot', type=bool, default=True)

    args = parser.parse_args()

    datapath = os.getcwd() + args.dataset_folder + args.dataset_name + "/processed_data/"
    gym = Gym(datapath, args)

    print(goal_eucledian_list())
