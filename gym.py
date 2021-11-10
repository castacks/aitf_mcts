import argparse
import os
from matplotlib import pyplot as plt

from utils import TrajectoryDataset
from dataset_utils import seq_collate_old
from utils import populate_traj_lib, direction_detect
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
        self.fig = plt.figure()
        self.sp = self.fig.add_subplot(111)
        self.fig.show()

    def load_action_space(self):

        self.traj_lib, self.index_lib = populate_traj_lib()

    def load_trajair(self):

        dataset_train = TrajectoryDataset(self.datapath + "test", obs_len=self.args.obs,
                                          pred_len=self.args.preds, step=self.args.preds_step, delim=self.args.delim)
        self.loader_train = DataLoader(
            dataset_train, batch_size=1, num_workers=4, shuffle=True, collate_fn=seq_collate_old)

    def get_random_start_position(self):

        # obs_traj, pred_traj, obs_traj_rel, pred_traj_rel, context, seq_start, goal_position, full_l2 = next(iter(self.loader_train))
        obs_traj, pred_traj, obs_traj_rel, pred_traj_rel, context, seq_start = next(iter(self.loader_train))

        return obs_traj[:, 0, :]  ##select one agent

    def get_random_goal_location(self, num_goals=10):

        return torch.from_numpy(np.eye(num_goals)[np.random.choice(num_goals, 1)]).float()

    def getActionSize(self):

        action_space_size = self.traj_lib.shape[0]
        return action_space_size

    def getGameEnded(self, curr_position, goal_position):
        for i in range(curr_position.shape[0]):
            current_pos = curr_position[i, :]  # check shape of traj input
            input_pos = torch.zeros(1, 3)
            dir_array = direction_detect(input_pos, current_pos)
            if (dir_array == goal_position).all():
                return 1
        return 0

    def getNextState(self, curr_position, action_choice):
        # rotate and translate action choice to end of previous executed traj
        difference = curr_position[-1, :] - curr_position[-3, :]
        difference = difference.cpu().numpy()  # verify device
        angle = np.arctan2(difference[1], difference[0])
        r = R.from_euler('z', angle)
        direction_matrx_rep = np.squeeze(r.as_matrix())

        trajs = np.zeros([20, 3])
        for j in range(20):
            trajs[j, 0:2] = torch.from_numpy(
                np.dot(direction_matrx_rep[0:2, 0:2], self.traj_lib[action_choice, 0:2, j].T))
        for i in range(20):
            trajs[i, 0] += curr_position[-1, 0]
            trajs[i, 1] += curr_position[-1, 1]
            trajs[i, 2] = self.traj_lib[action_choice, 2, i] + curr_position[-1, 2]

        return torch.from_numpy(trajs).float()

    def get_hash(self, curr_position):

        return str(curr_position[-1, 0]) + str(curr_position[-1, 1])

    def plot_env(self, curr_position):
        self.sp.grid(True)
        self.sp.plot(curr_position[:, 0], curr_position[:, 1], color='r')
        self.sp.scatter(curr_position[-1, 0], curr_position[-1, 1], color='b')
        self.sp.scatter(0, 0, color='k')
        self.sp.scatter(1.45, 0, color='k')
        plt.plot([0, 1.450], [0, 0], '--', color='k')
        plt.axis("equal")
        plt.grid(True)
        plt.xlim([-5, 5])
        self.fig.show()
        # plt.pause(0.05)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train MCTS model')

    parser.add_argument('--dataset_folder', type=str, default='/dataset/')
    parser.add_argument('--dataset_name', type=str, default='7days1')
    parser.add_argument('--obs', type=int, default=20)
    parser.add_argument('--preds', type=int, default=120)
    parser.add_argument('--preds_step', type=int, default=10)
    parser.add_argument('--delim', type=str, default=' ')

    args = parser.parse_args()

    datapath = os.getcwd() + args.dataset_folder + args.dataset_name + "/processed_data/"
    gym = Gym(datapath, args)
