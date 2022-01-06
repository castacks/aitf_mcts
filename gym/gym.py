import argparse
import os
from matplotlib import pyplot as plt
from costmap import CostMap

from gym.dataset_loader import TrajectoryDataset
from gym.dataset_utils import seq_collate_old
from gym.utils import populate_traj_lib, direction_goal_detect, goal_eucledian_list, goal_enum
from torch.utils.data import DataLoader
import torch
import numpy as np
from scipy.spatial.transform import Rotation as R
THRESH = 6.0

class Gym():

    def __init__(self, datapath, args):

        self.datapath = datapath
        self.args = args
        self.load_action_space()
        self.costpath = os.getcwd() + args.dataset_folder + '111_days' + "/processed_data/train"

        self.costmap = CostMap(self.costpath)
        if self.args.use_trajair:
            self.load_trajair()
        
        self.goal_list = goal_eucledian_list()
        self.hh = []


        if self.args.plot:
            self.fig = plt.figure()
            self.sp = self.fig.add_subplot(111)
            self.fig.show()
            self.fig_count = 965
        

    def get_cost(self,curr_position,curr_goal):
        
         # input data sample
        cost = 0.0
        for i in range(3,curr_position.shape[0]):
            x = curr_position[i,0].item() #(km)
            y = curr_position[i,1].item() #(km)
            z = curr_position[i,2].item() #(km)
            yaw_diff = curr_position[i,:] - curr_position[i-3,:]
            slope = torch.atan2(yaw_diff[1],yaw_diff[0])
            if goal_enum(curr_goal) == 'R2':
                wind = 1
            else:
                wind = -1

            angle = slope*180/np.pi #degrees
            # angle = 0
            if x>-0.2 and x<1.6 and abs(y) <0.5 :
                # print("Witin")
                cost +=  -1

            # try :
            cost += self.costmap.state_value(x, y, z, angle, wind) 
            # cost += c if 

            # except:
                # cost += -1


        return cost/((curr_position.shape[0]-3))

    def load_action_space(self):

        self.traj_lib, self.index_lib = populate_traj_lib()

    def load_trajair(self):

        dataset_train = TrajectoryDataset(self.datapath + "test", obs_len=self.args.obs,
                                          pred_len=self.args.preds, step=self.args.preds_step, delim=self.args.delim)
        self.loader_train = DataLoader(
            dataset_train,  batch_size=1, num_workers=1, shuffle=True, collate_fn=seq_collate_old)

    def get_random_start_position(self):

        if self.args.use_trajair:
            obs_traj, pred_traj, obs_traj_rel, pred_traj_rel, context, seq_start = next(iter(self.loader_train))
            return obs_traj[:, 0, :]  ##select one agent

        angle = np.deg2rad(np.random.randint(-180,180))
        x, y = THRESH*np.cos(angle+np.pi),THRESH*np.sin(angle+np.pi)
        z = 1.0
        r = R.from_euler('z', angle)
        direction_matrx_rep = np.squeeze(r.as_matrix())
        trajs = (np.dot(direction_matrx_rep,self.traj_lib[2]) + (np.array([x,y,z])[:,None])).T
        return torch.from_numpy(trajs).float()

    def get_random_goal_location(self, num_goals=10,p = None):

        return torch.from_numpy(np.eye(num_goals)[np.random.choice(num_goals, 1, p = p)]).float()

    def get_valid_start_goal(self):
            
        while True:

            start_position = self.get_random_start_position()
            # self.gym.plot_env(curr_position)
            # start_position = copy.deepcopy(curr_position)
            curr_goal = self.get_random_goal_location(p = [0,0,0,0,0,0,0,0,0,1])

            r,g = self.getGameEnded(start_position, curr_goal)
            if r == 0 and np.linalg.norm(start_position[-1,:2]) > 4:
                break ##make sure start is not goal
            # else:
                # print("No viable start")
           
        return start_position, curr_goal

    def getActionSize(self):

        action_space_size = self.traj_lib.shape[0]
        return action_space_size

    def getGameEnded(self, curr_position, goal_position):
        for i in range(3,curr_position.shape[0]):

            current_pos = curr_position[i, :]  # check shape of traj input
            second_pos = curr_position[i-3,:] #if i>3 else current_pos ##bug at zero
            dir_array = direction_goal_detect(current_pos,second_pos)
            if (dir_array == goal_position).all(): ##wanted goal
                return 1,dir_array
            if (dir_array.any()): ##unwanted goal
                return 0, dir_array
            
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
        return np.linalg.norm(curr_position[-1,:]-pos)
    
    def get_heuristic_dw(self, curr_position, curr_goal):
        # x = np.array([[ 0.0 ,43.32168421052632 ,86.64336842105263 ,129.96505263157894 ,173.28673684210526 ,216.6084210526316 ,259.93010526315794 ,303.2517894736842 ,346.5734736842105 ,389.89515789473677 ,433.2168421052632 ,476.5385263157894 ,519.8602105263158, 563.1818947368421, 606.5035789473684, 649.8252631578947, 693.146947368421, 736.4686315789475 ,779.7903157894735 ,823.1120000000001 ],[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ],[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ]])
        # x[0] = x[0] + 627.0
        x = np.arange(0.7,6.0,43.321/1000)
        y = np.repeat(-1.5,len(x))
        z = np.repeat(0.5,len(x))
        self.traj = np.vstack((x,y,z)).transpose()
        # print(self.traj.shape,curr_position.shape)
        idx_closest = np.argmin(np.linalg.norm(self.traj-np.tile(curr_position[0,:],(len(y),1)),axis=1))
        idx = min(idx_closest+20,len(x)-1)
        return np.linalg.norm(curr_position[-1,:]-self.traj[idx,:])
        

    def plot_env(self, curr_position,color='r',save=False,goal_position=None):
     
        self.sp.grid(True)
        if color == 'r':
            self.hh.append(self.sp.plot(curr_position[:, 0], curr_position[:, 1], color=color))
        if color != 'r':
            # self.reset_plot()
            for h in self.hh: 
                if len(h) != 0 :
                    h.pop(0).remove() 
            # for h in self.hh:
            #     
            self.sp.plot(curr_position[:, 0], curr_position[:, 1], color=color)
            if curr_position[-1,2] < 0.45:
                alt = 'r'
            elif curr_position[-1,2] > 0.7:
                alt = 'b'
            else:
                alt = 'g'    
            self.sp.scatter(curr_position[-1, 0], curr_position[-1, 1], color=alt)
        self.sp.scatter(0, 0, color='k')
        self.sp.scatter(1.45, 0, color='k')
        if goal_position is not None:
            print(goal_enum(goal_position))
            plt.text(0,-1,"Goal: " + goal_enum(goal_position)[0])
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
            plt.pause(0.01)


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
    print(goal_eucledian_list())

    gym = Gym(datapath, args)

