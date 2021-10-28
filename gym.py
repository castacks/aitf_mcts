
from utils import TrajectoryDataset, seq_collate

from torch.utils.data import DataLoader

import numpy as np

class Gym():

    def __init__(self,datapath,args):
        
        self.datapath = datapath
        self.args = args
        self.load_trajair()
        self.load_action()

    def load_action(self):


    def load_trajair(self):

        dataset_train = TrajectoryDataset(self.datapath + "test", obs_len=self.args.obs, pred_len=self.args.preds, step=self.args.preds_step, delim=self.args.delim)
        self.loader_train = DataLoader(dataset_train,batch_size=1,num_workers=4,shuffle=True,collate_fn=seq_collate)

    def get_random_start_position(self):

        obs_traj , pred_traj, obs_traj_rel, pred_traj_rel, context, seq_start = next(iter(self.loader_train)) 
        return obs_traj

    def get_random_goal_location(self,num_goals = 10):
        
        return  np.eye(num_goals)[np.random.choice(num_goals, 1)]

    def getNextState(self,curr_position,action):

        pass

    def getActionSize(self):
        pass