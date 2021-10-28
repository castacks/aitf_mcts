
from utils import TrajectoryDataset, seq_collate

from torch.utils.data import DataLoader

import numpy as np
from scipy.spatial.transform import Rotation as R

from net import *

class Gym():

	def __init__(self,datapath,args):
		
		self.datapath = datapath
		self.args = args
		self.load_trajair()
		self.load_action()
		self.traj_lib, self.index_lib = populate_traj_lib()

	def load_action(self):

		# to load model here
		model = TCN(input_channels, output_size, channel_sizes, kernel_size=kernel_size, dropout=dropout)
		model.to(device)

		checkpoint = torch.load('goalGAIL2_60.pt',map_location=torch.device('cpu'))
		model.load_state_dict(checkpoint['model_state_dict'], strict=False)

		softmax_out, output, latent_space_, multi_head = model(torch.transpose(obs_traj_all,1,2),torch.transpose(pred_traj_all,1,2),goal_position,torch.transpose(context,1,2),sort=True)

		traj_lib_choice = torch.argmax(recon_x,dim=0) # dim = 0 if single traj in a batch
		return traj_lib_choice


	def load_trajair(self):

		dataset_train = TrajectoryDataset(self.datapath + "test", obs_len=self.args.obs, pred_len=self.args.preds, step=self.args.preds_step, delim=self.args.delim)
		self.loader_train = DataLoader(dataset_train,batch_size=1,num_workers=4,shuffle=True,collate_fn=seq_collate)

	def get_random_start_position(self):

    def load_action(self):

		obs_traj , pred_traj, obs_traj_rel, pred_traj_rel, context, seq_start,goal_position,full_l2 = next(iter(self.loader_train)) 
		return obs_traj

	def get_random_goal_location(self,num_goals = 10):
		
		return  np.eye(num_goals)[np.random.choice(num_goals, 1)]

	def getNextState(self):

		action_choice = load_action()
		obs_traj = get_random_start_position()

		# rotate and translate action choice to end of previous executed traj
		difference = obs_traj[-1,:,:]-obs_traj[-3,:,:]
		difference = difference.cpu().numpy() # verify device
		unit_vector_1 = difference[:,0:2] / np.linalg.norm(difference[:,0:2])
		angle = np.arctan2(difference[:,1],difference[:,0])
		r = R.from_euler('z',angle)
		direction_matrx_rep = r.as_matrix()

		trajs = np.zeros([20,3])
		for j in range(20):
			trajs[j,0:2] = torch.from_numpy(np.dot(direction_matrx_rep[0:2,0:2],self.traj_lib[action_choice,0:2,j].T))
		for i in range(20):
			trajs[i,0]+=obs_traj[-1,0]
			trajs[i,1]+=obs_traj[-1,1]
			trajs[i,2]=self.traj_lib[traj_lib_choice,2,i]+obs_traj[-1,2]

    def getNextState(self,curr_position,action):
		# trajs is a numpy array
		return trajs

	def getActionSize(self):

		action_space_size = self.traj_lib.shape[0]
		return action_space_size
