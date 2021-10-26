import math
import os
from tqdm import tqdm

from torch import nn
import torch
from torch.utils.data import Dataset
from torch.nn.utils import weight_norm

import numpy as np
from scipy.spatial import distance_matrix
import random

from motion_primitives_no_azi import *


##Dataloader class

class TrajectoryDataset(Dataset):
	"""Dataloder for the Trajectory datasets"""
	def __init__(
		self, data_dir, traj_lib,index_lib,obs_len=20,pred_len=20, skip=1,step=1,
		min_ped=0, delim=' '):
		"""
		Args:
		- data_dir: Directory containing dataset files in the format
		<time> <id> <x> <y> <z> <wind_x>  <wind_y>,
		- obs_len: Number of time-steps in input trajectories
		- pred_len: Number of time-steps in output trajectories
		- skip: Number of frames to skip while making the dataset
		- threshold: Minimum error to be considered for non linear traj
		when using a linear predictor
		- min_ped: Minimum number of pedestrians that should be in a seqeunce
		- delim: Delimiter in the dataset files
		
		- runway ends are (0,0) and (1.45,0)
		"""
		super(TrajectoryDataset, self).__init__()
		self.max_peds_in_frame = 0

		self.data_dir = data_dir
		self.obs_len = obs_len
		self.skip = skip
		self.step = step
		self.pred_len = pred_len
		goal_data = []
		self.skip = skip
		self.step = step
		self.seq_len = self.obs_len + self.pred_len
		self.delim = delim
		self.seq_final_len = self.obs_len + int(math.ceil(self.pred_len/self.step))
		all_files = os.listdir(self.data_dir)
		all_files = [os.path.join(self.data_dir, _path) for _path in all_files]
		seq_list = []
		seq_list_rel = []
		num_peds_in_seq = []
		context_list =[]
		goal_data_list=[]
		for path in all_files:
			counting_goals = 0
			data = read_file(path, delim)
			if (len(data[:,0])==0):
				print("File is empty")
				continue
			frames = np.unique(data[:, 0]).tolist()
			frame_data = []
			for frame in frames:
				frame_data.append(data[frame == data[:, 0], :])
			df = pd.read_csv(path, sep = ' ', names = ["Frame","ID","x","y","z","w_x","w_y"])
			counts = df["ID"].value_counts()
			vals= counts.to_numpy()

			ownship_id2 = counts.first_valid_index()
#             print("df",len(df))
			#         print("ownship_id",ownship_id,ownship_id2,counts.iloc[vals[0]],vals[1],counts)
#             print(counts.index[0],counts.index[1],counts.index[2])
			sizes=np.sum(vals)
			start = 0
			data_dict = np.zeros((sizes,7))
			for i in range(len(vals)):
				ownship_id = counts.index[i]
				first_frame = df.loc[df["ID"]==ownship_id]["Frame"].iloc[0]
				last_frame = df.loc[df["ID"]==ownship_id]["Frame"].iloc[-1]


				df_pruned = df.loc[(df["Frame"]>=first_frame) & (df["Frame"]<=last_frame) & (df["ID"]==ownship_id)]

				df_based = df_pruned.copy()

				data_dict1 = df_based.to_numpy()

				data_dict[start:start+vals[i]]=data_dict1
				start = start+vals[i]
			num_sequences = int(math.ceil((len(frames) - self.seq_len + 1) / skip))

			for idx in range(0, num_sequences * self.skip + 1, skip):

				curr_seq_data = np.concatenate(
					frame_data[idx:idx + self.seq_len], axis=0)

				peds_in_curr_seq = np.unique(curr_seq_data[:, 1])

				self.max_peds_in_frame = max(self.max_peds_in_frame,len(peds_in_curr_seq))

				curr_seq_rel = np.zeros((len(peds_in_curr_seq), 2,
										 self.seq_final_len))

				curr_seq = np.zeros((len(peds_in_curr_seq), 3,self.seq_final_len ))
				curr_context =  np.zeros((len(peds_in_curr_seq), 2,self.seq_final_len ))
				goal_seq = np.zeros((len(peds_in_curr_seq), 3))

				num_peds_considered = 0
				all_data = np.concatenate( frame_data, axis=0)

				num_peds=0
				flag =[]
				agent = -1
				for _, ped_id in enumerate(peds_in_curr_seq):
					agent +=1

					curr_ped_seq = curr_seq_data[curr_seq_data[:, 1] ==
												 ped_id, :]
					if curr_seq_data[:,4].min() <0.1:

						continue

					goal_data= all_data[all_data[:,1]==
												 ped_id,:]
					origin_goal_list=(np.where(((goal_data[:,2:4] < [0.1, 0.05]).all(axis=1))  & ((goal_data[:,2:4] > [-0.1, -0.05]).all(axis=1))))
					
					origin_goal_list =origin_goal_list[0]
					flag_goal=2
					if origin_goal_list.size!=0 and origin_goal_list[0]>100 :
						flag_goal = 0
						goal_data = goal_data[origin_goal_list[0]]
					if origin_goal_list.size==0:
						end_goal_list=(np.where(((goal_data[:,2:4] < [1.7, 0.05]).all(axis=1))  & ((goal_data[:,2:4] > [1.3, -0.05]).all(axis=1))))
						end_goal_list =end_goal_list[0]

						if end_goal_list.size!=0 and end_goal_list[0]>100 :

							flag_goal = -1
							goal_data = goal_data[end_goal_list[0]]
					if  flag_goal!=0 and flag_goal!=-1:
						if int(goal_data[:,0].size) <80:
							goal_data = goal_data[-2]
						else:
							goal_data = goal_data[-40]

					pad_front = frames.index(curr_ped_seq[0, 0]) - idx
					pad_end = frames.index(curr_ped_seq[-1, 0]) - idx + 1

					if pad_end - pad_front != self.seq_len:
						continue

					curr_ped_seq = np.transpose(curr_ped_seq[:, 2:])


					obs = curr_ped_seq[:,:obs_len]
					t =np.arange(0, 20)
					if len(obs[0]) ==20:
						for i in range(3):

							spl = interpolate.splrep(t,obs[i], s=25,k=2)
							obs[i] = interpolate.splev(t, spl)

					pred = curr_ped_seq[:,obs_len::step]
					if len(pred[0]) ==20:
						for i in range(3):
							spl = interpolate.splrep(t,pred[i], s=25,k=2)
							pred[i] = interpolate.splev(t, spl)
					curr_ped_seq = np.hstack((obs,pred))

					context = curr_ped_seq[-2:,:]
				
#                     assert(~np.isnan(context).any())
					# Make coordinates relative
					rel_curr_ped_seq = np.zeros(curr_ped_seq.shape)
					rel_curr_ped_seq[:, 1:] = \
						curr_ped_seq[:, 1:] - curr_ped_seq[:, :-1]

					num_peds+=1

					if (curr_ped_seq.shape[1]!=self.seq_final_len):	
						continue
					if np.linalg.norm((obs[0:3,0]-obs[0:3,-1]), 2)<0.2 or np.linalg.norm((pred[0:3,0]-pred[0:3,-1]), 2)<0.2:
						flag.append(ped_id)
						num_peds-=1
						loc=int(np.where(peds_in_curr_seq == flag[0])[0])

						curr_seq_rel = np.delete(curr_seq_rel, loc, 0)
						curr_seq =np.delete(curr_seq, loc, 0)
						curr_context =  np.delete(curr_context, loc, 0)
						goal_seq = np.delete(goal_seq, loc, 0)

						continue

					else:

						_idx = num_peds_considered

						context[0,-(num_peds+pred_len)] = ped_id
						context[1,-(num_peds+pred_len)] = ownship_id2
						curr_seq[_idx, :, pad_front:pad_end] = curr_ped_seq[:3,:]
						curr_seq_rel[_idx, :, pad_front:pad_end] = rel_curr_ped_seq[:2,:]
						curr_context[_idx,:,pad_front:pad_end] = context
						goal_seq[_idx,0:3] =goal_data[2:5]
						num_peds_considered += 1
						break

				all_zeros = not np.any(curr_seq[:])
				
				if num_peds_considered > min_ped:
#                     
					
					
					num_peds_in_seq.append(num_peds_considered)
					seq_list.append(curr_seq[:num_peds_considered])
					seq_list_rel.append(curr_seq_rel[:num_peds_considered])
					context_list.append(curr_context[:num_peds_considered])
					goal_data_list.append(goal_seq[:num_peds_considered])
		self.num_seq = len(seq_list)

		seq_list = np.concatenate(seq_list, axis=0)
		goal_data_list = np.concatenate(goal_data_list, axis=0)

		seq_list_rel = np.concatenate(seq_list_rel, axis=0)
		context_list = np.concatenate(context_list, axis=0)
		# Convert numpy -> Torch Tensor
		self.obs_traj = torch.from_numpy(
			seq_list[:, :, :self.obs_len]).type(torch.float)
		self.obs_context = torch.from_numpy(	
			context_list[:,:,:self.obs_len]).type(torch.float)
		self.pred_traj = torch.from_numpy(
			seq_list[:, :, self.obs_len:]).type(torch.float)
		self.obs_traj_rel = torch.from_numpy(
			seq_list_rel[:, :, :self.obs_len]).type(torch.float)
		self.pred_traj_rel = torch.from_numpy(
			seq_list_rel[:, :, self.obs_len:]).type(torch.float)
		self.goal_position = torch.from_numpy(goal_data_list).type(torch.float)
		cum_start_idx = [0] + np.cumsum(num_peds_in_seq).tolist()
		self.seq_start_end = [
			(start, end)
			for start, end in zip(cum_start_idx, cum_start_idx[1:])
		]

		
		
		self.max_agents = -float('Inf')
		for (start, end) in self.seq_start_end:
			n_agents = end - start
			self.max_agents = n_agents if n_agents > self.max_agents else self.max_agents
			
		self.translated_dataset = np.zeros([self.num_seq,3,20])
		start, end = self.seq_start_end[i]
		self.obs_traj,self.pred_traj,self.goal_position =self.obs_traj*1000.0,self.pred_traj*1000.0,self.goal_position*1000.0
		pred_traj = self.pred_traj.cpu().numpy()
		self.translated_dataset=pred_traj[:] - pred_traj[:,:,0:1] 

		start_points = self.translated_dataset[:,:,3]
		norm=np.linalg.norm(start_points[:,:],axis=1)
		unit_vector_1 = start_points[:,:] / norm.reshape((self.num_seq,1))
		angle1 = np.arctan2(unit_vector_1[:,1],unit_vector_1[:,0])
		r = R.from_euler('z',-angle1)
		direction_matrix_rep = np.zeros([self.num_seq,3,3])
		self.rotated_dataset = np.zeros(self.translated_dataset.shape)
		direction_matrix_rep = r.as_matrix()
		self.rotated_dataset[:,0:2,:] = np.matmul(direction_matrix_rep[:,0:2,0:2],self.translated_dataset[:,0:2,:])
		self.rotated_dataset[:,2,:]=self.translated_dataset[:,2,:]

		self.height_rates = (pred_traj[:,2,19]-pred_traj[:,2,0])*3.28084*3 # unit conversion
		self.height_grouped =np.zeros(self.height_rates.shape)
		for agent in range(self.num_seq):
			if -20 <self.height_rates[agent] < 20.0:
				self.height_grouped[agent] = 0.0
			elif 20 <self.height_rates[agent] < 375.0:
				self.height_grouped[agent] = 250.0
			elif 374 <self.height_rates[agent] < 750:
				self.height_grouped[agent] = 500.0
			elif 749 <self.height_rates[agent] < 3450.0:
				self.height_grouped[agent] = 1000.0
			elif -375 <self.height_rates[agent] < -20.0:
				self.height_grouped[agent] = -250.0
			elif -750 <self.height_rates[agent] < -374:
				self.height_grouped[agent] = -500.0
			elif -1801 <self.height_rates[agent] < -749:
				self.height_grouped[agent] = -1000.0

		self.full_l2 = np.ones([self.num_seq,1])*np.inf

		self.full_l2_intermed_result = np.ones([self.num_seq,traj_lib.shape[0]])*np.inf

		self.full_l2_minval = np.ones([self.num_seq,1])*np.inf

		weight = 2.0
		low_weight = 1.0

		for agent in range(self.num_seq):
			for lib in range(traj_lib.shape[0]):
				if self.height_grouped[agent] == index_lib[lib,3]:

					dataset_endpts = weight*np.reshape(np.hstack((self.rotated_dataset[agent,0,16:20],self.rotated_dataset[agent,1,16:20])), (8,))
					lib_endpts = weight*np.reshape(np.hstack((traj_lib[lib,0,16:20],traj_lib[lib,1,16:20])),(8,))

					dataset_ = np.reshape(np.hstack((low_weight*self.rotated_dataset[agent,0],low_weight*self.rotated_dataset[agent,1],dataset_endpts)), (1,48))
					lib_ = np.reshape(np.hstack((low_weight*traj_lib[lib,0],low_weight*traj_lib[lib,1],lib_endpts)),(1,48))

					self.full_l2_intermed_result[agent,lib] = all_pairs_euclid_numpy(dataset_, lib_)

		for j in range(self.num_seq):
			self.full_l2_minval[j,0],self.full_l2[j,0]= torch.min(torch.from_numpy(self.full_l2_intermed_result[j,:] ).type(torch.float), dim=-1, keepdim=True)
		print(type(self.full_l2))

	def __len__(self):
		return self.num_seq
	
	
	def __max_agents__(self):
		return self.max_agents


	def __getitem__(self, index):
		start, end = self.seq_start_end[index]

		out = [
			self.obs_traj[start:end, :],self.pred_traj[start:end, :],
			self.obs_traj_rel[start:end, :], self.pred_traj_rel[start:end, :], self.obs_context[start:end, :],
			self.goal_position[start:end, :],self.full_l2[index]
		]

		return out

def get_hash():
	pass


def direction_detect(input_pos, goal):
	dir_array = torch.zeros([input_pos.shape[0],10])
	## [N, NE, E, SE, S, SW, W, NW, Takeoff, Landing]
	difference = goal-input_pos
#     print("diff",difference,'goal',goal, "input_pos",input_pos)
	for b in range(input_pos.shape[0]):
		planar_slope = torch.atan2(difference[b,1],difference[b,0])
		degrees_slope = planar_slope*180.0/np.pi

		if goal[b,0]<500 and goal[b,0]> -10 and abs(goal[b,1])<10: #1
			dir_array[b,9] = 1.0

		elif goal[b,0]<1500 and goal[b,0]> 1000 and abs(goal[b,1])<10:  #2,
			dir_array[b,8] = 1.0

		elif degrees_slope <22.5 and degrees_slope >-22.5: #east
			dir_array[b,2] = 1.0
		elif degrees_slope <67.5 and degrees_slope >22.5: #NE
			dir_array[b,1] = 1.0
		elif degrees_slope <112.5 and degrees_slope >67.5: #N
			dir_array[b,0] = 1.0
		elif degrees_slope <157.5 and degrees_slope >112.5: # NW
			dir_array[b,7] = 1.0
		elif degrees_slope <-157.5 or degrees_slope >157.5: # W
			dir_array[b,6] = 1.0
		elif degrees_slope <-22.5 and degrees_slope >67.5: #SE
			dir_array[b,3] = 1.0
		elif degrees_slope <-67.5 and degrees_slope >-112.5: #S
			dir_array[b,4] = 1.0
		elif degrees_slope <-112.5 and degrees_slope >-157.5: #SW:
			dir_array[b,5] = 1.0
	return dir_array



def seq_collate_old(data):
	(obs_seq_list, pred_seq_list, obs_seq_rel_list, pred_seq_rel_list,context_list) = zip(*data)

	_len = [len(seq) for seq in obs_seq_list]
	cum_start_idx = [0] + np.cumsum(_len).tolist()
	seq_start_end = [[start, end]
					 for start, end in zip(cum_start_idx, cum_start_idx[1:])]

	# Data format: batch, input_size, seq_len
	# LSTM input format: seq_len, batch, input_size
	obs_traj = torch.cat(obs_seq_list, dim=0).permute(2, 0, 1)
	pred_traj = torch.cat(pred_seq_list, dim=0).permute(2, 0, 1)
	obs_traj_rel = torch.cat(obs_seq_rel_list, dim=0).permute(2, 0, 1)
	pred_traj_rel = torch.cat(pred_seq_rel_list, dim=0).permute(2, 0, 1)
	context = torch.cat(context_list, dim=0 ).permute(2,0,1)
	seq_start_end = torch.LongTensor(seq_start_end)

	out = [
		obs_traj, pred_traj, obs_traj_rel, pred_traj_rel, context, seq_start_end
	]
	return tuple(out)

def seq_collate(data):
	(obs_seq_list, pred_seq_list, obs_seq_rel_list, pred_seq_rel_list,context_list,goal_seq_list,full_l2) = zip(*data)

	_len = [len(seq) for seq in obs_seq_list]
	cum_start_idx = [0] + np.cumsum(_len).tolist()
	seq_start_end = [[start, end]
					 for start, end in zip(cum_start_idx, cum_start_idx[1:])]

	# Data format: batch, input_size, seq_len
	# LSTM input format: seq_len, batch, input_size
	obs_traj = torch.cat(obs_seq_list, dim=0).permute(2, 0, 1)
	pred_traj = torch.cat(pred_seq_list, dim=0).permute(2, 0, 1)
	obs_traj_rel = torch.cat(obs_seq_rel_list, dim=0).permute(2, 0, 1)
	pred_traj_rel = torch.cat(pred_seq_rel_list, dim=0).permute(2, 0, 1)
	context = torch.cat(context_list, dim=0 ).permute(2,0,1)
	seq_start_end = torch.LongTensor(seq_start_end)
	goal_position = torch.cat(goal_seq_list, dim=0)
	full_l2=torch.from_numpy(np.array(full_l2))
	out = [
		obs_traj, pred_traj, obs_traj_rel, pred_traj_rel, context, seq_start_end, goal_position,full_l2
	]
	return tuple(out)


def read_file(_path, delim='\t'):
	data = []
	if delim == 'tab':
		delim = '\t'
	elif delim == 'space':
		delim = ' '
	with open(_path, 'r') as f:
		for line in f:
			line = line.strip().split(delim)
			line = [float(i) for i in line]
			data.append(line)
	return np.asarray(data)



class Chomp1d(nn.Module):
	def __init__(self, chomp_size):
		super(Chomp1d, self).__init__()
		self.chomp_size = chomp_size

	def forward(self, x):
		return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
	def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
		super(TemporalBlock, self).__init__()
		self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
										   stride=stride, padding=padding, dilation=dilation))
		self.chomp1 = Chomp1d(padding)
		self.bn1 = nn.BatchNorm1d(num_features=n_outputs)
		self.relu1 = nn.PReLU(num_parameters=n_outputs)
		self.dropout1 = nn.Dropout(dropout)

		self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
										   stride=stride, padding=padding, dilation=dilation))
		self.chomp2 = Chomp1d(padding)
		self.bn2 = nn.BatchNorm1d(num_features=n_outputs)
		self.relu2 = nn.PReLU(num_parameters=n_outputs)
		self.layer2= nn.LayerNorm([n_outputs,20])
		self.dropout2 = nn.Dropout(dropout)

		self.net = nn.Sequential(self.conv1, self.chomp1, self.bn1,self.relu1, self.dropout1,
								 self.conv2, self.chomp2, self.bn2,self.relu2, self.dropout2)
		self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
		self.relu = nn.PReLU(num_parameters=n_outputs)
#         self.tanh = nn.Tanh()
		self.init_weights()

	def init_weights(self):
		self.conv1.weight.data.normal_(0, 0.1)
		self.conv2.weight.data.normal_(0, 0.1)
		if self.downsample is not None:
			self.downsample.weight.data.normal_(0, 0.01)

	def forward(self, x):
		out = self.net(x)
		res = x if self.downsample is None else self.downsample(x)
		return self.relu(out + res)


class TemporalConvNet(nn.Module):
	def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
		super(TemporalConvNet, self).__init__()
		layers = []
		num_levels = len(num_channels)
#         print("num_levels",num_levels)
		for i in range(num_levels):
			dilation_size = 2 ** i
			in_channels = num_inputs if i == 0 else num_channels[i-1]
#             print("in_channels",in_channels)
			out_channels = num_channels[i]
			layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
									 padding=(kernel_size-1) * dilation_size, dropout=dropout)]

		self.network = nn.Sequential(*layers)

	def forward(self, x):
		return self.network(x)


def populate_traj_lib():
	main()
	
	# note position of motion prim library text files
	lib_path = os.getcwd() + '/traj_lib_0SI.txt'
	index_path = os.getcwd() + '/traj_index_0SI.txt'

	## obtain a 3d matrix of each trajectory's (x, y, z) positions into an array
	file1 = open(lib_path, 'r',newline='\r\n')
	traj_no = 252 # note the number
	count, j = 0,0
	traj_lib = np.zeros([traj_no,3,20])

	for line in open(lib_path, 'r',newline='\n'):
		if count%4 == 1:
		traj_lib[j,0] = np.fromstring( line.strip(), dtype=float, sep=' ' )
		if count%4 == 2:
		traj_lib[j,1] = np.fromstring( line.strip(), dtype=float, sep=' ' )
		if count%4 == 3:
		traj_lib[j,2] = np.fromstring( line.strip(), dtype=float, sep=' ' )
		j+=1
		count += 1

	## obtain the details of each trajectory from the index
	file1 = open(index_path, 'r',newline='\r\n')

	j = 0
	index_lib = np.zeros([traj_no,6])

	for line in open(index_path, 'r',newline='\n'):
		index_lib[j,:] = np.fromstring( line.strip(), dtype=float, sep=' ' )
		j+=1

	return traj_lib,index_lib


