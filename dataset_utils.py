import torch
import numpy as np



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