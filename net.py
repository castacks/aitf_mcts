import torch.nn.functional as F
from tcn import TemporalConvNet
import torch
from torch import nn
from dataset_utils import direction_detect

class Policy(nn.Module):
	def __init__(self, args):
		super(Policy, self).__init__()

		
		input_size = args.input_size
		num_channels = args.num_channels*[args.channel_size]
		kernel_size = args.kernal_size
		dropout = args.dropout
		
		self.tcn_encoder_x = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)

		self.multi_head_layer = nn.Linear(259,256)

		self.linear_decoder_x = nn.Linear(259,256)
		self.linear_x = nn.Linear(256,252)
		self.goal_expand = nn.Linear(10,128)

		self.context_conv = nn.Conv1d(in_channels=2, out_channels=1, kernel_size=3,padding=1)
		self.context_linear = nn.Linear(15,7)
		self.relu = nn.ReLU()
		self.output_relu = nn.ReLU()
		self.init_weights()

	def init_weights(self):

		self.linear_decoder_x.weight.data.normal_(0, 0.05)
		self.context_linear.weight.data.normal_(0, 0.05)
		self.context_conv.weight.data.normal_(0, 0.1)


	def forward(self, x, y, goal_position,context,sort=False):       

		output = torch.zeros([x.shape[2],252]).to(device)
		softmax_out =torch.zeros([x.shape[2],252]).to(device)
		latent_space_ =torch.zeros([x.shape[2],259]).to(device)
		
		encoded_appended_trajectories_x = []
		x1 = torch.transpose(x, 0, 2)

		if x.shape[2] <batch_size:
			input_pos =torch.reshape(x1[:,:,0], (x.shape[2],3))
		input_pos =torch.reshape(x1[:,:,0], (x.shape[2],3))

		if goal_position[0,0].cpu()<800 and goal_position[0,0].cpu()> -10 and abs(goal_position[0,1].cpu())<300: #1
			goal_position[0,0],goal_position[0,1] =0.0, 0.0

		elif goal_position[0,0].cpu()<1500 and goal_position[0,0].cpu()> 1000 and abs(goal_position[0,1].cpu())<300:  #2,
			goal_position[0,0],goal_position[0,1] =1450.0, 0.0

		goal_vector = direction_detect(input_pos.cpu(), goal_position.cpu())

		goal_vector= goal_vector.to(device)
		goal_expanded =self.goal_expand(goal_vector)

		x1=x1/1000.0
		encoded_x = self.tcn_encoder_x(x1)

		encoded_x =encoded_x[:,:,-1]
		full_encoded_x = torch.cat((encoded_x,goal_expanded,goal_position/1000.0),1)
		encoded_appended_trajectories_x.append(full_encoded_x)

		final_input = torch.squeeze(torch.stack(encoded_appended_trajectories_x))


		H_x = final_input
		decoded1 = (self.linear_decoder_x(H_x))
		decoded2 = (self.linear_x(decoded1))

		multi_head = self.multi_head_layer(H_x)

		output = torch.squeeze(decoded2,dim=0)

		softmax_out = F.softmax(decoded2, dim = 0)

		return softmax_out, output,latent_space_, multi_head    
