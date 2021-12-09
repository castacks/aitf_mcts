import torch.nn.functional as F
from model.tcn import TemporalConvNet
import torch
from torch import nn


class Policy(nn.Module):
    def __init__(self, args, device='cpu'):
        super(Policy, self).__init__()

        input_size = args.input_size
        num_channels = args.channel_size
        kernel_size = args.kernel_size
        dropout = args.dropout
        self.device = device
        self.tcn_encoder_x = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)

        self.linear_decoder_x = nn.Linear(256, 256)
        self.linear_x = nn.Linear(256, 252)
        self.linear_decoder_v = nn.Linear(256, 64)
        self.linear_v = nn.Linear(64, 64)
        self.output_v = nn.Linear(64, 1)
        self.goal_expand = nn.Linear(10, 128)

        self.context_conv = nn.Conv1d(in_channels=2, out_channels=1, kernel_size=3, padding=1)
        self.context_linear = nn.Linear(15, 7)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

        self.output_relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.linear_x.weight.data.normal_(0, 0.05)
        self.linear_decoder_x.weight.data.normal_(0, 0.05)
        self.goal_expand.weight.data.normal_(0, 0.05)

        self.linear_decoder_v.weight.data.normal_(0, 0.05)
        self.linear_v.weight.data.normal_(0, 0.05)
        self.output_v.weight.data.normal_(0, 0.05)

        self.context_linear.weight.data.normal_(0, 0.05)
        self.context_conv.weight.data.normal_(0, 0.1)

    def forward(self, x1, goal):
        x1 = torch.unsqueeze(x1, 2)
        x1 = torch.transpose(x1, 0, 2)

        # goal_vector = goal.to(self.device)
        goal_expanded = self.goal_expand(goal)

        encoded_x = self.tcn_encoder_x(x1)

        encoded_x = encoded_x[:, :, -1]
        H_x = torch.cat((encoded_x, goal_expanded), 1)

        decoded1_x = self.relu((self.linear_decoder_x(H_x)))
        decoded2_x = self.linear_x(decoded1_x)

        decoded1_v = self.relu((self.linear_decoder_v(H_x)))
        decoded2_v = self.relu((self.linear_v(decoded1_v)))
        # multi_head = self.multi_head_layer(H_x)

        # output = torch.squeeze(decoded2,dim=0)

        softmax_out = F.log_softmax(decoded2_x, dim=1)
        # print(softmax_out.shape)

        v = self.tanh(self.output_v(decoded2_v))
        # print(v)
        return softmax_out[0], v

