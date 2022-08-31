"""Modified from https://github.com/alexmonti19/dagnet"""

import torch
import torch.nn as nn
from model.gat_layers import GraphAttentionLayer


class GAT(nn.Module):
    """Dense version of GAT."""
    def __init__(self, nin, nhid, nout, alpha, nheads):
        super(GAT, self).__init__()

        self.attentions = [GraphAttentionLayer(nin, nhid, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, nout, alpha=alpha, concat=False)
        self.bn1 = nn.BatchNorm1d(nout)

    def forward(self, x, adj):
        # print(self.attentions[0](x, adj).shape)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        # print("after attn",x)
        x = self.out_att(x, adj)
        # print("after norm",x)
        x = self.bn1(x)
        return torch.tanh(x)