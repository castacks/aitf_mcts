from numpy.lib.function_base import disp
from torch.utils.data.dataset import Dataset
from torch.utils.data import WeightedRandomSampler
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from base_net import Policy
import numpy as np
import torch
import torch.optim as optim

class Net():

    def __init__(self,args):
        
        self.args = args
        self.nnet = Policy(args)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.nnet.to(self.device)

        if args.model_weights is not None:
            modelpath = os.getcwd() + args.models_folder + args.model_weights
            self.load_hot_start(modelpath)

    def load_hot_start(self, modelpath):

        checkpoint = torch.load(modelpath, map_location=torch.device('cpu'))
        miss, unex = self.nnet.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print("miss", miss, "unex", unex)
        print(" Pre-trainined model weights loaded from ", modelpath)

    def train(self,examples):
        print("Device is ",self.device)
        # self.nnet.to(self.device)
        self.nnet.train()
        optimizer = optim.AdamW(self.nnet.parameters(), lr =0.0001)
        train_data = ReplayDataLoader(examples)
        target = np.hstack([x[3] for x in examples])


        print('target train -1/1: {}/{}'.format(len(np.where(target == -1)[0]), len(np.where(target == 1)[0])))
        class_sample_count = np.array([len(np.where(target == t)[0]) for t in np.unique(target)])
        weight = 1. / class_sample_count
        samples_weight = np.array([weight[t] for t in target])

        samples_weight = torch.from_numpy(samples_weight)
        samples_weigth = samples_weight.double()
        sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
        train_dataloader = DataLoader(train_data, batch_size=64, shuffle=False, sampler=sampler)

        for epoch in range(self.args.epochs):
            loss_pi = 0
            loss_v = 0

            for batch in tqdm(train_dataloader):
                optimizer.zero_grad()

                batch = [tensor.to(self.device) for tensor in batch]
                position,goal, target_pis, target_vs = batch
                total_loss = 0
                for i in range(position.shape[0]):
                    out_pi, out_v = self.nnet(position[i],goal[i])
                    l_pi = self.loss_pi(target_pis, out_pi)
                    l_v = self.loss_v(target_vs, out_v)
                    total_loss += l_pi + l_v
                    loss_v += l_v.item()  
                    loss_pi += l_pi.item()
                    assert(not np.isnan(loss_v))
                    assert(not np.isnan(loss_pi))

                total_loss.backward()
                optimizer.step()
            print("Epoch #",epoch,"Loss_pi", loss_pi, "Loss_v", loss_v)
        
    def predict(self,curr_position,goal_position):

        # self.nnet.eval()

        # self.nnet.to('cpu')
        pi, v = self.nnet.forward(curr_position, goal_position)
        return torch.exp(pi), v
    
    
    def loss_pi(self, targets, outputs):
        return -torch.sum(targets * (outputs)) / targets.size()[0]

    def loss_v(self, targets, outputs):
        return torch.sum((targets - outputs.view(-1)) ** 2) / targets.size()[0]


class ReplayDataLoader(Dataset):
    def __init__(self, examples):
        self.examples = examples

    def balance_data(self):

        pass
    
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        
        position = self.examples[index][0]
        goal = self.examples[index][1]
        pi = self.examples[index][2]
        v = self.examples[index][3]

        return position,goal, pi, v  
