"""
# -*- coding: utf-8 -*-
# @Author : Sun JJ
# @File : GCN_Model.py
# @Time : 2022/5/9 9:16
# code is far away from bugs with the god animal protecting
#         ┌─┐       ┌─┐
#      ┌──┘ ┴───────┘ ┴──┐
#      │                 │
#      │       ───       │
#      │  ─┬┘       └┬─  │
#      │                 │
#      │       ─┴─       │
#      │                 │
#      └───┐         ┌───┘
#          │         │
#          │         │
#          │         │
#          │         └──────────────┐
#          │                        │
#          │                        ├─┐
#          │                        ┌─┘
#          │                        │
#          └─┐  ┐  ┌───────┬──┐  ┌──┘
#            │ ─┤ ─┤       │ ─┤ ─┤
#            └──┴──┘       └──┴──┘
"""


import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.utils.data import Dataset,DataLoader


SEED = 2020
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.set_device(0)
    torch.cuda.manual_seed(SEED)


LAYER = 10
ALPHA = 0.5
LAMBDA = 1.3
VARIANT = True
DROPOUT = 0.1
INPUT_DIM = 67
HIDDEN_DIM = 256

BATCH_SIZE = 1
NUM_CLASSES = 2
NUMBER_EPOCHS = 50
WEIGHT_DECAY = 0
LEARNING_RATE = 1E-3

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def normalize(mx):

    rowsum = np.array(mx.sum(1))
    r_inv = (rowsum ** -0.5).flatten()
    r_inv[np.isinf(r_inv)] = 0
    r_mat_inv = np.diag(r_inv)
    result = np.dot(np.dot(r_mat_inv,mx),r_mat_inv)

    return result

def load_graph(sequence_name):

    fpath = './data/adjacency_matrix/discrete/' + sequence_name + '.npy'
    adjacency_matrix = np.load(fpath)
    norm_matrix = normalize(adjacency_matrix.astype(np.float32))

    return norm_matrix

def get_node_features(sequence_name):

    # fpath = './data/node_features/fusion2/' + sequence_name + '.npy'
    # node_features = np.load(fpath)

    pssm_path = './data/node_features/pssm/' + sequence_name + '.npy'
    blosum_path = './data/node_features/blosum/' + sequence_name + '.npy'
    aaphy_path = './data/node_features/AAPHY/' + sequence_name + '.npy'
    psp_path = './data/node_features/psp/' + sequence_name + '.npy'

    pssm = np.load(pssm_path)
    blosum = np.load(blosum_path)
    aaphy = np.load(aaphy_path)
    psp = np.load(psp_path)

    node_features = np.concatenate([pssm, blosum, psp, aaphy], axis=1)

    return node_features

class ProDataset(Dataset):

    def __init__(self,dataframe):

        self.names = dataframe['ID'].values
        self.sequences = dataframe['sequence'].values
        self.labels = dataframe['label'].values

    def __getitem__(self,index):

        sequence_name = self.names[index]
        sequence = self.sequences[index]
        label = np.array(self.labels[index])

        node_features = get_node_features(sequence_name)

        graph = load_graph(sequence_name)

        return sequence_name,sequence,label,node_features,graph

    def __len__(self):

        return len(self.labels)

class GraphConvolution(nn.Module):

    def __init__(self,in_features,out_features,residual = False,variant = False):

        super(GraphConvolution,self).__init__()
        self.variant = variant
        if self.variant:
            self.in_features = 2 * in_features
        else:
            self.in_features = in_features

        self.out_features = out_features
        self.residual = residual
        self.weight = Parameter(torch.FloatTensor(self.in_features,self.out_features))
        self.reser_parameters()

    def reset_parameters(self):

        stdv = 1. / math.sqrt(self.out_features)
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self,input,adj,h0,lamda,alpha,l):

        theta = min(1,math.log(lamda / l + 1))
        hi = torch.spmm(adj,input)

        if self.variant:
            support = torch.cat([hi,h0],1)
            r = (1 - alpha) * hi + alpha * h0
        else:
            support = (1 - alpha) * hi + alpha * h0
            r = support
        output = theta * torch.mm(support,self.weight) + (1- theta) * r

        if self.residual:
            output = output + input

        return output

class deepGCN(nn.Module):

    def __init__(self,nlayers,nfeat,nhidden,nclass,dropout,lamda,alpha,variant):

        super(deepGCN,self).__init__()
        self.convs = nn.ModuleList()

        for _ in range(nlayers):
            self.convs.append(GraphConvolution(nhidden,nhidden,variant = variant,residual = True))

        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(nfeat,nhidden))
        self.fcs.append(nn.Linear(nhidden,nclass))
        self.act_fn = nn.ReLU()
        self.dropout = dropout
        self.alpha = alpha
        self.lamda = lamda

    def forward(self,x,adj):

        _layers = []
        x = F.dropout(x,self.dropout,training = self.training)
        layer_inner = self.act_fn(self.fcs[0](x))
        _layers.append(layer_inner)

        for i,con in enumerate(self.convs):

            layer_inner = F.dropout(layer_inner,self.dropout,training = self.training)
            layer_inner = self.act_fn(con(layer_inner,adj,_layers[0],self.lamda,self.alpha,i + 1))

        layer_inner = F.dropout(layer_inner,self.dropout,training = self.training)
        layer_inner = self.fcs[-1](layer_inner)

        return layer_inner

class GraphPLBR(nn.Module):

    def __init__(self,nlayers,nfeat,nhidden,nclass,dropout,lamda,alpha,variant):

        super(GraphPLBR,self).__init__()

        self.deep_gcn = deepGCN(
            nlayers = nlayers,
            nfeat = nfeat,
            nhidden = nhidden,
            nclass = nclass,
            dropout = dropout,
            lamda = lamda,
            alpha = alpha,
            variant = variant
        )
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(),lr = LEARNING_RATE,weight_decay = WEIGHT_DECAY)

    def forward(self,x,adj):

        x = x.float()
        output = self.deep_gcn(x,adj)

        return output