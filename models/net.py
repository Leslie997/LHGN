import torch
import torch.nn as nn

import torch.nn.functional as F

from .layers import GraphAttentionLayer
#from torch_geometric.nn.conv import ClusterGCNConv
from sklearn.neighbors import NearestNeighbors

import numpy as np

import random


class LHGNets(nn.Module):
    def __init__(self, view_num, sample_size, view_dim, cls_num, dropout_rate, nheads, gpu, latent_dim=128):
        super(LHGNets, self).__init__()
        self.view_num = view_num
        self.sample_size = sample_size
        self.view_dim = view_dim
        self.latent_dim = latent_dim
        self.cls_num = cls_num
        self.dropout_rate = dropout_rate
        self.nheads = nheads
        self.gpu = gpu
        # latent representation
        self.latent = nn.Parameter(torch.FloatTensor(sample_size, latent_dim))
        nn.init.xavier_uniform_(self.latent.data, gain=1.414)

        # reconstruction encoder
        self.encoder = self.build_encoder()
        self.classifiar = self.build_classifiar()

        # meta path attention
        self.meta_att = self.build_meta_att()

        # semantic agg
        self.semantic_agg = self.build_semantic_agg()
        

    def build_encoder(self):
        encoder_lst = nn.ModuleList()
        for v in range(self.view_num):
            encoder_lst.append(
                nn.Sequential(
                    nn.Linear(self.latent_dim, self.view_dim[v]),
                    nn.Dropout(self.dropout_rate)
                )
            )
        return encoder_lst

    def build_meta_att(self):
        meta_att_lst = nn.ModuleList()
        for v in range(self.view_num):
            meta_att_lst.append(
                MetaAtt(self.latent_dim, self.latent_dim, self.dropout_rate, self.nheads)
            )
        return meta_att_lst

    def build_classifiar(self):
        classifiar = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim),
            # nn.Linear(self.latent_dim * self.view_num * self.nheads, self.cls_num),
            nn.ELU(),
        )
        return classifiar


    def build_semantic_agg(self):
        agg = nn.Sequential(
            nn.Linear(self.latent_dim * self.view_num * self.nheads, self.latent_dim)
        )
        return agg


    # def get_local_mask(self, length, rate):
    #     local_mask = torch.cuda.FloatTensor(length, length).uniform_() > rate
    #     return local_mask

    def get_view_adj(self, meta_path, nn_graph):
        adj = torch.matmul(meta_path.unsqueeze(1), meta_path.unsqueeze(0))
        adj = torch.mul(adj, nn_graph)
        # print(torch.sum(adj, dim=1))
        # print(adj)
        # print(torch.sum(adj, dim=1))
        if self.gpu != '-1':
            adj = adj + torch.eye(adj.shape[0]).cuda()
        else:
            adj = adj + torch.eye(adj.shape[0])
        return adj

    def get_nn_graph(self, x):
        x = x.cpu().detach().numpy()
        # todo
        # nbrs = NearestNeighbors(n_neighbors=10, algorithm='ball_tree').fit(x)
        nbrs = NearestNeighbors(n_neighbors=4, algorithm='auto').fit(x)
        nn_graph = nbrs.kneighbors_graph(x).toarray()
        if self.gpu != '-1':
            nn_graph = torch.Tensor(nn_graph).cuda()
        else:
            nn_graph = torch.Tensor(nn_graph)
        return nn_graph

    def forward(self, feature_mask):
        rec_vec = []
        meta_att_vec = []
        for v in range(self.view_num):
            # reconstruction
            rec_vec.append(
                self.encoder[v](self.latent)
            )
            nn_graph = self.get_nn_graph(self.latent)

            # meta path attention
            meta_path = feature_mask[:, v]
            meta_adj = self.get_view_adj(meta_path, nn_graph)
            meta_att_output = self.meta_att[v](self.latent, meta_adj)
            meta_att_vec.append(meta_att_output)        

        # semantic attention
        semantic = torch.cat(meta_att_vec, dim=1)
        semantic = self.semantic_agg(semantic)

        # classifiar
        output = self.classifiar(semantic)
        output = F.log_softmax(output, dim=1)
        return rec_vec, output,semantic


class MetaAtt(nn.Module):
    def __init__(self, input_size, output_size, dropout_rate, nheads=4):
        super(MetaAtt, self).__init__()
        self.dropout_rate = dropout_rate
        self.attentions = [GraphAttentionLayer(input_size, output_size, dropout_rate=dropout_rate, concat=False) for _ in range(nheads)]
        # self.attentions = [ClusterGCNConv(input_size, output_size) for _ in range(nheads)]

        for i, attention in enumerate(self.attentions):
            self.add_module('meta_path_attention_{}'.format(i), attention)

    def forward(self, x, adj):
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout_rate, training=self.training)
        return x
