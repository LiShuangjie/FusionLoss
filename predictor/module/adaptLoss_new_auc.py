from sklearn.neighbors import kneighbors_graph
from scipy import sparse
import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean
from utils import *
import numpy as np
from torch_geometric.nn import GCNConv
from tqdm import tqdm
from torch_scatter import scatter_add, scatter_mean
from torch_geometric.utils import add_remaining_self_loops

def kl_loss_compute(pred, soft_targets, reduce=True, tempature=1):
    pred = pred / tempature
    soft_targets = soft_targets / tempature
    kl = F.kl_div(F.log_softmax(pred, dim=1), F.softmax(soft_targets, dim=1), reduce=False)
    if reduce:
        return torch.mean(torch.sum(kl, dim=1))
    else:
        return torch.sum(kl, 1)

class Edge_Discriminator(nn.Module):
    def __init__(self,nnodes, nfeat, alpha,device, emb_dim=128, temperature=1.0, bias=0.0 + 0.0001):
        super(Edge_Discriminator, self).__init__()
        self.device = device
        self.temperature = temperature
        self.alpha = alpha
        self.bias = bias
        self.nnodes = nnodes
        # MLP(feat)
        self.feat_embedding_layers = nn.ModuleList()
        self.feat_embedding_layers.append(nn.Linear(nfeat, emb_dim))
        # MLP(struct)
        self.edge_mlp = nn.Linear(emb_dim * 2, 1)

    def get_feat_embedding(self, h):
        # 1 lagers
        for layer in self.feat_embedding_layers:
            h = layer(h)
            h = F.relu(h)
        return h
  
    def get_edge_weight(self, embeddings, edges):
        s1 = self.edge_mlp(torch.cat((embeddings[edges[0]], embeddings[edges[1]]), dim=1)).flatten()
        s2 = self.edge_mlp(torch.cat((embeddings[edges[1]], embeddings[edges[0]]), dim=1)).flatten()
        return (s1 + s2) / 2
    
    # gumbel reparameter
    def gumbel_sampling(self, edges_weights_raw):
        eps = (self.bias - (1 - self.bias)) * torch.rand(edges_weights_raw.size()) + (1 - self.bias)
        gate_inputs = torch.log(eps) - torch.log(1 - eps)
        gate_inputs = gate_inputs.to(self.device)
        gate_inputs = (gate_inputs + edges_weights_raw) / self.temperature
        return torch.sigmoid(gate_inputs).squeeze()

    def weight_to_adj_pyg(self, edge_index, weights_lp, weights_hp):
        # add self loop
        edge_index_lp, tmp_edge_weight = add_remaining_self_loops(
            edge_index, weights_lp, num_nodes=self.nnodes)
        weights_lp = tmp_edge_weight
        # normalize
        row, col = edge_index_lp[0], edge_index_lp[1]
        idx = col  
        deg = scatter_add(weights_lp, idx, dim=0, dim_size=self.nnodes)
        deg_inv_sqrt = (deg + 1e-6).pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)

        weights_lp = deg_inv_sqrt[row] * weights_lp * deg_inv_sqrt[col]
        # adj_lp = torch.sparse_coo_tensor(edge_index_lp, weights_lp, (self.nnodes, self.nnodes))

        # add self loop
        edge_index_hp, tmp_edge_weight = add_remaining_self_loops(
            edge_index, weights_hp, num_nodes=self.nnodes)
        weights_hp = tmp_edge_weight
        # normalize
        row, col = edge_index_hp[0], edge_index_hp[1]
        idx = col  
        deg = scatter_add(weights_hp, idx, dim=0, dim_size=self.nnodes)
        deg_inv_sqrt = (deg + 1e-6).pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)

        weights_hp = deg_inv_sqrt[row] * weights_hp * deg_inv_sqrt[col]
        weights_hp *= - self.alpha
        weights_hp[edge_index.shape[1]:] = 1
        # adj_hp = torch.sparse_coo_tensor(edge_index_hp, weights_hp, (self.nnodes, self.nnodes))
        return edge_index_lp, edge_index_hp, weights_lp, weights_hp
   
    def cal_localSim(self, edge_index, weights_lp, nnodes):
        src, tgt = edge_index
        localsim = scatter_mean(weights_lp, tgt, out=torch.zeros([nnodes]).to(edge_index.device))
        return localsim

    def forward(self, features, edge_index):
        feat_embeddings = self.get_feat_embedding(features)
        # struct_embeddings = self.get_structure_embedding(adj)
        # embeddings = torch.cat((feat_embeddings, struct_embeddings), 1)
        # 通过MLP2计算有连边的节点i和节点j的权重，即\theta{i,j}
        edges_weights_raw = self.get_edge_weight(feat_embeddings, edge_index)
        # 对权重采样，使用gumbel
        weights_lp = self.gumbel_sampling(edges_weights_raw)
        
        weights_hp = 1 - weights_lp
        # weights_lp 是1 x |edges|的tensor，即每条边一个权重
        edge_index_lp, edge_index_hp, edge_weights_lp, edge_weights_hp = self.weight_to_adj_pyg(edge_index, weights_lp, weights_hp)
        return  edge_index_lp, edge_index_hp, edge_weights_lp, edge_weights_hp, weights_lp

class PredictModel(nn.Module):
    def __init__(self,conf, nnodes, nfeat, nhid, nclass, device, batch_size):
        super(PredictModel, self).__init__()
        self.device = device
        self.conf = conf
        self.batch_size = batch_size
        self.conv1 = GCNConv(nfeat, nhid, normalize=False)
        self.conv2 = GCNConv(nhid, nclass, normalize=False)

        self.conv3 = GCNConv(nfeat, nhid, normalize=False)
        self.conv4 = GCNConv(nhid, nclass, normalize=False)

        self.norm = torch.nn.BatchNorm1d(nhid)
  
    def forward(self, x, edge_idx_lp, edge_idx_hp, edge_weight_lp, edge_weight_hp, training = False):
        x_lp = self.conv1(x, edge_idx_lp, edge_weight=edge_weight_lp)
        x_lp = self.norm(x_lp)
        x_lp = F.relu(x_lp)
        x_lp = F.dropout(x_lp, p=self.conf.model['dropout'], training=training)
        x_lp = self.conv2(x_lp, edge_idx_lp, edge_weight=edge_weight_lp)
        # return x_lp
        
        x_hp = self.conv3(x, edge_idx_hp, edge_weight=edge_weight_hp)
        x_hp = self.norm(x_hp)
        x_hp = F.relu(x_hp)
        x_hp = F.dropout(x_hp, p=self.conf.model['dropout'], training=training)
        x_hp = self.conv4(x_hp, edge_idx_hp, edge_weight=edge_weight_hp)
        
        return x_lp, x_hp

    def cal_cl(self, emb1, emb2):
        return self.batch_nce_loss(emb1, emb2)
    
    def set_mask_knn(self, X, k, dataset,y=None,adj=None, metric='cosine'):
        if k != 0:
            path = './data/knn/{}'.format(dataset)
            if not os.path.exists(path):
                os.makedirs(path)
            file_name = path + '/{}_{}.npz'.format(dataset, k)
            if os.path.exists(file_name):
                knn = sparse.load_npz(file_name)
                # print('Load exist knn graph.')
            else:
                print('Computing knn graph...')
                knn = kneighbors_graph(X, k, metric=metric)
                sparse.save_npz(file_name, knn)
                print('Done. The knn graph is saved as: {}.'.format(file_name))
            knn = torch.tensor(knn.toarray()) + torch.eye(X.shape[0])
        else:
            # knn = torch.eye(X.shape[0])
            knn = adj.to_dense().cpu()

        # if k == 0:
        #     knn = adj.to_dense().cpu()
        # self.count_same_label_neighbors_fast(knn,y)
        # self.count_same_label_neighbors(knn,y)
        self.pos_mask = knn
        self.neg_mask = 1 - self.pos_mask

    def batch_nce_loss(self, z1, z2, temperature=0.2, pos_mask=None, neg_mask=None):
        if pos_mask is None and neg_mask is None:
            pos_mask = self.pos_mask
            neg_mask = self.neg_mask

        nnodes = z1.shape[0]
        if (self.batch_size == 0) or (self.batch_size > nnodes):
            loss_0 = self.infonce(z1, z2, pos_mask, neg_mask, temperature)
            loss_1 = self.infonce(z2, z1, pos_mask, neg_mask, temperature)
            loss = (loss_0 + loss_1) / 2.0
        else:
            node_idxs = list(range(nnodes))
            random.shuffle(node_idxs)
            batches = self.split_batch(node_idxs, self.batch_size)
            loss = 0
            for b in batches:
                weight = len(b) / nnodes
                loss_0 = self.infonce(z1[b], z2[b], pos_mask[:,b][b,:], neg_mask[:,b][b,:], temperature)
                loss_1 = self.infonce(z2[b], z1[b], pos_mask[:,b][b,:], neg_mask[:,b][b,:], temperature)
                loss += (loss_0 + loss_1) / 2.0 * weight
        return loss

    def infonce(self, anchor, sample, pos_mask, neg_mask, tau):
        pos_mask = pos_mask.to(self.device)
        neg_mask = neg_mask.to(self.device)
        sim = self.similarity(anchor, sample) / tau
        exp_sim = torch.exp(sim) * neg_mask
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True))
        loss = log_prob * pos_mask
        loss = loss.sum(dim=1) / pos_mask.sum(dim=1)
        return -loss.mean()

    def similarity(self, h1: torch.Tensor, h2: torch.Tensor):
        h1 = F.normalize(h1)
        h2 = F.normalize(h2)
        return h1 @ h2.t()
    
    def split_batch(self, init_list, batch_size):
        groups = zip(*(iter(init_list),) * batch_size)
        end_list = [list(i) for i in groups]
        count = len(init_list) % batch_size
        end_list.append(init_list[-count:]) if count != 0 else end_list
        return end_list

    
        
class LabelDivision(nn.Module):
    def __init__(self, conf, n_nodes,device):
        super(LabelDivision, self).__init__()
        self.conf = conf
        self.device = device
        self.n_nodes = n_nodes
    
    def division(self, idx_train, emb_lp, emb_hp, label_noise, localsim, epoch,binary_y, noise_rate=None):
        z_lp= emb_lp[idx_train]
        z_hp= emb_hp[idx_train]
        
        y_train_noise = label_noise[idx_train]
        train_nodes_localsim = localsim[idx_train]

        loss_pick_lp = F.cross_entropy(z_lp, y_train_noise, reduction='none')
        loss_pick_hp = F.cross_entropy(z_hp, y_train_noise, reduction='none')
        # 这里做自适应
        # loss_pick = train_nodes_localsim * loss_pick_lp + (1-train_nodes_localsim) * loss_pick_hp
        # mean
        loss_pick = (loss_pick_lp +  loss_pick_hp)/2
        # all low or all high
        # loss_pick = loss_pick_lp +  loss_pick_hp

 
        ind_sorted = torch.argsort(loss_pick)
        loss_sorted = loss_pick[ind_sorted]
        
        ## define drop rate schedule
        forget_rate = 1.25 * noise_rate
        rate_schedule = np.ones(self.conf.training['n_epochs'])
        rate_schedule[:self.conf.training['n_epochs']] = np.linspace(0, forget_rate**1, self.conf.training['n_epochs'])
        remember_rate = 1 - rate_schedule[epoch]
        
        # print("======remember_rate: =====",remember_rate)
        num_remember = int(remember_rate * len(loss_sorted))

        '''Loss for clean labels'''
        label_clean = ind_sorted[:num_remember]
        ind_all = torch.arange(y_train_noise.shape[0]).long() 
        ind_update_1 = torch.LongTensor(list(set(ind_all.detach().cpu().numpy())-set(label_clean.detach().cpu().numpy()))).to(self.device)
        
        # return loss_label
        return label_clean,ind_update_1,loss_pick

