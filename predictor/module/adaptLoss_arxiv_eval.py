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
from torch_scatter import scatter_add, scatter_mean
import scipy.sparse as sp

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
        
        x_hp = self.conv3(x, edge_idx_hp, edge_weight=edge_weight_hp)
        x_hp = self.norm(x_hp)
        x_hp = F.relu(x_hp)
        x_hp = F.dropout(x_hp, p=self.conf.model['dropout'], training=training)
        x_hp = self.conv4(x_hp, edge_idx_hp, edge_weight=edge_weight_hp)
        
        return x_lp, x_hp


    def cal_cl(self, emb1, emb2):
        return self.batch_nce_loss(emb1, emb2)
    

    def set_mask_knn(self, X, k, dataset, metric='cosine'):
        if k != 0:
            path = './data/knn/{}'.format(dataset)
            if not os.path.exists(path):
                os.makedirs(path)
            file_name = path + '/{}_{}.npz'.format(dataset, k)

            if os.path.exists(file_name):
                knn = sparse.load_npz(file_name)
            else:
                print('Computing knn graph...')
                knn = kneighbors_graph(X, k, metric=metric, mode='connectivity', include_self=False)
                sparse.save_npz(file_name, knn)
                print('Done. The knn graph is saved as: {}.'.format(file_name))

            # 提取稀疏邻接矩阵的 COO 格式
            knn_coo = knn.tocoo()
            row = knn_coo.row
            col = knn_coo.col

            # 构建 edge_index
            knn = torch.tensor(np.vstack([row, col]), dtype=torch.long)

            # 添加自环
            num_nodes = X.shape[0]
            self_loop = torch.arange(num_nodes, dtype=torch.long)
            self_loop = torch.stack([self_loop, self_loop], dim=0)
            knn = torch.cat([knn, self_loop], dim=1)

            # 去重
            knn = torch.unique(knn, dim=1)

        else:
            # 若 k=0，仅添加自环边
            num_nodes = X.shape[0]
            self_loop = torch.arange(num_nodes, dtype=torch.long)
            self_loop = torch.stack([self_loop, self_loop], dim=0)
            knn = self_loop
        self.pos_mask = knn


    def batch_nce_loss(self, z1, z2, temperature=0.2, pos_mask=None, neg_mask=None):
        if pos_mask is None and neg_mask is None:
            pos_mask = self.pos_mask
            # neg_mask = self.neg_mask

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
                b = torch.tensor(b)
                src, dst = pos_mask
                mask_src = torch.isin(src, b)  # 起点在 batch_nodes 中的掩码
                start_edge = pos_mask[:, mask_src]
                src, dst = start_edge
                mask_dst = torch.isin(dst, b) # 终点在 batch_nodes 中的掩码
                pos_edge_index = start_edge[:,mask_dst]
                neg_edge_index = self.subgraph_negative_sampling(pos_edge_index, num_neg_samples =1000)
                pos_edge_index =  pos_edge_index.to(self.device)
                neg_edge_index =  neg_edge_index.to(self.device)

                # 构建子图 节点数为b
                # loss_0 = self.infonce(z1[b], z2[b], pos_mask[:,b][b,:], neg_mask[:,b][b,:], temperature)
                # loss_1 = self.infonce(z2[b], z1[b], pos_mask[:,b][b,:], neg_mask[:,b][b,:], temperature)
                loss_0 = self.infonce_edge_index(z1, z2, pos_edge_index, neg_edge_index, temperature)
                loss_1 = self.infonce_edge_index(z2, z1, pos_edge_index, neg_edge_index, temperature)
                # breakpoint()
                loss += (loss_0 + loss_1) / 2.0 * weight
        return loss
    

    def subgraph_negative_sampling(self, edge_index_subgraph, num_neg_samples=None, method='sparse'):
        from torch_geometric.utils import negative_sampling

        """
        在给定子图中进行负采样。

        参数:
            edge_index_subgraph (LongTensor): 子图中的边 [2, num_edges]
            num_neg_samples (int): 负样本的数量，默认为与正样本相同
            method (str): 采样方法，可选 'sparse' 或 'dense'

        返回:
            neg_edge_index (LongTensor): 采样得到的负边 [2, num_neg_samples]
        """
        # 提取子图中的节点并构造连续编号映射
        nodes = torch.unique(edge_index_subgraph)
        node_id_map = {nid.item(): i for i, nid in enumerate(nodes)}
        remapped_edges = torch.stack([
            torch.tensor([node_id_map[n.item()] for n in edge_index_subgraph[0]]),
            torch.tensor([node_id_map[n.item()] for n in edge_index_subgraph[1]])
        ], dim=0)

        num_nodes_subgraph = len(nodes)
        if num_neg_samples is None:
            num_neg_samples = edge_index_subgraph.size(1)

        # 使用 PyG 的 negative_sampling
        neg_edges_remapped = negative_sampling(
            edge_index=remapped_edges,
            num_nodes=num_nodes_subgraph,
            num_neg_samples=num_neg_samples,
            method=method
        )

        # 反映射回原始节点编号
        id_reverse_map = {v: k for k, v in node_id_map.items()}
        neg_edge_index = torch.stack([
            torch.tensor([id_reverse_map[i.item()] for i in neg_edges_remapped[0]]),
            torch.tensor([id_reverse_map[i.item()] for i in neg_edges_remapped[1]])
        ], dim=0)

        return neg_edge_index

    def infonce_edge_index(self, anchor, sample, pos_edge_index, neg_edge_index, tau):
        # 正样本余弦相似度
        pos_sim = self.similarity(anchor[pos_edge_index[0]], sample[pos_edge_index[1]]) / tau
        # 负样本余弦相似度（用于归一化分母）
        neg_sim = self.similarity(anchor[neg_edge_index[0]], sample[neg_edge_index[1]]) / tau

        # 把负样本 sim 进行 exp 并加到每个 anchor 的分母中：exp(sim_neg) 聚合
        neg_i = neg_edge_index[0]  # 每条边的源节点（anchor）
        exp_neg_sim = torch.exp(neg_sim)
        denom = torch.zeros(anchor.size(0), device=anchor.device).index_add_(0, neg_i, exp_neg_sim)
        pos_i = pos_edge_index[0]
        log_prob = pos_sim - torch.log(denom[pos_i] + 1e-8)

        # 求平均 loss（正样本可能多个）
        loss = -log_prob.mean()
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
        return torch.sum(h1 * h2, dim=1) 
        # return h1 @ h2.t()
    
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
    
    def division(self, idx_train, emb_lp, emb_hp,label_noise,localsim, epoch,binary_y, noise_rate=None):
        z_lp= emb_lp[idx_train]
        z_hp= emb_hp[idx_train]
        
        y_train_noise = label_noise[idx_train]
        train_nodes_localsim = localsim[idx_train]

        loss_pick_lp = F.cross_entropy(z_lp, y_train_noise, reduction='none')
        loss_pick_hp = F.cross_entropy(z_hp, y_train_noise, reduction='none')
        # print("=====train_nodes_localsim===========================",train_nodes_localsim)
        # 这里做自适应
        loss_pick = train_nodes_localsim * loss_pick_lp + (1-train_nodes_localsim) * loss_pick_hp
 
        ind_sorted = torch.argsort(loss_pick)
        loss_sorted = loss_pick[ind_sorted]
        
        ## define drop rate schedule
        forget_rate = 1.25 * noise_rate
        rate_schedule = np.ones(self.conf.training['n_epochs'])
        rate_schedule[:self.conf.training['n_epochs']] = np.linspace(0, forget_rate**1, self.conf.training['n_epochs'])
        remember_rate = 1 - rate_schedule[epoch]
        
        # print("======remember_rate: =====",remember_rate)
        num_remember = int(remember_rate * len(loss_sorted))

        binary_y_sorted = binary_y[ind_sorted]
        selected_y = binary_y_sorted[:num_remember]
        print("=====selected clean numbers:{} |  percent:{}===".format(selected_y.shape[0], selected_y.int().sum()/selected_y.shape[0]))

        '''Loss for clean labels'''
        label_clean = ind_sorted[:num_remember]
        ind_all = torch.arange(y_train_noise.shape[0]).long() 
        ind_update_1 = torch.LongTensor(list(set(ind_all.detach().cpu().numpy())-set(label_clean.detach().cpu().numpy()))).to(self.device)
        
        # p_1 = F.softmax(z_lp, dim=1)
        # ----------Pseudo-label cross entropy for noise data------------
        # max_probs, targets_u = torch.max(p_1[ind_update_1], dim=1)
        # pseudo_mask = max_probs >= self.conf.model['confidence']
        # loss_lp = F.cross_entropy(z_hp[ind_update_1], targets_u, reduction='none')
        # loss_lp = loss_lp * pseudo_mask.float()
        # loss_dc = loss_lp.mean()
        # # print("=========pseudo_mask.sum()========",pseudo_mask.sum())
        # loss_noise_lp =  F.cross_entropy(z_lp[ind_update_1], y_train_noise[ind_update_1], reduction='none')*(~pseudo_mask).float()
        # loss_noise_hp = F.cross_entropy(z_hp[ind_update_1], y_train_noise[ind_update_1], reduction='none')*(~pseudo_mask).float()
        # loss_noise = (loss_noise_lp.mean() + loss_noise_hp.mean())/2
        # print("=========noise.sum()========",(~pseudo_mask).sum())

        # loss_clean_lp = torch.mean( F.cross_entropy(z_lp[label_clean], y_train_noise[label_clean], reduction='none'))
        # loss_clean_hp = torch.mean( F.cross_entropy(z_hp[label_clean], y_train_noise[label_clean], reduction='none'))
        # loss_clean = (loss_clean_lp + loss_clean_hp)/2

        # loss_label = 0.1*loss_noise + loss_clean + loss_dc

        # return loss_label
        return label_clean,ind_update_1,loss_pick_lp,loss_pick_hp

