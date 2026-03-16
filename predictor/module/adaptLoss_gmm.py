from torch.nn import Sequential, Linear, ReLU
from sklearn.neighbors import kneighbors_graph
from scipy import sparse
import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor
from torch_geometric.utils import add_self_loops, degree, to_dense_adj
from torch_geometric.nn import SGConv
from torch_scatter import scatter_add, scatter_mean
from torch_geometric.utils import add_remaining_self_loops
from utils import *
import numpy as np


def kl_loss_compute(pred, soft_targets, reduce=True, tempature=1):
    pred = pred / tempature
    soft_targets = soft_targets / tempature
    kl = F.kl_div(F.log_softmax(pred, dim=1), F.softmax(soft_targets, dim=1), reduction='none')
    if reduce:
        return torch.mean(torch.sum(kl, dim=1))
    else:
        return torch.sum(kl, 1)

class Edge_Discriminator(nn.Module):
    def __init__(self,conf,nlayer, nnodes, input_feat_dim, emb_dim, alpha, sparse,device,batch_size, hidden_dim=128, temperature=1.0, bias=0.0 + 0.0001):
        super(Edge_Discriminator, self).__init__()
        self.device = device
        self.conf = conf
        self.batch_size = batch_size

        # MLP(feat)
        self.feat_embedding_layers = nn.ModuleList()
        self.feat_embedding_layers.append(nn.Linear(input_feat_dim, hidden_dim))
        # MLP(struct)
        self.struct_embedding_layers = nn.ModuleList()
        self.struct_embedding_layers.append(nn.Linear(nnodes, hidden_dim))
        # calculate homphily probability of each edge
        self.edge_mlp = nn.Linear(hidden_dim * 2, 1)
        # SGC encoder ==> embedding
        self.encoder1 = SGC(input_feat_dim, emb_dim, nlayer)
        self.encoder2 = SGC(input_feat_dim, emb_dim, nlayer)

        self.temperature = temperature
        self.bias = bias
        self.nnodes = nnodes
        self.sparse = sparse
        self.alpha = alpha

    def get_embedding(self, features, adj_lp, adj_hp):
        emb1 = self.encoder1(features,adj_lp)
        emb2 = self.encoder2(features,adj_hp)
        # return torch.cat((emb1, emb2), dim=1)
        return emb1, emb2

    def cal_cl(self, emb1, emb2):
        return self.batch_nce_loss(emb1, emb2)
    

    def get_feat_embedding(self, h):
        # 1 lagers
        for layer in self.feat_embedding_layers:
            h = layer(h)
            h = F.relu(h)
        return h
    
    def get_structure_embedding(self, h):
        # 1 lager
        for layer in self.struct_embedding_layers:
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

    def weight_forward(self, features, adj, edges):
        feat_embeddings = self.get_feat_embedding(features)
        # struct_embeddings = self.get_structure_embedding(adj)
        # embeddings = torch.cat((feat_embeddings, struct_embeddings), 1)
        embeddings = feat_embeddings
        # 通过MLP2计算有连边的节点i和节点j的权重，即\theta{i,j}
        edges_weights_raw = self.get_edge_weight(embeddings, edges)
        # 对权重采样，使用gumbel
        weights_lp = self.gumbel_sampling(edges_weights_raw)
        # weights_lp = edges_weights_raw
        weights_hp = 1 - weights_lp
        return weights_lp, weights_hp, edges_weights_raw
    
    def set_mask_knn(self, X, k, dataset, metric='cosine'):
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
            knn = torch.eye(X.shape[0])
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
        adj_lp = torch.sparse_coo_tensor(edge_index_lp, weights_lp, (self.nnodes, self.nnodes))
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
        adj_hp = torch.sparse_coo_tensor(edge_index_hp, weights_hp, (self.nnodes, self.nnodes))
        return edge_index_lp, adj_lp, adj_hp

    def forward(self, features, adj, edge_index):
        # 这里的feature是拼接了str_encoding 和 feature
        weights_lp, weights_hp, edges_weights_raw = self.weight_forward(features, adj, edge_index)

        # weights_lp 是1 x |edges|的tensor，即每条边一个权重
        # SGC 内部有一个norm 这里外部不进行norm
        edge_index, adj_lp, adj_hp = self.weight_to_adj_pyg(edge_index, weights_lp, weights_hp)
        return  edge_index, adj_lp, adj_hp, weights_lp, edges_weights_raw

        
class LabelDivision(nn.Module):
    def __init__(self, conf, emb_dim, input_feat_dim, n_nodes,device):
        super(LabelDivision, self).__init__()
        self.conf = conf
        self.device = device
        self.n_nodes = n_nodes
        # 分类MLP
        # SGC 表示拼接
        # self.gate1 = nn.Linear(emb_dim*self.conf.model['nlayers']+input_feat_dim, self.conf.model['n_classes']).to(self.device)
        # self.gate2 = nn.Linear(emb_dim*self.conf.model['nlayers']+input_feat_dim, self.conf.model['n_classes']).to(self.device)
        self.gate1 = nn.Linear(emb_dim, self.conf.model['n_classes']).to(self.device)
        self.gate2 = nn.Linear(emb_dim, self.conf.model['n_classes']).to(self.device)
        
        torch.nn.init.xavier_uniform_(self.gate1.weight, gain=1.414)
        torch.nn.init.xavier_uniform_(self.gate2.weight, gain=1.414)
        
        self.increment = 0.5/conf.training['n_epochs']
    
    def to_prob(self, emb_lp, emb_hp):        
        x_lp = self.gate1(emb_lp)
        x_hp = self.gate2(emb_hp)
        return x_lp, x_hp
    
    def forward(self, z_lp, z_hp):
        x_lp = self.gate1(z_lp)
        x_hp = self.gate2(z_hp)
        return x_lp, x_hp

    # def cal_localSim(self, edge_index, weights_lp, nnodes):
    #     src, tgt = edge_index
    #     localsim = scatter_mean(weights_lp, tgt, out=torch.zeros([nnodes]).to(edge_index.device))
    #     return localsim

    def division(self, clean_mask, idx_train, emb_lp, emb_hp,label_noise, gmm_prob, epoch):
        z_lp= emb_lp[idx_train]
        z_hp= emb_hp[idx_train]
        # obtain predictors of embedding 
        # 所有样本的预测概率
        x_lp, x_hp = self.forward(z_lp, z_hp)

        '''Loss for clean labels'''
        # label_clean_score = torch.ones(len(idx_clean)).to(self.device)
        # -----------------------------基于GMM的标签组合---------------------------
        # # 基于GMM的划分结果 添加伪标签
        # y_guess = self.guess_label(x_lp, x_hp)
        # mask_noisy = np.isin(idx_train, idx_noisy)
        # mask_clean = np.isin(idx_train, idx_clean)
        
        # gmm_prob = torch.tensor(gmm_prob).to(self.device)
        # train_label_noise = label_noise[idx_train]
        # # 转换为 one-hot 格式
        # one_hot = F.one_hot(train_label_noise, num_classes=self.conf.model['n_classes'])
        # # breakpoint()
        # y_mix_distribution = gmm_prob.view(-1,1) * one_hot + (1- gmm_prob).view(-1,1) * y_guess
        # y_mix_distribution = y_mix_distribution.to(torch.float32)
        # # breakpoint()
        # loss_clean_lp = torch.mean(F.cross_entropy(x_lp, train_label_noise, reduction='none')[mask_clean])
        # loss_clean_hp = torch.mean(F.cross_entropy(x_hp, train_label_noise, reduction='none')[mask_clean])
        # loss_clean_mean =  (loss_clean_lp + loss_clean_hp)/2

        # loss_noisy_lp = F.kl_div(F.log_softmax(x_lp,dim=-1), y_mix_distribution,reduction='none')[mask_noisy].sum(dim=-1).mean()
        # loss_noisy_hp = F.kl_div(F.log_softmax(x_hp,dim=-1), y_mix_distribution,reduction='none')[mask_noisy].sum(dim=-1).mean()
        # loss_noisy_mean = (loss_noisy_lp + loss_noisy_hp) / 2
        # inter_view_loss = kl_loss_compute(x_lp, x_hp).mean() +  kl_loss_compute(x_hp, x_lp).mean()
        # inter_view_loss = 0
        # loss_label = loss_clean_mean + loss_noisy_mean
        # --------------------------over---基于GMM的标签组合---------------------------

        y_train_noise = label_noise[idx_train]

        loss_clean_lp = torch.mean(F.cross_entropy(x_lp, y_train_noise, reduction='none')[clean_mask])
        loss_clean_hp = torch.mean(F.cross_entropy(x_hp, y_train_noise, reduction='none')[clean_mask])
        loss_clean =  (loss_clean_lp + loss_clean_hp)/2
        
        # loss_dc
        # idx_noisy = idx_train[~clean_mask]
        noisy_mask = ~clean_mask

        p_1 = F.softmax(x_lp,dim=-1)
        p_2 = F.softmax(x_hp,dim=-1)

        filter_mask = ((x_lp.max(dim=1)[1][noisy_mask] != y_train_noise[noisy_mask]) &
                            (x_lp.max(dim=1)[1][noisy_mask] == x_hp.max(dim=1)[1][noisy_mask]) &
                            (p_1.max(dim=1)[0][noisy_mask] * p_2.max(dim=1)[0][noisy_mask]  > (1-(1-min(0.5, 1/x_lp.shape[1]))*epoch/self.conf.training['n_epochs'])))
        
        # filter_mask = ((x_lp.max(dim=1)[1][noisy_mask] != y_train_noise[noisy_mask]) &
        #             (p_1.max(dim=1)[0][noisy_mask] * p_2.max(dim=1)[0][noisy_mask]  > (1-(1-min(0.5, 1/x_lp.shape[1]))*epoch/self.conf.training['n_epochs'])))
        
        # filter_mask 应该与noisy_mask 相同长度
        # TODO 验证一下有多少label能打正确的label
        # filtered_nums = binary_y[noisy_mask][filter_mask]
        print("====filter_mask.sum===", filter_mask.int().sum())

        # pre_tmp = torch.ones(len(filter_mask)).to(self.device)

        
        loss_dc = (F.cross_entropy(x_lp[noisy_mask][filter_mask],x_lp[noisy_mask].max(dim=1)[1][filter_mask], reduction='none')+ \
                                   F.cross_entropy(x_hp[noisy_mask][filter_mask], x_lp[noisy_mask].max(dim=1)[1][filter_mask], reduction='none'))
        loss_dc = loss_dc.sum()/x_lp.shape[0]

        # idx_noisy  = torch.LongTensor(list(set(idx_train_noisy.tolist()) - set(dc_idx.tolist()))).to(self.device)
        remain_mask = [~filter_mask]

        t_noise = y_train_noise[noisy_mask][remain_mask]

        loss_remain_lp = F.cross_entropy(x_lp[noisy_mask][remain_mask], t_noise, reduction='none')
        loss_remain_hp = F.cross_entropy(x_hp[noisy_mask][remain_mask], t_noise, reduction='none')
        loss_noise = loss_remain_lp  + loss_remain_hp

        loss_noise = loss_noise.sum()/x_lp.shape[0]

        # inter_view_loss = kl_loss_compute(y_1, y_2).mean() +  kl_loss_compute(y_2, y_1).mean()
        loss_label = loss_clean + loss_dc + loss_noise
        inter_view_loss = 0
        # breakpoint()
        return loss_label, inter_view_loss
     
    

    def guess_label(self, logits_lp, logits_hp, T=0.5):
        #logits_lp 模型预测结果
        import torch.nn.functional as F
        
        # 计算概率分布
        p_lp = F.softmax(logits_lp,dim=-1)
        p_hp = F.softmax(logits_hp,dim=-1)

        avg_p = (p_lp + p_hp) / 2

        # 温度缩放处理
        p_target = (avg_p ** (1 / T))
        p_target /= p_target.sum(dim=1, keepdim=True)

        return p_target

class SGC(nn.Module):
    """
    A Simple PyTorch Implementation of Logistic Regression.
    Assuming the features have been preprocessed with k-step graph propagation.
    """
    def __init__(self,in_dim, emb_dim, nlayer):
        super(SGC, self).__init__()
        self.nlayer = nlayer
        self.W = nn.Linear(in_dim, emb_dim)

    # TODO 添加残差连接
    def forward(self, x, adj):
        for _ in range(self.nlayer):
            with torch.no_grad():
                # adj 应为sparse_coo格式
                x = torch.spmm(adj, x)
        return self.W(x)
    
    # SGC 拼接中间表示版
    # def forward(self, x, adj):
    #     x_list = [x]
    #     x = torch.relu(self.W(x))

    #     for _ in range(self.nlayer):
    #         with torch.no_grad():
    #             x = torch.spmm(adj, x)
    #             x_list.append(x)
        
    #     result = torch.cat(x_list,dim=1)
    #     return result
        
    
class PseudoLoss(nn.Module):
    def __init__(self):
        super(PseudoLoss, self).__init__()

    def forward(self, y_1, y_2, idx_add,co_lambda=0.1):
        pseudo_label = y_1.max(dim=1)[1]
        loss_pick_1 = F.cross_entropy(y_1[idx_add], pseudo_label[idx_add], reduction='none')
        loss_pick_2 = F.cross_entropy(y_2[idx_add], pseudo_label[idx_add], reduction='none')
        loss_pick = loss_pick_1.mean() + loss_pick_2.mean()
        inter_view_loss = kl_loss_compute(y_1[idx_add], y_2[idx_add]).mean() + kl_loss_compute(y_2[idx_add], y_1[idx_add]).mean()
        # loss = torch.mean(loss_pick)+co_lambda*inter_view_loss
        loss = torch.mean(loss_pick)

        return loss

