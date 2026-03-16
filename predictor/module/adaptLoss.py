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
    kl = F.kl_div(F.log_softmax(pred, dim=1), F.softmax(soft_targets, dim=1), reduce=False)
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
        self.edge_mlp = nn.Linear(hidden_dim * 2, 1)
        # encoder [feature || structure] use hidden_dim * 4
        # self.edge_mlp = nn.Linear(hidden_dim * 4, 1)

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
        edge_index, adj_lp, adj_hp = self.weight_to_adj_pyg(edge_index, weights_lp, weights_hp)
        return  edge_index, adj_lp, adj_hp, weights_lp, edges_weights_raw
    def get_weights(self, features, adj, edge_index):
        # 这里的feature是拼接了str_encoding 和 feature
        weights_lp, _, _ = self.weight_forward(features, adj, edge_index)
        return  weights_lp
        
class LabelDivision(nn.Module):
    def __init__(self, conf, emb_dim, input_feat_dim, n_nodes,device):
        super(LabelDivision, self).__init__()
        self.conf = conf
        self.device = device
        self.n_nodes = n_nodes
        self.gate1 = nn.Linear(emb_dim, self.conf.model['n_classes']).to(self.device)
        self.gate2 = nn.Linear(emb_dim, self.conf.model['n_classes']).to(self.device)
        
        torch.nn.init.xavier_uniform_(self.gate1.weight, gain=1.414)
        torch.nn.init.xavier_uniform_(self.gate2.weight, gain=1.414)
        
        # self.increment = 0.5/conf.training['n_epochs']
    
    def to_prob(self, emb_lp, emb_hp):        
        x_lp = self.gate1(emb_lp)
        x_hp = self.gate2(emb_hp)
        return x_lp, x_hp
    
    def forward(self, z_lp, z_hp):
        x_lp = self.gate1(z_lp)
        x_hp = self.gate2(z_hp)
        return x_lp, x_hp

    def cal_localSim(self, edge_index, weights_lp, nnodes):
        src, tgt = edge_index
        localsim = scatter_mean(weights_lp, tgt, out=torch.zeros([nnodes]).to(edge_index.device))
        return localsim

    def plotloss(self,loss):
        import matplotlib.pyplot as plt
        import numpy as np
        plt.figure(figsize=(10, 7))
        plt.plot(range(loss.shape[0]), loss.cpu().detach().numpy(), label='loss')
        plt.xlabel('idx')
        plt.ylabel('loss')
        plt.grid(True)
        plt.legend()
        plt.savefig("loss.png")

    def division(self, idx_train, emb_lp, emb_hp,label_noise,clean_label, epoch,binary_y, edge_index, weights_lp=None, T = 0.5, noise_rate=None):
        z_lp= emb_lp[idx_train]
        z_hp= emb_hp[idx_train]
        # obtain predictors of embedding 
        x_lp, x_hp = self.forward(z_lp, z_hp)
        
        y_train_noise = label_noise[idx_train]
        # breakpoint()
        # dist 相当于localsim
        localsim = self.cal_localSim(edge_index, weights_lp, self.n_nodes)
        train_nodes_localsim = localsim[idx_train]

        loss_pick_lp = F.cross_entropy(x_lp, y_train_noise, reduction='none')
        loss_pick_hp = F.cross_entropy(x_hp, y_train_noise, reduction='none')
        # print("==========train_nodes_localsim======",train_nodes_localsim)
        # 这里做自适应
        loss_pick = train_nodes_localsim * loss_pick_lp + (1-train_nodes_localsim) * loss_pick_hp

        ind_sorted = torch.argsort(loss_pick)
        loss_sorted = loss_pick[ind_sorted]
        
        # forget_rate = 0.5 * (epoch /self.conf.training['n_epochs'])
        # remember_rate = 1 - forget_rate
        # mean_v = loss_sorted.mean()
        # idx_small = torch.where(loss_sorted < mean_v)[0]
        # remember_rate_small = idx_small.shape[0]/y_train_noise.shape[0]
        # remember_rate = max(remember_rate, remember_rate_small)
        
        ## define drop rate schedule
        forget_rate = 1.25 * noise_rate
        rate_schedule = np.ones(self.conf.training['n_epochs'])
        rate_schedule[:self.conf.training['n_epochs']] = np.linspace(0, forget_rate**1, self.conf.training['n_epochs'])
        remember_rate = 1 - rate_schedule[epoch]
        
        print("======remember_rate: =====",remember_rate)
        num_remember = int(remember_rate * len(loss_sorted))
        # breakpoint()
        # 添加按类选择呢？
        binary_y_sorted = binary_y[ind_sorted]
        selected_y = binary_y_sorted[:num_remember]
        self.plotloss(loss_sorted)
        # breakpoint()

        print("=====selected clean numbers:{} |  percent:{}===".format(selected_y.shape[0], selected_y.int().sum()/selected_y.shape[0]))

        '''Loss for clean labels'''
        label_clean = ind_sorted[:num_remember]

        ind_all = torch.arange(y_train_noise.shape[0]).long() 
        ind_update_1 = torch.LongTensor(list(set(ind_all.detach().cpu().numpy())-set(label_clean.detach().cpu().numpy()))).to(self.device)
        
        p_1 = F.softmax(x_lp, dim=1)
        # p_2 = F.softmax(x_hp, dim=1)

        # ----------Pseudo-label cross entropy for noise data------------
        max_probs, targets_u = torch.max(p_1[ind_update_1], dim=1)
        pseudo_mask = max_probs >= self.conf.model['confidence']
        loss_lp = F.cross_entropy(x_hp[ind_update_1], targets_u, reduction='none')
        loss_lp = loss_lp * pseudo_mask.float()
        loss_dc = loss_lp.mean()
        print("=========pseudo_mask.sum()========",pseudo_mask.sum())
        loss_noise_lp =  F.cross_entropy(x_lp[ind_update_1], y_train_noise[ind_update_1], reduction='none')*(~pseudo_mask).float()
        loss_noise_hp = F.cross_entropy(x_hp[ind_update_1], y_train_noise[ind_update_1], reduction='none')*(~pseudo_mask).float()
        loss_noise = (loss_noise_lp.mean() + loss_noise_hp.mean())/2
        print("=========noise.sum()========",(~pseudo_mask).sum())

        loss_clean_lp = torch.mean( F.cross_entropy(x_lp[label_clean], y_train_noise[label_clean], reduction='none'))
        loss_clean_hp = torch.mean( F.cross_entropy(x_hp[label_clean], y_train_noise[label_clean], reduction='none'))
        loss_clean = (loss_clean_lp + loss_clean_hp)/2

        loss_label = 0.1*loss_noise + loss_clean + loss_dc

        return loss_label, loss_pick, 0, loss_pick_lp, loss_pick_hp
        # -----------------over---------------

        # filter_mask = ((x_lp.max(dim=1)[1][ind_update_1] != y_train_noise[ind_update_1]) &
        #             (x_lp.max(dim=1)[1][ind_update_1] == x_hp.max(dim=1)[1][ind_update_1]) &
        #             ( p_1.max(dim=1)[0][ind_update_1] * p_2.max(dim=1)[0][ind_update_1] > (1-(1-min(0.5, 1/x_lp.shape[1]))*epoch/self.conf.training['n_epochs'])))
        # # filter_mask = ((x_lp.max(dim=1)[1][ind_update_1] != y_train_noise[ind_update_1]) &
        # #                     ( p_1.max(dim=1)[0][ind_update_1] * p_2.max(dim=1)[0][ind_update_1] > (1-(1-min(0.5, 1/x_lp.shape[1]))*epoch/self.conf.training['n_epochs'])))
        
        # label_correct = ind_update_1[filter_mask]
     
        # label_correct_score = (p_1.max(dim=1)[0][label_correct]*p_2.max(dim=1)[0][label_correct])**0.5
        # # # 纠正标签
        # y_train_noise[label_correct] = x_lp.max(dim=1)[1][label_correct]
        # # p_1_mask = p_1.max(dim=1)[0][label_correct] > p_2.max(dim=1)[0][label_correct]
        # # # breakpoint()
        # # y_train_noise[label_correct][p_1_mask] = x_lp.max(dim=1)[1][label_correct][p_1_mask] 
        # # y_train_noise[label_correct][~p_1_mask] = x_hp.max(dim=1)[1][label_correct][~p_1_mask]
        # # y_train_noise[label_correct] = clean_label[idx_train][label_correct] 

        # # y_train_noise[label_correct] = torch.where(mask_lp, x_lp.max(dim=1)[1][label_correct], x_hp.max(dim=1)[1][label_correct])
        # # 验证选择的高置信节点数，和能正确打伪标签的比例 self.clean_label
        # # breakpoint()
        # y_update_clean = clean_label[idx_train][label_correct]        
        # update_correct_percent = (y_update_clean == y_train_noise[label_correct]).int().sum() / label_correct.shape[0]
        # print("======label_update.nums=====, percent:=====",label_correct.shape[0], update_correct_percent.item())

        # label_remain = torch.LongTensor(list(set(ind_update_1.detach().cpu().numpy())-set(label_correct.detach().cpu().numpy()))).to(self.device)
        # label_remain_score = 0.1 * torch.ones(len(label_remain)).to(self.device)
        
        # index = torch.cat((label_clean, label_correct, label_remain))
        # score = torch.cat((label_clean_score, label_correct_score, label_remain_score))
        # score_temp = torch.ones(len(idx_train)).to(self.device)
        # score_temp[index] = score
        # score_temp = score_temp.detach()
        # # breakpoint()
        # loss_lp = torch.mean(score_temp*F.cross_entropy(x_lp[index], y_train_noise[index], reduction='none'))
        # loss_hp = torch.mean(score_temp*F.cross_entropy(x_hp[index], y_train_noise[index], reduction='none'))
        # loss_label = (loss_lp + loss_hp)/ 2
        # # breakpoint()
        # # inter_view_loss = kl_loss_compute(x_lp, x_hp).mean() +  kl_loss_compute(x_hp, x_lp).mean()
        # inter_view_loss = 0

        # return loss_label, loss_pick, inter_view_loss  

        # 下面的代码是验证label加权是否有效----------------------
        # 转换为 one-hot 格式
        # one_hot = F.one_hot(y_train_noise, num_classes=self.conf.model['n_classes'])
        # # breakpoint()
        # # y_mix_distribution = train_nodes_localsim * p_1 + (1- train_nodes_localsim) * one_hot
        # # breakpoint()
        # y_mix_distribution = train_nodes_localsim.view(-1,1) * p_1 + (1- train_nodes_localsim).view(-1,1) * p_2

        # # 温度缩放处理
        # p_target = (y_mix_distribution ** (1 / T))
        # p_target /= p_target.sum(dim=1, keepdim=True)

        # loss_clean_lp = torch.mean(F.cross_entropy(x_lp, y_train_noise, reduction='none')[label_clean])
        # loss_clean_hp = torch.mean(F.cross_entropy(x_hp, y_train_noise, reduction='none')[label_clean])
        # loss_clean_mean =  (loss_clean_lp + loss_clean_hp)/2
        
        # loss_noisy_lp = F.kl_div(F.log_softmax(x_lp,dim=-1), p_target,reduction='none')[ind_update_1].sum(dim=-1).mean()
        # loss_noisy_hp = F.kl_div(F.log_softmax(x_hp,dim=-1), p_target,reduction='none')[ind_update_1].sum(dim=-1).mean()
        # loss_noisy_mean = (loss_noisy_lp + loss_noisy_hp) / 2

        # loss_label = loss_clean_mean + loss_noisy_mean
        # ---------------over-------------------------------
     

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
        loss_pick_1 = F.cross_entropy(y_1[idx_add], pseudo_label[idx_add], reduce=False)
        loss_pick_2 = F.cross_entropy(y_2[idx_add], pseudo_label[idx_add], reduce=False)
        loss_pick = loss_pick_1.mean() + loss_pick_2.mean()
        inter_view_loss = kl_loss_compute(y_1[idx_add], y_2[idx_add]).mean() + kl_loss_compute(y_2[idx_add], y_1[idx_add]).mean()
        # loss = torch.mean(loss_pick)+co_lambda*inter_view_loss
        loss = torch.mean(loss_pick)

        return loss

