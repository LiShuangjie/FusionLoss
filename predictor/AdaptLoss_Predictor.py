from predictor.Base_Predictor import Predictor
from predictor.module.adaptLoss_new import  Edge_Discriminator, LabelDivision, PredictModel

import torch
import numpy as np
import time
import torch.nn.functional as F
import nni
from copy import deepcopy
import os
import scipy.sparse as sp
from torch_geometric.utils import add_remaining_self_loops, remove_self_loops
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
from sklearn.metrics import roc_curve, auc

class NegEntropy(object):
    def __call__(self,outputs):
        probs = torch.softmax(outputs, dim=1)
        probs = torch.clamp(probs, min=1e-10, max=1.0)
        return torch.mean(torch.sum(probs.log()*probs, dim=1))


def plot2distribution(dist1, dist2,fil, num_scales = 15, width = 0.1) :
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 7))

    upper_bound: float = max(dist1.max().item(), dist2.max().item())
    lower_bound: float = min(dist1.min().item(), dist2.min().item())
    gap: float = upper_bound - lower_bound
    dist1 = (dist1 - lower_bound) / gap
    dist1 = dist1 * 100
    dist2 = (dist2 - lower_bound) / gap
    dist2 = dist2 * 100

    x_slice: np.ndarray = np.linspace(0, 100, num_scales)

    counter1 = [0] * (num_scales - 1)
    for elem in dist1:
        for slice_idx in range(len(x_slice) - 1):
            left, right = x_slice[slice_idx], x_slice[slice_idx + 1]
            if elem == x_slice[-1]:
                counter1[-1] += 1
                break
            if left <= elem < right:
                counter1[slice_idx] += 1
                break

    counter2 = [0] * (num_scales - 1)
    for elem in dist2:
        for slice_idx in range(len(x_slice) - 1):
            left, right = x_slice[slice_idx], x_slice[slice_idx + 1]
            if elem == x_slice[-1]:
                counter2[-1] += 1
                break
            if left <= elem < right:
                counter2[slice_idx] += 1
                break

    counter1 = np.array(counter1)
    counter2 = np.array(counter2)


    plt.ylabel(f'Sample count')
    plt.xlabel(f'Loss value')

    x = list()
    for idx in range(len(x_slice) - 1):
        x.append((x_slice[idx].item() + x_slice[idx + 1]) / 2)
    x = np.array(x)

    plt.bar(x, counter1, width=width, label='Clean', color='blue', alpha=0.6)
    plt.bar(x, counter2, width=width, label='Noisy', color='red', alpha=0.6)

    plt.legend()
    plt.savefig("loss_line"+fil+".png")

# 计算AUC
def scipy_coo_matrix_to_torch_sparse_tensor(sparse_mx):
    indices1 = torch.from_numpy(np.stack([sparse_mx.row, sparse_mx.col]).astype(np.int64))
    values1 = torch.from_numpy(sparse_mx.data)
    shape1 = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices=indices1, values=values1, size=shape1)

def calculate_auc(lce_lp, y):
    lce_lp= lce_lp.detach().cpu().numpy()
    # lce_hp= lce_hp.detach().cpu().numpy()
    # loss_combine= loss_combine.detach().cpu().numpy()

    y=y.cpu().numpy()
    fpr1, tpr1, thresholds1 = roc_curve(y, lce_lp, pos_label=0)

    # fpr2, tpr2, thresholds2 = roc_curve(y, lce_hp, pos_label=0)
    # fpr3, tpr3, thresholds3 = roc_curve(y, loss_combine, pos_label=0)

    return auc(fpr1, tpr1)

def cal_filter(nnodes, beta, edge_index, transposed=False):
    edge_index = edge_index.cpu()
    N = nnodes
    beta = beta
    # csr_matrix矩阵的格式：
    edge_index, _ = remove_self_loops(edge_index=edge_index)
    adj_data = np.ones([edge_index.shape[1]], dtype=np.float32)
    if transposed:
        adj_sp = sp.csr_matrix((adj_data, (edge_index[0], edge_index[1])), shape=[N, N])
    else:
        adj_sp = sp.csr_matrix((adj_data, (edge_index[1], edge_index[0])), shape=[N, N])

    edge_index_sl, _ = add_remaining_self_loops(edge_index=edge_index)
    adj_sl_data = np.ones([edge_index_sl.shape[1]], dtype=np.float32)
    if transposed:
        adj_sl_sp = sp.csr_matrix((adj_sl_data, (edge_index_sl[0], edge_index_sl[1])), shape=[N, N])
    else:
        adj_sl_sp = sp.csr_matrix((adj_sl_data, (edge_index_sl[1], edge_index_sl[0])), shape=[N, N])

    # D-1/2
    deg = np.array(adj_sl_sp.sum(axis=1)).flatten()
    deg_sqrt_inv = np.power(deg, -0.5)
    deg_sqrt_inv[deg_sqrt_inv == float('inf')] = 0.0
    deg_sqrt_inv = sp.diags(deg_sqrt_inv)

    # filters
    I = sp.eye(N,dtype=np.float32)
    DAD = deg_sqrt_inv * adj_sp * deg_sqrt_inv
    filter_l = sp.coo_matrix(beta * I + DAD)
    filter_h = sp.coo_matrix((1. - beta) * I - DAD)
    # filter_l = sp.coo_matrix(DAD)
    # filter_h = sp.coo_matrix(I - DAD)
    
    filter_l = scipy_coo_matrix_to_torch_sparse_tensor(filter_l)
    filter_h = scipy_coo_matrix_to_torch_sparse_tensor(filter_h)
    filter_l = filter_l.coalesce()
    edge_idx_lp = filter_l.indices()
    edge_weight_lp = filter_l.values()
    filter_h = filter_h.coalesce()
    edge_idx_hp = filter_h.indices()
    edge_weight_hp = filter_h.values()
    return edge_idx_lp, edge_idx_hp, edge_weight_lp, edge_weight_hp


class adaptloss_Predictor(Predictor):
    def __init__(self, conf, data, device='cuda:0'):
        super().__init__(conf, data, device)
        self.acc =  {'lp': -1, 'hp': -1, 'mean': -1}

    def method_init(self, conf, data):
        # self.conf_penalty = NegEntropy()

        self.idx_unlabel = torch.LongTensor(list(set(range(self.feats.shape[0])) - set(self.train_mask))).to(self.device)
        self.predictModel = PredictModel(conf, self.n_nodes, conf.model['n_feat'], conf.model['hidden_dim'], conf.model['n_classes'],
                              self.device, conf.model['cl_batch_size']).to(self.device)
        
        self.discriminator = Edge_Discriminator(self.n_nodes, conf.model['n_feat'], conf.model['alpha'],self.device, conf.model['emb_dim']).to(self.device)

        # 训练之前根据特征X生成KNN邻居 用于对比学习的正负样本
        self.predictModel.set_mask_knn(self.feats.cpu(), k=conf.model['k'], y=data.labels, adj=self.adj, dataset=data.name)
        
        
        self.criterion = LabelDivision(conf, self.n_nodes, self.device).to(self.device)
        self.optimizer = torch.optim.Adam([{'params':self.predictModel.parameters(), 'lr': conf.training['lr']}, {'params':self.discriminator.parameters(), 'lr': conf.model['lr_dis']}], weight_decay=conf.training['weight_decay'])
        
    def train(self,binary_y, noise_rate):

        self.noise_rate = noise_rate

        for epoch in range(self.conf.training['n_epochs']):
            
            self.predictModel.train()
            self.discriminator.train()
            self.optimizer.zero_grad()
          
            improve = ''
            t0 = time.time()
            edge_idx_lp, edge_idx_hp, edge_weight_lp, edge_weight_hp, weights_lp = self.discriminator(self.feats,self.edge_index)
            self.edge_idx_lp = edge_idx_lp.to(self.device)
            self.edge_idx_hp = edge_idx_hp.to(self.device)
            self.edge_weight_lp = edge_weight_lp.to(self.device)
            self.edge_weight_hp = edge_weight_hp.to(self.device)
            self.localsim = self.discriminator.cal_localSim(self.edge_index, weights_lp, self.n_nodes)

            # fusion
            log_lp, log_hp = self.predictModel(self.feats, self.edge_idx_lp, self.edge_idx_hp, self.edge_weight_lp, self.edge_weight_hp, training = True)
            # all low
            # log_lp, log_hp = self.predictModel(self.feats, self.edge_idx_lp, self.edge_idx_lp, self.edge_weight_lp, self.edge_weight_lp, training = True)
            # all high
            # log_lp, log_hp = self.predictModel(self.feats, self.edge_idx_hp, self.edge_idx_hp, self.edge_weight_hp, self.edge_weight_hp, training = True)
            
            # log_lp = self.predictModel(self.feats, self.edge_idx_lp, self.edge_idx_hp, self.edge_weight_lp, self.edge_weight_hp, training = True)

            # warm up
            # if epoch < self.conf.model['warm_up']:
            if epoch < 0:
                # loss_train = F.cross_entropy(log_lp[self.train_mask], self.noisy_label[self.train_mask],reduction='none')
                # acc_train = self.metric(self.noisy_label[self.train_mask].cpu().numpy(), log_lp[self.train_mask].detach().cpu().numpy())
                # loss_train = F.cross_entropy(log_hp[self.train_mask], self.noisy_label[self.train_mask],reduction='none')             
                # acc_train = self.metric(self.noisy_label[self.train_mask].cpu().numpy(), log_hp[self.train_mask].detach().cpu().numpy())
                
                # loss_train_pick = (F.cross_entropy(log_lp[self.train_mask], self.noisy_label[self.train_mask],reduction='none') + F.cross_entropy(log_hp[self.train_mask], self.noisy_label[self.train_mask],reduction='none'))/2

                loss_train = (F.cross_entropy(log_lp[self.train_mask], self.noisy_label[self.train_mask]) + F.cross_entropy(log_hp[self.train_mask], self.noisy_label[self.train_mask]))/2
                acc_train = (self.metric(self.noisy_label[self.train_mask].cpu().numpy(), log_lp[self.train_mask].detach().cpu().numpy())
                            +self.metric(self.noisy_label[self.train_mask].cpu().numpy(), log_hp[self.train_mask].detach().cpu().numpy()))/2
                # 在这里只使用loss_train计算AUC
                # auc_re_lp= calculate_auc(loss_train_pick, binary_y.int()) 
                # auc_lp_epoch.append(auc_re_lp.item())  
                # loss_train = torch.mean(loss_train)              

            else:
                cl_loss = self.predictModel.cal_cl(log_lp, log_hp)
                label_loss, loss_pick = self.criterion.division(self.train_mask, log_lp, log_hp,self.noisy_label, self.localsim, epoch,binary_y, noise_rate=self.noise_rate)
                
                # w/o cl_loss
                # loss_train = label_loss
                loss_train = cl_loss + label_loss

                print("[TRAIN] Epoch:{:04d} |  laebl Loss {:.4f} ".format(epoch, label_loss))
                acc_train = (self.metric(self.noisy_label[self.train_mask].cpu().numpy(),log_lp[self.train_mask].detach().cpu().numpy())
                            +self.metric(self.noisy_label[self.train_mask].cpu().numpy(),log_hp[self.train_mask].detach().cpu().numpy()))/2

            loss_add = self.get_pseudo_label(log_lp, log_hp)
            total_loss = loss_train + loss_add
            # w/o Un
            # total_loss = loss_train 

            total_loss.backward()
            self.optimizer.step()

            loss_val, acc_val, acc_lp, acc_hp = self.evaluate(self.val_mask, self.noisy_label)


            flag, flag_earlystop = self.recoder.add(loss_val, acc_val)
            if flag:
                improve = '*'
                self.total_time = time.time() - self.start_time
                self.best_val_loss = loss_val
                self.result['valid'] = acc_val
                self.result['train'] = acc_train
                self.acc['lp'] = acc_lp
                self.acc['hp'] = acc_hp
                self.predictModel_weigths = deepcopy(self.predictModel.state_dict())
                self.discriminator_weigths = deepcopy(self.discriminator.state_dict())

            elif flag_earlystop:
                break
            if self.conf.training['debug']:
                nni.report_intermediate_result(acc_val)
                print(
                    "Epoch {:05d} | Time(s) {:.4f} | Loss(train) {:.4f} | Acc(train) {:.4f} | Loss(val) {:.4f} | Acc(val) {:.4f} | {}".format(
                        epoch + 1, time.time() - t0, loss_train.item(), acc_train, loss_val, acc_val, improve))


        loss_test, acc_test, acc_lp, acc_hp = self.test(self.test_mask)
        self.result['test'] = acc_test
        if self.conf.training['debug']:
            print('Optimization Finished!')
            print('Time(s): {:.4f}'.format(self.total_time))
            print("Loss(test) {:.4f} | Acc(test) {:.4f}".format(loss_test.item(), acc_test))
            print("acc_lp {:.4f} | acc_hp {:.4f}".format(acc_lp.item(), acc_hp.item()))

        return self.result, _
            

    def evaluate(self, mask, label):
        self.predictModel.eval()
        self.discriminator.eval()
        
        log_lp, log_hp = self.predictModel(self.feats, self.edge_idx_lp, self.edge_idx_hp, self.edge_weight_lp, self.edge_weight_hp, training = False)
       
        acc_val_lp = self.metric(label[mask].cpu().numpy(), log_lp[mask].detach().cpu().numpy()) 
        acc_val_hp = self.metric(label[mask].cpu().numpy(), log_hp[mask].detach().cpu().numpy()) 
        acc_val = (acc_val_lp + acc_val_hp) /2

        loss_val_lp = F.cross_entropy(log_lp[mask], label[mask])
        loss_val_hp = F.cross_entropy(log_hp[mask], label[mask])
        loss_val = (loss_val_lp + loss_val_hp) /2

        return loss_val, acc_val, acc_val_lp, acc_val_hp 



    def test(self, mask):
        if self.predictModel_weigths is not None:
            self.predictModel.load_state_dict(self.predictModel_weigths)
            self.discriminator.load_state_dict(self.discriminator_weigths)

        loss_test, acc_pred_test, acc_lp, acc_hp= self.evaluate(mask, self.clean_label)
        return loss_test, acc_pred_test, acc_lp, acc_hp 

    
    def get_pseudo_label(self, x_lp, x_hp):
        p_1 = F.softmax(x_lp, dim=1).detach()
        p_2 = F.softmax(x_hp, dim=1).detach()
        max_probs, targets_u = torch.max(p_1[self.idx_unlabel], dim=1)
        pseudo_mask = (max_probs >= self.conf.model['confidence']).float()
        loss_lp = F.cross_entropy(x_hp[self.idx_unlabel], targets_u, reduction='none')
        loss_lp = loss_lp * pseudo_mask
        loss = loss_lp.mean()
        return loss


