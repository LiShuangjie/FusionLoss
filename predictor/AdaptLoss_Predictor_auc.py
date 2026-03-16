from predictor.Base_Predictor import Predictor
from predictor.module.adaptLoss_new_auc import  Edge_Discriminator, LabelDivision, PredictModel

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


def plotAuc(auc_lp_epoch):
        import matplotlib.pyplot as plt
        import numpy as np
        plt.figure(figsize=(10, 7))
        x = range(len(auc_lp_epoch))
        # print("====auc_epoch===",auc_epoch)
        plt.plot(x, auc_lp_epoch, label='auc_lp_epoch', color='blue')

        plt.xlabel('Epoch')
        plt.ylabel('AUC')
        plt.grid(True)
        plt.legend()
        plt.savefig("auc_ada.png")

# 计算AUC
def scipy_coo_matrix_to_torch_sparse_tensor(sparse_mx):
    indices1 = torch.from_numpy(np.stack([sparse_mx.row, sparse_mx.col]).astype(np.int64))
    values1 = torch.from_numpy(sparse_mx.data)
    shape1 = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices=indices1, values=values1, size=shape1)

def calculate_auc(lce_lp, y):
    lce_lp= lce_lp.detach().cpu().numpy()
    y=y.cpu().numpy()
    fpr1, tpr1, thresholds1 = roc_curve(y, lce_lp, pos_label=0)
    return auc(fpr1, tpr1)

class adaptloss_Predictor(Predictor):
    def __init__(self, conf, data, device='cuda:0'):
        super().__init__(conf, data, device)
        self.acc =  {'lp': -1, 'hp': -1, 'mean': -1}

    def method_init(self, conf, data):

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
        auc_epoch = []
        # auc_hp_epoch = []
        # auc_com_epoch = []
        # # loss_array =[]

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

            # log_lp, log_hp = self.predictModel(self.feats, self.edge_idx_lp, self.edge_idx_hp, self.edge_weight_lp, self.edge_weight_hp, training = True)

            # warm up
            if epoch < 0:
                log_lp, log_hp = self.predictModel(self.feats, self.edge_idx_lp, self.edge_idx_hp, self.edge_weight_lp, self.edge_weight_hp, training = True)
                # log_lp= self.predictModel(self.feats, self.edge_idx_lp, self.edge_idx_hp, self.edge_weight_lp, self.edge_weight_hp, training = True)
                # loss_train = (F.cross_entropy(log_lp[self.train_mask], self.noisy_label[self.train_mask]) + F.cross_entropy(log_hp[self.train_mask], self.noisy_label[self.train_mask]))/2
                # acc_train = (self.metric(self.noisy_label[self.train_mask].cpu().numpy(), log_lp[self.train_mask].detach().cpu().numpy())
                #             +self.metric(self.noisy_label[self.train_mask].cpu().numpy(), log_hp[self.train_mask].detach().cpu().numpy()))/2
                # loss_add = self.get_pseudo_label(log_lp, log_hp)
                loss_train = F.cross_entropy(log_lp[self.train_mask], self.noisy_label[self.train_mask])
                acc_train = self.metric(self.noisy_label[self.train_mask].cpu().numpy(), log_lp[self.train_mask].detach().cpu().numpy())
                # total_loss = loss_train + loss_add
                loss_train.backward()
                self.optimizer.step()

            else:
                # 在eval状态下划分clean 和noise
                self.predictModel.eval()
                self.discriminator.eval()

                with torch.no_grad():
                    # 全部使用低通/高通/fusion的auc
                    # fusion
                    # log_lp, log_hp = self.predictModel(self.feats, self.edge_idx_lp, self.edge_idx_hp, self.edge_weight_lp, self.edge_weight_hp, training = False)
                    # all low
                    log_lp, log_hp = self.predictModel(self.feats, self.edge_idx_lp, self.edge_idx_lp, self.edge_weight_lp, self.edge_weight_lp, training = False)
                    # all high
                    # log_lp, log_hp = self.predictModel(self.feats, self.edge_idx_hp, self.edge_idx_hp, self.edge_weight_hp, self.edge_weight_hp, training = False)
                    label_clean,ind_update_1,loss_pick_lp   = self.criterion.division(self.train_mask, log_lp, log_hp,self.noisy_label, self.localsim, epoch,binary_y, noise_rate=self.noise_rate)
                # 计算AUC
                auc= calculate_auc(loss_pick_lp, binary_y.int()) 
                # auc_lp_epoch.append(auc_re_lp.item()) 
                auc_epoch.append(round(auc.item(), 2))

                self.predictModel.train()
                self.discriminator.train()

                log_lp, log_hp = self.predictModel(self.feats, self.edge_idx_lp, self.edge_idx_hp, self.edge_weight_lp, self.edge_weight_hp, training = True)
                # 计算对比损失
                cl_loss = self.predictModel.cal_cl(log_lp, log_hp)

                z_lp= log_lp[self.train_mask]
                z_hp= log_hp[self.train_mask]
   
                p_1 = F.softmax(z_lp, dim=1)
                y_train_noise = self.noisy_label[self.train_mask]

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

                loss_clean_lp = torch.mean( F.cross_entropy(z_lp[label_clean], y_train_noise[label_clean], reduction='none'))
                loss_clean_hp = torch.mean( F.cross_entropy(z_hp[label_clean], y_train_noise[label_clean], reduction='none'))
                loss_clean = (loss_clean_lp + loss_clean_hp)/2

                # loss_label = 0.1*loss_noise + loss_clean + loss_dc
                loss_label = loss_clean
                loss_add = self.get_pseudo_label(log_lp, log_hp)
                loss_train = loss_label + loss_add + cl_loss

                acc_train = (self.metric(self.noisy_label[self.train_mask].cpu().numpy(), log_lp[self.train_mask].detach().cpu().numpy())
                            +self.metric(self.noisy_label[self.train_mask].cpu().numpy(), log_hp[self.train_mask].detach().cpu().numpy()))/2
                
                
                loss_train.backward()
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
            # print("acc_lp {:.4f} | acc_hp {:.4f}".format(acc_lp.item(), acc_hp)) 
        
        # print("auc_lp_epoch",auc_epoch)
        # plotAuc(auc_epoch)
        return self.result, auc_epoch
    
    def evaluate(self, mask, label):
        self.predictModel.eval()
        
        log_lp,log_hp = self.predictModel(self.feats, self.edge_idx_lp, self.edge_idx_hp, self.edge_weight_lp, self.edge_weight_hp, training = False)
       
        acc_val_lp = self.metric(label[mask].cpu().numpy(), log_lp[mask].detach().cpu().numpy()) 
        acc_val_hp = self.metric(label[mask].cpu().numpy(), log_hp[mask].detach().cpu().numpy()) 
        acc_val = (acc_val_lp + acc_val_hp)/2

        loss_val_lp = F.cross_entropy(log_lp[mask], label[mask])
        loss_val_hp = F.cross_entropy(log_hp[mask], label[mask])
        loss_val = (loss_val_lp + loss_val_hp) /2
        # loss_val = loss_val_lp 
        return loss_val, acc_val, acc_val_lp, loss_val_hp


    def test(self, mask):
        if self.predictModel_weigths is not None:
            self.predictModel.load_state_dict(self.predictModel_weigths)

        loss_test, acc_pred_test, acc_lp, acc_hp= self.evaluate(mask, self.clean_label)
        return loss_test, acc_pred_test, acc_lp, acc_hp 

    
    # =======强弱增强伪标签==========
    def get_pseudo_label(self, x_lp, x_hp):
        p_1 = F.softmax(x_lp, dim=1).detach()
        p_2 = F.softmax(x_hp, dim=1).detach()
        max_probs, targets_u = torch.max(p_1[self.idx_unlabel], dim=1)
        pseudo_mask = (max_probs >= self.conf.model['confidence']).float()
        loss_lp = F.cross_entropy(x_hp[self.idx_unlabel], targets_u, reduction='none')
        loss_lp = loss_lp * pseudo_mask
        print("=========unlabeled pseudo_mask.sum()========",pseudo_mask.sum())
        loss = loss_lp.mean()
        return loss


