from predictor.Base_Predictor import Predictor
from predictor.module.adaptLoss import Edge_Discriminator, LabelDivision, PseudoLoss
import torch
import numpy as np
import time
import torch.nn.functional as F
import torch.nn as nn
import nni
from copy import deepcopy
from sklearn.metrics import roc_curve, auc
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

torch.autograd.set_detect_anomaly(True)

def plotAuc(auc_lp_epoch, auc_hp_epoch, auc_com_epoch):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 7))
    x = range(len(auc_lp_epoch))
    # print("====auc_epoch===",auc_epoch)
    plt.plot(x, auc_lp_epoch, label='auc_lp_epoch', color='blue')
    plt.plot(x, auc_hp_epoch, label='auc_hp_epoch', color='red')
    plt.plot(x, auc_com_epoch, label='auc_com_epoch', color='green')

    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.grid(True)
    plt.legend()
    plt.savefig("auc_ada_v1.png")

def calculate_auc(lce_lp,lce_hp,loss_combine, y):
    lce_lp= lce_lp.detach().cpu().numpy()
    lce_hp= lce_hp.detach().cpu().numpy()
    loss_combine= loss_combine.detach().cpu().numpy()

    y=y.cpu().numpy()
    fpr1, tpr1, thresholds1 = roc_curve(y, lce_lp, pos_label=0)

    fpr2, tpr2, thresholds2 = roc_curve(y, lce_hp, pos_label=0)
    fpr3, tpr3, thresholds3 = roc_curve(y, loss_combine, pos_label=0)


    return auc(fpr1, tpr1), auc(fpr2, tpr2),  auc(fpr3, tpr3)


class adaptloss_Predictor(Predictor):
    def __init__(self, conf, data, device='cuda:0'):
        super().__init__(conf, data, device)
        self.acc =  {'lp': -1, 'hp': -1, 'mean': -1}

    def method_init(self, conf, data):
        # self.dist = self.cal_dist( self.feats, self.edge_index)
        # unlabeled nodes 打伪标签
        self.criterion_pse = PseudoLoss()
        self.idx_unlabel = torch.LongTensor(list(set(range(self.feats.shape[0])) - set(self.train_mask))).to(self.device)
        # Edge_Discriminator是生成同配图和异配图的
        self.discriminator = Edge_Discriminator(conf, conf.model['nlayers'], self.n_nodes, conf.model['n_feat'], conf.model['emb_dim'],
                             conf.model['alpha'],  conf.dataset['sparse'], self.device, conf.model['cl_batch_size'], 
                             conf.model['hidden_dim']).to(self.device)
        # 训练之前根据特征X生成KNN邻居 用于对比学习的正负样本
        self.discriminator.set_mask_knn(self.feats.cpu(), k=conf.model['k'], dataset=data.name)
        # noise的划分和分类器
        self.criterion = LabelDivision(conf, conf.model['emb_dim'],conf.model['n_feat'], self.n_nodes, self.device).to(self.device)
        self.optimizer = torch.optim.Adam([{'params': self.discriminator.parameters(), 'lr':conf.model['lr_dis']}, {'params': self.criterion.parameters(), 'lr': conf.model['lr_cri']}],weight_decay=conf.training['weight_decay'])

    def train(self,binary_y, noise_rate):
        # epochs是外循环次数
        # lp_clean_epoch = []
        # lp_noise_epoch = []
        # hp_clean_epoch = []
        # hp_noise_epoch = []
        # ada_clean_epoch = []
        # ada_noise_epoch = []
        auc_lp_epoch = []
        auc_hp_epoch = []
        auc_com_epoch = []
        loss_array =[]

        self.noise_rate = noise_rate

        for epoch in range(self.conf.training['n_epochs']):
            self.criterion.train()
            self.discriminator.train()
            self.optimizer.zero_grad()
          
            improve = ''
            t0 = time.time()
            
            edge_index_loops, adj_lp, adj_hp, weights_lp,edges_weights_raw = self.discriminator(self.feats, self.adj, self.edge_index)
            # breakpoint()
            emb_lp, emb_hp = self.discriminator.get_embedding(self.feats, adj_lp, adj_hp)
            x_lp, x_hp = self.criterion.to_prob(emb_lp, emb_hp)

            # warm up
            if epoch < self.conf.model['warm_up']:
                loss_train = (F.cross_entropy(x_lp[self.train_mask], self.noisy_label[self.train_mask]) + F.cross_entropy(x_hp[self.train_mask], self.noisy_label[self.train_mask]))/2
                acc_train = (self.metric(self.noisy_label[self.train_mask].cpu().numpy(), x_lp[self.train_mask].detach().cpu().numpy())
                            +self.metric(self.noisy_label[self.train_mask].cpu().numpy(), x_hp[self.train_mask].detach().cpu().numpy()))/2

            else:
                # 计算对比损失
                cl_loss = self.discriminator.cal_cl(emb_lp, emb_hp)
                # 计算label_loss predict_consistency_loss   
                label_loss, loss_combine, inter_view_loss,loss_pick_lp, loss_pick_hp = self.criterion.division(self.train_mask,  emb_lp, emb_hp, self.noisy_label, self.clean_label, epoch, binary_y, edge_index=self.edge_index, weights_lp=weights_lp, noise_rate= self.noise_rate)
                # loss_train =  label_loss + cl_loss 
                loss_train =  label_loss 

                acc_train = (self.metric(self.noisy_label[self.train_mask].cpu().numpy(),x_lp[self.train_mask].detach().cpu().numpy())
                            +self.metric(self.noisy_label[self.train_mask].cpu().numpy(),x_hp[self.train_mask].detach().cpu().numpy()))/2
                
                auc_re_lp, auc_re_hp, auc_com = calculate_auc(loss_pick_lp, loss_pick_hp,loss_combine, binary_y.int()) 
                auc_lp_epoch.append(auc_re_lp)
                auc_hp_epoch.append(auc_re_hp)
                auc_com_epoch.append(auc_com)
                loss_array.append(loss_train.item())
                # lp_clean = loss_pick_lp[binary_y]
                # lp_noise = loss_pick_lp[~binary_y]
                # hp_clean = loss_pick_hp[binary_y]
                # hp_noise = loss_pick_hp[~binary_y]
                # lp_clean_epoch.append(lp_clean.mean().item())
                # lp_noise_epoch.append(lp_noise.mean().item())
                # hp_clean_epoch.append(hp_clean.mean().item())
                # hp_noise_epoch.append(hp_clean.mean().item())
                # ada_clean = loss_pick[binary_y]
                # ada_noise = loss_pick[~binary_y]
                # ada_clean_epoch.append(ada_clean.mean().item())
                # ada_noise_epoch.append(ada_noise.mean().item())

                # auc_re = self.calculate_auc(loss_pick, binary_y) 
                # auc_epoch.append(auc_re)

            # # 伪标签监督
            loss_add = self.get_pseudo_label(x_lp, x_hp)


            # breakpoint()
            total_loss = loss_train + loss_add

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
                self.criterion_weigths = deepcopy(self.criterion.state_dict())
                self.discriminator_weigths = deepcopy(self.discriminator.state_dict())

            elif flag_earlystop:
                break
            if self.conf.training['debug']:
                nni.report_intermediate_result(acc_val)
                print(
                    "Epoch {:05d} | Time(s) {:.4f} | Loss(train) {:.4f} | Acc(train) {:.4f} | Loss(val) {:.4f} | Acc(val) {:.4f} | {}".format(
                        epoch + 1, time.time() - t0, loss_train.item(), acc_train, loss_val, acc_val, improve))
        
        # self.plotloss(lp_clean_epoch,lp_noise_epoch,hp_clean_epoch,hp_noise_epoch,ada_clean_epoch,ada_noise_epoch)
        plotAuc(auc_lp_epoch, auc_hp_epoch, auc_com_epoch)


        loss_test, acc_test, acc_lp, acc_hp = self.test(self.test_mask)
        self.result['test'] = acc_test
        if self.conf.training['debug']:
            print('Optimization Finished!')
            print('Time(s): {:.4f}'.format(self.total_time))
            print("Loss(test) {:.4f} | Acc(test) {:.4f}".format(loss_test.item(), acc_test))
            print("acc_lp {:.4f} | acc_hp {:.4f}".format(acc_lp.item(), acc_hp.item()))

        return self.result

    def evaluate(self, mask, label):
        self.criterion.eval()
        self.discriminator.eval()
        edge_index, adj_lp, adj_hp, weights_lp,edges_weights_raw = self.discriminator(self.feats, self.adj, self.edge_index)
        emb1, emb2 = self.discriminator.get_embedding(self.feats, adj_lp, adj_hp)
        x_lp, x_hp = self.criterion.to_prob(emb1, emb2)

        loss_val = (F.cross_entropy(x_lp[mask], label[mask]) + F.cross_entropy(x_hp[mask], label[mask]))/2
        acc_lp = self.metric(label[mask].cpu().numpy(), x_lp[mask].detach().cpu().numpy()) 
        acc_hp = self.metric(label[mask].cpu().numpy(), x_hp[mask].detach().cpu().numpy()) 

        acc_val = (acc_lp + acc_hp)/2
        
        return loss_val, acc_val, acc_lp, acc_hp 


    def test(self, mask):
        if self.discriminator_weigths is not None:
            self.discriminator.load_state_dict(self.discriminator_weigths)
        if self.criterion_weigths is not None:
            self.criterion.load_state_dict(self.criterion_weigths)

        loss_test, acc_pred_test, acc_lp, acc_hp= self.evaluate(mask, self.clean_label)
        return loss_test, acc_pred_test, acc_lp, acc_hp 


    def get_pseudo_label(self, x_lp, x_hp):
        p_1 = F.softmax(x_lp, dim=1).detach()
        p_2 = F.softmax(x_hp, dim=1).detach()
        # 这里改为使用hp或lp的预测作为伪标签
        # filter_mask = ((p_1.max(dim=1)[1][self.idx_unlabel] == p_2.max(dim=1)[1][self.idx_unlabel])&
        #                     (p_1.max(dim=1)[0][self.idx_unlabel]*p_2.max(dim=1)[0][self.idx_unlabel] > self.conf.model['psedo_th']**2))
  
        # idx_add = self.idx_unlabel[filter_mask]
        # pseudo_label = x_lp.max(dim=1)[1]
        # loss_pick_1 = F.cross_entropy(x_lp[idx_add], pseudo_label[idx_add], reduction='none')
        # loss_pick_2 = F.cross_entropy(x_hp[idx_add], pseudo_label[idx_add], reduction='none')
        # loss = (loss_pick_1.mean() + loss_pick_2.mean())/2
        # # inter_view_loss = kl_loss_compute(y_1[idx_add], y_2[idx_add]).mean() + kl_loss_compute(y_2[idx_add], y_1[idx_add]).mean()
        # # loss = torch.mean(loss_pick)+co_lambda*inter_view_loss
        # =======强弱增强伪标签==========
        max_probs, targets_u = torch.max(p_1[self.idx_unlabel], dim=1)
        pseudo_mask = (max_probs >= self.conf.model['confidence']).float()
        loss_lp = F.cross_entropy(x_hp[self.idx_unlabel], targets_u, reduction='none')
        # loss_lp = (loss_lp * pseudo_mask).mean()
        loss_lp = loss_lp * pseudo_mask
        print("=========unlabeled pseudo_mask.sum()========",pseudo_mask.sum())
        loss = loss_lp.mean()
        return loss


    def plotloss(self,lp_clean_epoch,lp_noise_epoch,hp_clean_epoch,hp_noise_epoch,ada_clean_epoch,ada_noise_epoch):
        import matplotlib.pyplot as plt
        import numpy as np
        plt.figure(figsize=(10, 7))
        x = range(len(lp_clean_epoch))
        plt.plot(x, lp_clean_epoch , marker='o', label='lp_clean_epoch', color='blue')
        plt.plot(x, lp_noise_epoch , marker='s', label='lp_noise_epoch', color='green')
        plt.plot(x, hp_clean_epoch , marker='^', label='hp_clean_epoch', color='red')
        plt.plot(x, hp_noise_epoch, marker='d', label='hp_noise_epoch', color='purple')

        plt.plot(x, ada_clean_epoch, marker='D', label='ada_clean_epoch', color='#9467bd')
        plt.plot(x, ada_noise_epoch, marker='p', label='ada_noise_epoch', color='#8c564b')
        

        plt.xlabel('Epoch')
        plt.ylabel('loss')
        plt.grid(True)
        plt.legend()
        plt.savefig("loss_lp_hp.png")
    # 计算AUC
    def calculate_auc(self,lce, y):
        from sklearn.metrics import roc_curve, auc
        lce= lce.detach().cpu().numpy()
        y=y.cpu().numpy()
        fpr, tpr, thresholds = roc_curve(y, lce, pos_label=0)
        return auc(fpr, tpr)
    

    # def cal_dist(self, x, edge_index):
    #     def _d(x, src, tgt):
    #         if self.conf.model['localSim_method'] == 'cos':
    #             # d = (x[src] * x[tgt]).sum(dim=-1)
    #             # 计算src和tgt对应的特征向量
    #             x_src = x[src]
    #             x_tgt = x[tgt]
    #             # 对每个向量进行L2范数归一化
    #             x_src_norm = x_src / x_src.norm(dim=-1, keepdim=True)
    #             x_tgt_norm = x_tgt / x_tgt.norm(dim=-1, keepdim=True)
    #             # 计算归一化后的向量之间的点积，即余弦相似度
    #             d = (x_src_norm * x_tgt_norm).sum(dim=-1)
    #             d = (d + 1) / 2
    #         elif self.conf.model['localSim_method'] == 'norm2':
    #             d = torch.norm(x[src] - x[tgt], p=2, dim=-1)
    #         return d

    #     split_size = 10000
    #     dist = []
    #     for ei_i in tqdm(edge_index.split(split_size, dim=-1), ncols=70):
    #         src_i, tgt_i = ei_i
    #         d = _d(x, src_i, tgt_i)
    #         dist.append(d)
    #     dist = torch.cat(dist, dim=0)
       
    #     dev = x.device
    #     _, tgt = edge_index
    #     dist = scatter_mean(
    #         dist.cpu(), tgt.cpu(), out=torch.zeros([x.shape[0]])).to(dev)
    #     return dist
        


#   "psedo_th": {
#     "_type": "choice",
#     "_value": [
#       0.4,
#       0.5,
#       0.6,
#       0.7,
#       0.8
#     ]
#   },




