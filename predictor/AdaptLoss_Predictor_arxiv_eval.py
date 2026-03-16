from predictor.Base_Predictor import Predictor
from predictor.module.adaptLoss_arxiv_eval import  Edge_Discriminator, LabelDivision, PredictModel

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

def plotAuc(auc_lp_epoch, auc_hp_epoch, auc_com_epoch):
        import matplotlib.pyplot as plt
        import numpy as np
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
        plt.savefig("auc_ada.png")

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
        # self.conf_penalty = NegEntropy()

        self.idx_unlabel = torch.LongTensor(list(set(range(self.feats.shape[0])) - set(self.train_mask))).to(self.device)
        self.predictModel = PredictModel(conf, self.n_nodes, conf.model['n_feat'], conf.model['hidden_dim'], conf.model['n_classes'],
                              self.device, conf.model['cl_batch_size']).to(self.device)
        
        self.discriminator = Edge_Discriminator(self.n_nodes, conf.model['n_feat'], conf.model['alpha'],self.device, conf.model['emb_dim']).to(self.device)

        # 训练之前根据特征X生成KNN邻居 用于对比学习的正负样本
        self.predictModel.set_mask_knn(self.feats.cpu(), k=conf.model['k'], dataset=data.name)
        
        self.criterion = LabelDivision(conf, self.n_nodes, self.device).to(self.device)
        self.optimizer = torch.optim.Adam([{'params':self.predictModel.parameters(), 'lr': conf.training['lr']}, {'params':self.discriminator.parameters(), 'lr': conf.model['lr_dis']}], weight_decay=conf.training['weight_decay'])
        
    def train(self,binary_y, noise_rate):

        self.noise_rate = noise_rate
        # epochs是外循环次数
        auc_lp_epoch = []
        auc_hp_epoch = []
        auc_com_epoch = []
        loss_array =[]

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


            # warm up
            if epoch < self.conf.model['warm_up']:
                log_lp, log_hp = self.predictModel(self.feats, self.edge_idx_lp, self.edge_idx_hp, self.edge_weight_lp, self.edge_weight_hp, training = True)
                loss_train = (F.cross_entropy(log_lp[self.train_mask], self.noisy_label[self.train_mask]) + F.cross_entropy(log_hp[self.train_mask], self.noisy_label[self.train_mask]))/2
                acc_train = (self.metric(self.noisy_label[self.train_mask].cpu().numpy(), log_lp[self.train_mask].detach().cpu().numpy())
                            +self.metric(self.noisy_label[self.train_mask].cpu().numpy(), log_hp[self.train_mask].detach().cpu().numpy()))/2
                loss_add = self.get_pseudo_label(log_lp, log_hp)
                total_loss = loss_train + loss_add
                total_loss.backward()
                self.optimizer.step()

            else:
                # 在eval状态下划分clean 和noise
                self.predictModel.eval()
                self.discriminator.eval()
                with torch.no_grad():
                    log_lp, log_hp = self.predictModel(self.feats, self.edge_idx_lp, self.edge_idx_hp, self.edge_weight_lp, self.edge_weight_hp, training = False)
                    label_clean,ind_update_1,loss_pick_lp,loss_pick_hp  = self.criterion.division(self.train_mask, log_lp, log_hp,self.noisy_label, self.localsim, epoch,binary_y, noise_rate=self.noise_rate)
                # 在这里只使用loss_train计算AUC
                auc_re_lp= calculate_auc(loss_pick_lp, binary_y.int()) 
                auc_lp_epoch.append(auc_re_lp.item()) 

                self.predictModel.train()
                self.discriminator.train()
                log_lp, log_hp = self.predictModel(self.feats, self.edge_idx_lp, self.edge_idx_hp, self.edge_weight_lp, self.edge_weight_hp, training = True)
                # 计算对比损失
                cl_loss = self.predictModel.cal_cl(log_lp, log_hp)

                z_lp= log_lp[self.train_mask]
                z_hp= log_hp[self.train_mask]
                p_1 = F.softmax(z_lp, dim=1)
                max_probs, targets_u = torch.max(p_1[ind_update_1], dim=1)
                pseudo_mask = max_probs >= self.conf.model['confidence']
                loss_lp = F.cross_entropy(z_hp[ind_update_1], targets_u, reduction='none')
                loss_lp = loss_lp * pseudo_mask.float()
                loss_dc = loss_lp.mean()
                y_train_noise = self.noisy_label[self.train_mask]
                # print("=========pseudo_mask.sum()========",pseudo_mask.sum())
                loss_noise_lp =  F.cross_entropy(z_lp[ind_update_1], y_train_noise[ind_update_1], reduction='none')*(~pseudo_mask).float()
                loss_noise_hp = F.cross_entropy(z_hp[ind_update_1], y_train_noise[ind_update_1], reduction='none')*(~pseudo_mask).float()
                loss_noise = (loss_noise_lp.mean() + loss_noise_hp.mean())/2
                print("=========noise.sum()========",(~pseudo_mask).sum())

                loss_clean_lp = torch.mean( F.cross_entropy(z_lp[label_clean], y_train_noise[label_clean], reduction='none'))
                loss_clean_hp = torch.mean( F.cross_entropy(z_hp[label_clean], y_train_noise[label_clean], reduction='none'))
                loss_clean = (loss_clean_lp + loss_clean_hp)/2

                label_loss = 0.1*loss_noise + loss_clean + loss_dc

                loss_train = cl_loss + label_loss
                # loss_train = label_loss
                loss_add = self.get_pseudo_label(log_lp, log_hp)


                print("[TRAIN] Epoch:{:04d} | CL Loss {:.4f} | laebl Loss {:.4f} ".format(epoch, cl_loss, label_loss))
                acc_train = (self.metric(self.noisy_label[self.train_mask].cpu().numpy(),log_lp[self.train_mask].detach().cpu().numpy())
                            +self.metric(self.noisy_label[self.train_mask].cpu().numpy(),log_hp[self.train_mask].detach().cpu().numpy()))/2

                # loss_add = self.get_pseudo_label(log_lp, log_hp)
                total_loss = loss_train + loss_add

                total_loss.backward()
                self.optimizer.step()

            # 不能直接和R2LP的结果直接比，因为R2LP只有训练集加了noise 验证集和测试集都是clean的
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
        
        # print("loss_pick_lp",loss_pick_lp)
        # print("=============================")
        # print("loss_pick_hp",loss_pick_hp)

        # plotAuc(auc_lp_epoch, auc_hp_epoch, auc_com_epoch)


        loss_test, acc_test, acc_lp, acc_hp = self.test(self.test_mask)
        self.result['test'] = acc_test
        if self.conf.training['debug']:
            print('Optimization Finished!')
            print('Time(s): {:.4f}'.format(self.total_time))
            print("Loss(test) {:.4f} | Acc(test) {:.4f}".format(loss_test.item(), acc_test))
            print("acc_lp {:.4f} | acc_hp {:.4f}".format(acc_lp.item(), acc_hp.item()))

        return self.result

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
        loss_lp = loss_lp * pseudo_mask
        print("=========unlabeled pseudo_mask.sum()========",pseudo_mask.sum())
        loss = loss_lp.mean()
        return loss

    # def devide_labeled_nodes(self, binary_y):

    #     self.predictModel.eval()
    #     self.discriminator.eval()
    #     with torch.no_grad():
    #         log_lp, log_hp = self.predictModel(self.feats, self.edge_idx_lp, self.edge_idx_hp, self.edge_weight_lp, self.edge_weight_hp, training = False)

    #         loss_pick_lp = F.cross_entropy(log_lp[self.train_mask], self.noisy_label[self.train_mask], reduction='none')
    #         loss_pick_hp = F.cross_entropy(log_hp[self.train_mask], self.noisy_label[self.train_mask], reduction='none')
    #         # 这里做自适应
    #         train_localsim = self.localsim[self.train_mask]
    #         # train_localsim_square = self.localsim_square[self.train_mask]

    #         # loss_combine = train_localsim_square * loss_train_lp + (1-train_localsim_square) * loss_train_hp
    #         loss_combine = train_localsim * loss_pick_lp + (1-train_localsim) * loss_pick_hp
    #         prob = self.gmm_eval(loss_combine) 
    #     print("=======prob==========", prob)  

    #     clean_mask = (prob > self.conf.model['gmm_th'])      
        
    #     selected_y = binary_y[clean_mask]
    #     print("=====selected clean numbers:{} |  percent:{}===".format(selected_y.shape[0], selected_y.sum()/selected_y.shape[0]))

    #     return clean_mask, prob, loss_combine,loss_pick_lp, loss_pick_hp

    # def gmm_eval(self,losses): 
    #     # CE = torch.nn.CrossEntropyLoss(reduction='none')
    #     # losses = CE(log_pred,train_label) 
    #     losses = (losses-losses.min())/(losses.max()-losses.min())    

    #     losses = losses.reshape(-1,1)
    #     gmm = GaussianMixture(n_components=2,max_iter=20,tol=1e-2,reg_covar=5e-4)
    #     losses = losses.cpu()
    #     gmm.fit(losses)
    #     prob = gmm.predict_proba(losses) 
    #     prob = prob[:,gmm.means_.argmin()] 
    #     return prob



