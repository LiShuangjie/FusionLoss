from predictor.Base_Predictor import Predictor
from predictor.module.adaptLoss_gmm import Edge_Discriminator, LabelDivision, PseudoLoss
import torch
import numpy as np
import time
import torch.nn.functional as F
import torch.nn as nn
import nni
from copy import deepcopy
from tqdm import tqdm
import os
from torch_scatter import scatter_mean
from sklearn.mixture import GaussianMixture

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

torch.autograd.set_detect_anomaly(True)

class NegEntropy(object):
    def __call__(self,outputs):
        probs = torch.softmax(outputs, dim=1)
        probs = torch.clamp(probs, min=1e-10, max=1.0)
        return torch.mean(torch.sum(probs.log()*probs, dim=1))

class adaptloss_Predictor(Predictor):
    def __init__(self, conf, data, device='cuda:0'):
        super().__init__(conf, data, device)
        self.acc =  {'lp': -1, 'hp': -1, 'mean': -1}

    def method_init(self, conf, data):
        # self.dist = self.cal_dist( self.feats, self.edge_index)
        self.conf_penalty = NegEntropy()
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
        auc_epoch = []
        self.binary_y = binary_y

        for epoch in range(self.conf.training['n_epochs']):
            self.criterion.train()
            self.discriminator.train()
            self.optimizer.zero_grad()
          
            improve = ''
            t0 = time.time()
            
            edge_index_loops, adj_lp, adj_hp, weights_lp,edges_weights_raw = self.discriminator(self.feats, self.adj, self.edge_index)
            emb_lp, emb_hp = self.discriminator.get_embedding(self.feats, adj_lp, adj_hp)
            x_lp, x_hp = self.criterion.to_prob(emb_lp, emb_hp)

            # warm up
            if epoch < self.conf.model['warm_up']:
                loss_train = (F.cross_entropy(x_lp[self.train_mask], self.noisy_label[self.train_mask]) + F.cross_entropy(x_hp[self.train_mask], self.noisy_label[self.train_mask]))/2
                acc_train = (self.metric(self.noisy_label[self.train_mask].cpu().numpy(), x_lp[self.train_mask].detach().cpu().numpy())
                            +self.metric(self.noisy_label[self.train_mask].cpu().numpy(), x_hp[self.train_mask].detach().cpu().numpy()))/2
                penalty_lp = self.conf_penalty(x_lp)
                penalty_hp = self.conf_penalty(x_hp)
                penalty = (penalty_lp + penalty_hp) /2
                loss_train = loss_train + penalty

            else:
                # 计算对比损失
                cl_loss = self.discriminator.cal_cl(emb_lp, emb_hp)
                # 计算label_loss predict_consistency_loss 
                clean_mask, prob, loss_pick = self.devide_labeled_nodes(self.noisy_label, self.criterion, self.train_mask, self.edge_index, emb_lp, emb_hp, binary_y, weights_lp=weights_lp) 
                label_loss, inter_view_loss = self.criterion.division(clean_mask, self.train_mask,  emb_lp, emb_hp, self.noisy_label, prob,epoch)
                # loss_train = cl_loss + label_loss + inter_view_loss
                # breakpoint()
                loss_train =  label_loss + cl_loss
                # print("[TRAIN] Epoch:{:04d} | CL Loss {:.4f} | laebl Loss {:.4f} ".format(epoch, cl_loss, label_loss))
                acc_train = (self.metric(self.noisy_label[self.train_mask].cpu().numpy(),x_lp[self.train_mask].detach().cpu().numpy())
                            +self.metric(self.noisy_label[self.train_mask].cpu().numpy(),x_hp[self.train_mask].detach().cpu().numpy()))/2
            
                auc_re = self.calculate_auc(loss_pick, binary_y.int()) 
                print("===auc_re===",auc_re)
                # auc_epoch.append(auc_re)
            # breakpoint()
            # 伪标签监督
            self.idx_add= self.get_pseudo_label(x_lp, x_hp)
            if len(self.idx_add) != 0:
                print("======len(self.idx_add)======",len(self.idx_add))
                loss_add = self.criterion_pse(x_lp, x_hp, self.idx_add)
            else:
                loss_add = torch.Tensor([0]).to(self.device)

            total_loss = loss_train + loss_add
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
                self.criterion_weigths = deepcopy(self.criterion.state_dict())
                self.discriminator_weigths = deepcopy(self.discriminator.state_dict())

            elif flag_earlystop:
                break
            if self.conf.training['debug']:
                nni.report_intermediate_result(acc_val)
                print(
                    "Epoch {:05d} | Time(s) {:.4f} | Loss(train) {:.4f} | Acc(train) {:.4f} | Loss(val) {:.4f} | Acc(val) {:.4f} | {}".format(
                        epoch + 1, time.time() - t0, loss_train.item(), acc_train, loss_val, acc_val, improve))
        
        # self.plotAuc(self.conf.training['n_epochs']-self.conf.model['warm_up'],auc_epoch)

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
        pred0 = F.softmax(x_lp, dim=1).detach()
        pred1 = F.softmax(x_hp, dim=1).detach()
        # breakpoint()
        filter_condition = ((pred0.max(dim=1)[1][self.idx_unlabel] == pred1.max(dim=1)[1][self.idx_unlabel])&
                            (pred0.max(dim=1)[0][self.idx_unlabel]*pred1.max(dim=1)[0][self.idx_unlabel] > self.conf.model['psedo_th']**2))
        idx_add = self.idx_unlabel[filter_condition]
        # TODO 判断条件改一下不同的？
        return idx_add.detach()
    
    
    def plotAuc(self,epochs,auc_epoch):
        import matplotlib.pyplot as plt
        import numpy as np
        plt.figure(figsize=(10, 7))
        plt.plot(range(epochs), auc_epoch, label='AUC')
        plt.xlabel('Epoch')
        plt.ylabel('AUC')
        plt.grid(True)
        plt.legend()
        plt.savefig("v1_auc_ada.png")

    # 计算AUC
    def calculate_auc(self,lce, y):
        from sklearn.metrics import roc_curve, auc
        lce= lce.detach().cpu().numpy()
        y=y.cpu().numpy()
        fpr, tpr, thresholds = roc_curve(y, lce, pos_label=0)
        return auc(fpr, tpr)
    

    def devide_labeled_nodes(self, label_noise, criterion, idx_train, edge_index, emb_lp, emb_hp, binary_y, weights_lp):
        criterion.eval()
        with torch.no_grad():
            z_lp= emb_lp[idx_train]
            z_hp= emb_hp[idx_train]
            # obtain predictors of embedding 
            x_lp, x_hp = criterion.forward(z_lp, z_hp)
            y_train_noise = label_noise[idx_train]
            # breakpoint()
            # dist 相当于localsim
            localsim = self.cal_localSim(edge_index, weights_lp, self.n_nodes)
            train_nodes_localsim = localsim[idx_train]
            loss_pick_lp = F.cross_entropy(x_lp, y_train_noise, reduction='none')
            loss_pick_hp = F.cross_entropy(x_hp, y_train_noise, reduction='none')
            # 这里做自适应
            loss_pick = train_nodes_localsim * loss_pick_lp + (1-train_nodes_localsim) * loss_pick_hp
            # loss_pick = loss_pick_lp + loss_pick_hp
            # loss_pick = loss_pick_hp 
            # loss_pick = loss_pick_lp

            # GMM
            prob = self.gmm_eval(loss_pick)   
            # print("========prob =======",prob)

        clean_mask = (prob > self.conf.model['gmm_th'])      
        
        selected_y = binary_y[clean_mask]
        print("=====selected clean numbers:{} |  percent:{}===".format(selected_y.shape[0], selected_y.sum()/selected_y.shape[0]))

        return clean_mask, prob, loss_pick
    
    def cal_localSim(self, edge_index, weights_lp, nnodes):
        src, tgt = edge_index
        localsim = scatter_mean(weights_lp, tgt, out=torch.zeros([nnodes]).to(edge_index.device))
        return localsim
        
    def gmm_eval(self, losses): 
        breakpoint()
        # self.plot_line(losses)  
        clean_dist = losses[self.binary_y].cpu().numpy()
        noisy_dist = losses[~self.binary_y].cpu().numpy()
        plot2distribution(clean_dist, noisy_dist, num_scales=80,width=1.2)
        
        losses = (losses-losses.min())/(losses.max()-losses.min()) 
        losses = losses.reshape(-1,1)
        gmm = GaussianMixture(n_components=2,max_iter=10,covariance_type='diag',tol=1e-2,reg_covar=5e-4)
        losses = losses.cpu()
        gmm.fit(losses)
        prob = gmm.predict_proba(losses)
        prob = prob[:,gmm.means_.argmin()] 
        return prob
    
    def plot_line(self,losses):
        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.kdeplot(losses.cpu().numpy(), bw_adjust=0.5, fill=True)
        plt.title("Empirical PDF of Training Loss")
        plt.xlabel("Loss Value")
        plt.ylabel("Density")
        plt.grid(True)
        plt.savefig("loss_line.png")

def plot2distribution(dist1, dist2 , num_scales = 15, width = 0.1) :
    import matplotlib.pyplot as plt

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
    plt.savefig("loss_line.png")





#  // "gmm_th": {
#   //   "_type": "choice",
#   //   "_value": [
#   //     0.7,
#   //     0.8,
#   //     0.9,
#   //     0.95
#   //   ]
#   // },


