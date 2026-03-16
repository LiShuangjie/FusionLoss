import argparse
import nni
import torch
from predictor.AdaptLoss_Predictor_auc import plotAuc
from utils.dataloader import ArxivData, ArxivYearData, Dataset, NewDataset
from utils.tools import load_conf, setup_seed
from utils.labelnoise import label_process
import numpy as np


from predictor.AdaptLoss_Predictor import adaptloss_Predictor
# from predictor.AdaptLoss_Predictor_arxiv import adaptloss_Predictor
import random

def merge_params(model_conf):
    turner_params = nni.get_next_parameter()
    print(turner_params)
    for item in turner_params.keys():
        print(item)
        if item in ['lr', 'weight_decay']:
            model_conf.training[item] = turner_params[item]
        else:
            model_conf.model[item] = turner_params[item]
    print(model_conf)
    return model_conf


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str,
                    default='cora',
                    choices=['cora', 'citeseer', 'pubmed', 'amazoncom', 'amazonpho', 'squirrel',
                             'dblp', 'blogcatalog', 'flickr','cornell', 'wisconsin','texas','chameleon', 'film',
                             'amazon-ratings', 'roman-empire','arxiv', 'arxiv-year'],
                    help='Select dataset')
parser.add_argument('--method', type=str,
                    default='mlp',
                    choices=['gcn', 'gin', 'smodel', 'jocor', 'coteaching',
                             'apl', 'sce', 'forward', 'backward', 'lcat', 'mlp',
                             'nrgnn', 'rtgnn', 'cp', 'unionnet', 'cgnn',
                             'crgnn', 'clnode', 'rncgln', 'pignn','cfgd', 'dgnn', 'r2lp', 'heterln', 'het', 'adaptloss'],
                    help="Select methods")
parser.add_argument('--noise_type', type=str,
                    default='uniform',
                    choices=['clean', 'uniform', 'pair', 'random'], help='Type of label noise')
parser.add_argument('--noise_rate', type=float,
                    default='0',
                    help='Label noise rate')
parser.add_argument('--device', type=str,
                    default='cuda:1',
                    help='Device')
parser.add_argument('--seed', type=int,
                    default=3000,
                    help="Random Seed")
                    
args = parser.parse_args()


if __name__ == '__main__':
    print(args)
    data_path = './data/'
    data_conf = load_conf('./config/_dataset/' + args.dataset + '.yaml')
    results = []
    auclist = []

    for trail in range(1):
        if nni.get_trial_id() == "STANDALONE":
            setup_seed(3000+trail)
        if args.dataset in ['cornell', 'wisconsin','chameleon', 'texas', 'film', 'squirrel']:
            data = NewDataset(args.dataset, trail,
                        feat_norm=data_conf.norm['feat_norm'], adj_norm=data_conf.norm['adj_norm'],
                        add_self_loop=data_conf.modify['add_self_loop'], device=args.device)
        elif args.dataset in ['arxiv']:
            data = ArxivData(args.dataset,
                        feat_norm=data_conf.norm['feat_norm'], adj_norm=data_conf.norm['adj_norm'],
                        add_self_loop=data_conf.modify['add_self_loop'], device=args.device)
        elif args.dataset in ['arxiv-year']:
            data = ArxivYearData(args.dataset,trail,
                        feat_norm=data_conf.norm['feat_norm'], adj_norm=data_conf.norm['adj_norm'],
                        add_self_loop=data_conf.modify['add_self_loop'], device=args.device)
        else:
            data = Dataset(args.dataset, trail, path=data_path,
                        feat_norm=data_conf.norm['feat_norm'], adj_norm=data_conf.norm['adj_norm'],
                        train_size=data_conf.split['train_size'],
                        val_size=data_conf.split['val_size'],
                        test_size=data_conf.split['test_size'],
                        train_percent=data_conf.split['train_percent'],
                        val_percent=data_conf.split['val_percent'],
                        test_percent=data_conf.split['test_percent'],
                        train_examples_per_class=data_conf.split['train_examples_per_class'],
                        val_examples_per_class=data_conf.split['val_examples_per_class'],
                        test_examples_per_class=data_conf.split['test_examples_per_class'],
                        add_self_loop=data_conf.modify['add_self_loop'],
                        from_npz=data_conf.modify['from_npz_largest_component'],
                        device=args.device,
                        split_type=data_conf.split['split_type'])
            

        model_conf = load_conf(None, args.method, data.name)
        if nni.get_trial_id() != "STANDALONE":
            model_conf = merge_params(model_conf)
        if  args.dataset in ['citeseer','amazonpho','roman-empire','amazon-ratings','arxiv','arxiv-year', 'pubmed']:
 
            data.noisy_label = torch.load("./noise/"+args.dataset+"/"+args.noise_type+str(args.noise_rate)+".pth", weights_only=False)
            data.noisy_label = data.noisy_label.type(torch.int64).to(args.device)
        else:
            data.noisy_label, modified_mask = label_process(labels=data.labels, n_classes=data.n_classes,
                                                            noise_type=args.noise_type, noise_rate=args.noise_rate,
                                                            random_seed=args.seed+trail, debug=True)
 
        model_conf.model['n_feat'] = data.dim_feats
        model_conf.model['n_classes'] = data.n_classes
        model_conf.training['debug'] = True
        predictor = eval(args.method + '_Predictor')(model_conf, data, args.device)
        # 将idx_train 分为干净idx和噪声idx 干净的为1 噪声的为0 转为一个二类问题
        binary_clean_mask = data.noisy_label[data.train_masks] == data.labels[data.train_masks]
        if args.method == 'adaptloss':
            result = predictor.train(binary_clean_mask, args.noise_rate)

        else:
            result = predictor.train()

        results.append(result['test'])
        print("results",results)




    test_mean = np.mean(results)*100
    test_std = np.std(results)*100
    print("test_mean {:.2f} \pm {:.2f}".format(test_mean.item(), test_std))


    if nni.get_trial_id() != "STANDALONE":
        # nni.report_final_result(float(result['test']))
        nni.report_final_result(float(test_mean))

