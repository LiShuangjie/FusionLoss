import copy
import torch
from utils.functional import normalize
from utils.tools import get_npz_data
from torch_geometric.datasets import Planetoid, Amazon, Coauthor, WikiCS, WikipediaNetwork, WebKB, Actor, \
    AttributedGraphDataset, TUDataset, CitationFull, HeterophilousGraphDataset
from torch_geometric.utils import degree
import os
from .datasplit import get_split
from collections import defaultdict
import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch.nn.functional as F

class Dataset:
    '''
    Dataset Class.
    This class loads, preprocesses and splits various datasets.

    Parameters
    ----------
    data : str
        The name of dataset.
    feat_norm : bool
        Whether to normalize the features.
    verbose : bool
        Whether to print statistics.
    n_splits : int
        Number of data splits.
    path : str
        Path to save dataset files.
    '''

    def __init__(self, data, trail, feat_norm=False, adj_norm=False, verbose=True, path='./data/',
                 train_size=None, val_size=None, test_size=None,
                 train_percent=None, val_percent=None, test_percent=None,
                 train_examples_per_class=None, val_examples_per_class=None, test_examples_per_class=None,
                 add_self_loop=True, split_type='default', from_npz=False, device='cuda:0'):
        self.name = data
        self.path = path
        self.trail = trail
        self.device = torch.device(device)
        self.single_graph = True
        self.self_loop = add_self_loop
        self.split_type = split_type

        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size
        self.train_percent = train_percent
        self.val_percent = val_percent
        self.test_percent = test_percent
        self.train_examples_per_class = train_examples_per_class
        self.val_examples_per_class = val_examples_per_class
        self.test_examples_per_class = test_examples_per_class

        self.prepare_data(data, feat_norm, from_npz)
        self.feats = self.feats.to(torch.float)
        # if self.single_graph:
        self.split_data(verbose)
        # else:
        #   self.split_graphs(verbose)
        # self.homophily = get_homophily(self.labels, self.adj.to_dense(), type='edge', fill=None)
        if add_self_loop:
            self.adj = self.adj + torch.eye(self.adj.shape[0], device=self.adj.device).to_sparse()
        if adj_norm:
            self.adj = normalize(self.adj, add_loop=False)
        self.adj = self.adj.coalesce()

    def prepare_data(self, ds_name, feat_norm, from_npz):
        '''
        Function to Load various datasets.
        Homophilous datasets are loaded via pyg, while heterophilous datasets are loaded with `hetero_load`.
        The results are saved as `self.feats, self.adj, self.labels, self.train_masks, self.val_masks, self.test_masks`.
        Noth that `self.adj` is undirected and has no self loops.

        Parameters
        ----------
        ds_name : str
            The name of dataset.
        feat_norm : bool
            Whether to normalize the features.
        from_npz : bool
            Whether to load data from an existing npz file.

        '''

        if from_npz:
            adj, features, labels = get_npz_data(self.path + ds_name + '.npz', self_loop=self.self_loop)
            self.adj = adj.to(self.device).coalesce()
            self.feats = features.to(self.device)
            self.labels = torch.tensor(labels, dtype=torch.int64).to(self.device)
            self.n_nodes = self.feats.shape[0]
            self.dim_feats = self.feats.shape[1]
            self.n_edges = self.adj.indices().shape[1] / 2
            self.n_classes = labels.max() + 1
            if feat_norm:
                self.feats = normalize(self.feats, style='row')

        elif ds_name in ['cora', 'pubmed', 'citeseer', 'amazoncom', 'amazonpho', 'coauthorcs', 'coauthorph',
                         'blogcatalog', 'flickr', 'wikics', 'cornell', 'chameleon','texas', 'wisconsin', 'dblp', 'amazon-ratings',
                         'roman-empire']:
            self.data_raw = pyg_load_dataset(ds_name, path=self.path)
            self.g = self.data_raw[0]
            self.feats = self.g.x  # unnormalized
            if ds_name == 'flickr':
                self.feats = self.feats.to_dense()
            self.n_nodes = self.feats.shape[0]
            self.dim_feats = self.feats.shape[1]
            self.labels = self.g.y
            self.adj = torch.sparse_coo_tensor(self.g.edge_index, torch.ones(self.g.edge_index.shape[1]),
                                               [self.n_nodes, self.n_nodes])
            self.n_edges = self.g.num_edges / 2
            self.n_classes = self.data_raw.num_classes

            self.feats = self.feats.to(self.device)
            self.labels = self.labels.to(self.device)

            self.adj = self.adj.to(self.device)
            # normalize features
            if feat_norm:
                self.feats = normalize(self.feats, style='row')

        self.adj = self.adj.coalesce()
        row = self.adj.indices()[0]
        d = degree(row, self.n_nodes)
        self.ave_degree = float(torch.mean(d))

        print("""----Data statistics------'
                Name: %s
                #Nodes %d
                #Edges %d
                #Classes %d
                #Ave_degree %.2f""" %
              (self.name, self.n_nodes, self.n_edges, self.n_classes, self.ave_degree))

    def split_data(self, verbose=True):

        '''
        Function to conduct data splitting for various datasets.

        Parameters
        ----------
        verbose : bool
            Whether to print statistics.
        '''

        self.train_masks = None
        self.val_masks = None
        self.test_masks = None

        if self.split_type == 'default':
            if not hasattr(self.g, 'train_mask'):
                print('Split error, split type=' + self.split_type + '. Dataset ' + self.name + ' has no default split')
                exit(0)
            train_indices = torch.nonzero(self.g.train_mask, as_tuple=False).squeeze().numpy()
            val_indices = torch.nonzero(self.g.val_mask, as_tuple=False).squeeze().numpy()
            test_indices = torch.nonzero(self.g.test_mask, as_tuple=False).squeeze().numpy()
            train_type = 'default'
            val_type = 'default'
            test_type = 'default'
        elif self.split_type == 'percent':
            if self.train_size is not None:
                train_size = self.train_size
                train_type = 'specified'
            elif self.train_percent is not None:
                train_size = int(self.n_nodes * self.train_percent)
                train_type = str(self.train_percent * 100) + ' % of nodes'
            else:
                print('Split error: split type = percent. Train size and train percent were not configured')
                exit(0)

            if self.val_size is not None:
                val_size = self.val_size
                val_type = 'specified'
            elif self.val_percent is not None:
                val_size = int(self.n_nodes * self.val_percent)
                val_type = str(self.val_percent * 100) + ' % of nodes'
            else:
                print('Split error: split type = percent. Val size and Val percent were not configured')
                exit(0)

            if self.test_size is not None:
                test_size = self.test_size
                test_type = 'specified'
            elif self.test_percent is not None:
                test_size = int(self.n_nodes * self.test_percent)
                test_type = str(self.test_percent * 100) + ' % of nodes'
            else:
                test_size = None
                test_type = 'remaining'
            train_indices, val_indices, test_indices = get_split(self.labels.cpu().numpy(),
                                                                 train_size=train_size,
                                                                 val_size=val_size,
                                                                 test_size=test_size, )
        elif self.split_type == 'samples_per_class':
            train_size = None
            val_size = None
            test_size = None
            if self.train_examples_per_class is not None:
                train_examples_per_class = self.train_examples_per_class
                train_type = str(self.train_examples_per_class) + ' nodes per class'
            elif self.train_size is not None:
                train_examples_per_class = None
                train_size = self.train_size
                train_type = 'specified'
            else:
                print('Split error: split type = samples_per_class. Train size and train percent were not configured')
                exit(0)

            if self.val_examples_per_class is not None:
                val_examples_per_class = self.val_examples_per_class
                val_type = str(self.val_examples_per_class) + ' nodes per class'
            elif self.val_size is not None:
                val_examples_per_class = None
                val_size = self.val_size
                val_type = 'specified'
            else:
                print('Split error: split type = samples_per_class. Val size and val percent were not configured')
                exit(0)

            if self.test_examples_per_class is not None:
                test_examples_per_class = self.test_examples_per_class
                test_type = str(self.test_examples_per_class) + ' nodes per class'
            elif self.test_size is not None:
                test_examples_per_class = None
                test_size = self.test_size
                test_type = 'specified'
            else:
                test_examples_per_class = None
                test_size = None
                test_type = 'remaining'
            train_indices, val_indices, test_indices = get_split(self.labels.cpu().numpy(),
                                                                 train_examples_per_class=train_examples_per_class,
                                                                 val_examples_per_class=val_examples_per_class,
                                                                 test_examples_per_class=test_examples_per_class,
                                                                 train_size=train_size,
                                                                 val_size=val_size,
                                                                 test_size=test_size)
        else:
            print('Split error: split type ' + self.split_type + ' not implemented')
            exit(0)

        # if self.name in ['amazon-ratings', 'roman-empire']:

        #     # train_val_split={
        #     #             'train_mask':self.train_masks,
        #     #             'val_mask':self.val_masks,
        #     #             'test_mask':self.test_masks
        #     #         }
        #     # torch.save(train_val_split,"./splits/"+self.name+"_"+str(0)+".pth")
        #     train_val_split = torch.load("./splits/"+self.name+"_"+str(self.trail)+".pth")
        #     self.train_masks = train_val_split['train_mask']
        #     self.val_masks = train_val_split['val_mask']
        #     self.test_masks = train_val_split['test_mask']
        # else:
        #     self.train_masks = train_indices
        #     self.val_masks = val_indices
        #     self.test_masks = test_indices
        self.train_masks = train_indices
        self.val_masks = val_indices
        self.test_masks = test_indices
        if verbose:
            print("""----Split statistics------'
                #Train samples %d (%s)
                #Val samples %d (%s)
                #Test samples %d (%s)""" %
                  (len(self.train_masks), train_type,
                   len(self.val_masks), val_type,
                   len(self.test_masks), test_type))


class NewDataset:
    '''
    Dataset Class.
    This class loads, preprocesses and splits various datasets.

    Parameters
    ----------
    data : str
        The name of dataset.
    feat_norm : bool
        Whether to normalize the features.
    verbose : bool
        Whether to print statistics.
    n_splits : int
        Number of data splits.
    path : str
        Path to save dataset files.
    '''

    def __init__(self, data, split, feat_norm=False, adj_norm=False,
                 add_self_loop=True, device='cuda:0'):
        self.name = data
        self.device = torch.device(device)
        self.self_loop = add_self_loop

 
        self.load_data_new(data, split)
        self.feats = self.feats.to(torch.float)
            
        if add_self_loop:
            self.adj = self.adj + torch.eye(self.adj.shape[0], device=self.adj.device).to_sparse()
        if adj_norm:
            self.adj = normalize(self.adj, add_loop=False)
                    # normalize features
        if feat_norm:
            self.feats = normalize(self.feats, style='row')
        self.adj = self.adj.coalesce()

    def load_data_new(self, dataset_str, split):
        """
        Loads input data from gcn/data directory

        ind.dataset_str.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
        ind.dataset_str.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
        ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training instances
            (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
        ind.dataset_str.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
        ind.dataset_str.ty => the one-hot labels of the test instances as numpy.ndarray object;
        ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
        ind.dataset_str.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
            object;
        ind.dataset_str.test.index => the indices of test instances in graph, for the inductive setting as list object.

        All objects above must be saved using python pickle module.

        :param dataset_str: Dataset name
        :return: All data input files loaded (as well the training/test data).
        """
        
        if dataset_str in ['chameleon', 'cornell', 'film', 'squirrel', 'texas', 'wisconsin']:
            # print(os.path)
            graph_adjacency_list_file_path = os.path.join(
                './new_data', dataset_str, 'out1_graph_edges.txt')
            graph_node_features_and_labels_file_path = os.path.join('./new_data', dataset_str,
                                                                    f'out1_node_feature_label.txt')
            graph_dict = defaultdict(list)
            with open(graph_adjacency_list_file_path) as graph_adjacency_list_file:
                graph_adjacency_list_file.readline()
                for line in graph_adjacency_list_file:
                    line = line.rstrip().split('\t')
                    assert (len(line) == 2)
                    graph_dict[int(line[0])].append(int(line[1]))
                    graph_dict[int(line[1])].append(int(line[0]))

            # print(sorted(graph_dict))
            graph_dict_ordered = defaultdict(list)
            for key in sorted(graph_dict):
                graph_dict_ordered[key] = graph_dict[key]
                graph_dict_ordered[key].sort()

            adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph_dict_ordered))
            # adj = sp.csr_matrix(adj)

            graph_node_features_dict = {}
            graph_labels_dict = {}

            if dataset_str == 'film':
                with open(graph_node_features_and_labels_file_path) as graph_node_features_and_labels_file:
                    graph_node_features_and_labels_file.readline()
                    for line in graph_node_features_and_labels_file:
                        line = line.rstrip().split('\t')
                        assert (len(line) == 3)
                        assert (int(line[0]) not in graph_node_features_dict and int(
                            line[0]) not in graph_labels_dict)
                        feature_blank = np.zeros(932, dtype=np.uint8)
                        feature_blank[np.array(
                            line[1].split(','), dtype=np.uint16)] = 1
                        graph_node_features_dict[int(line[0])] = feature_blank
                        graph_labels_dict[int(line[0])] = int(line[2])
            else:
                with open(graph_node_features_and_labels_file_path) as graph_node_features_and_labels_file:
                    graph_node_features_and_labels_file.readline()
                    for line in graph_node_features_and_labels_file:
                        line = line.rstrip().split('\t')
                        assert (len(line) == 3)
                        assert (int(line[0]) not in graph_node_features_dict and int(
                            line[0]) not in graph_labels_dict)
                        graph_node_features_dict[int(line[0])] = np.array(
                            line[1].split(','), dtype=np.uint8)
                        graph_labels_dict[int(line[0])] = int(line[2])

            features_list = []
            for key in sorted(graph_node_features_dict):
                features_list.append(graph_node_features_dict[key])
            features = np.vstack(features_list)
            features = sp.csr_matrix(features)

            labels_list = []
            for key in sorted(graph_labels_dict):
                labels_list.append(graph_labels_dict[key])

            label_classes = max(labels_list) + 1
            labels = np.eye(label_classes)[labels_list]

            splits_file_path = 'splits/' + dataset_str + \
                '_split_0.6_0.2_' + str(split) + '.npz'

            with np.load(splits_file_path) as splits_file:
                train_mask = splits_file['train_mask']
                val_mask = splits_file['val_mask']
                test_mask = splits_file['test_mask']

            idx_train = np.where(train_mask == 1)[0]
            idx_val = np.where(val_mask == 1)[0]
            idx_test = np.where(test_mask == 1)[0]
        # adj = normalize(adj + sp.eye(adj.shape[0]))
        self.adj = self.sparse_mx_to_torch_sparse_tensor(adj)

        # features = normalize(features)
        self.feats = torch.tensor(np.array(features.todense()))
        # self.labels = torch.LongTensor(np.where(labels))[1]
        self.labels = torch.argmax(torch.tensor(labels), dim=1) 
        self.train_masks = torch.tensor(idx_train)
        self.val_masks = torch.tensor(idx_val)
        self.test_masks = torch.tensor(idx_test)
        self.n_nodes = self.feats.shape[0]
        self.dim_feats = self.feats.shape[1]
        self.n_classes = (self.labels.max() + 1).item()
        
        self.feats = self.feats.to(self.device)
        self.labels = self.labels.to(self.device)
        self.adj = self.adj.to(self.device)
    
    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)


class ArxivData:
    '''
    Dataset Class.
    This class loads, preprocesses and splits various datasets.

    Parameters
    ----------
    data : str
        The name of dataset.
    feat_norm : bool
        Whether to normalize the features.
    verbose : bool
        Whether to print statistics.
    n_splits : int
        Number of data splits.
    path : str
        Path to save dataset files.
    '''

    def __init__(self, data, feat_norm=False, adj_norm=False,
                 add_self_loop=True, device='cuda:0'):
        self.name = data
        self.device = torch.device(device)
        self.self_loop = add_self_loop

        self.load_arxiv(data)

        if feat_norm:
            # breakpoint()
            # self.feats = normalize(self.feats, style='row') 
            self.feats = F.normalize(self.feats, p=2, dim=-1)
        

    def load_arxiv(self, dataset_str):
        from ogb.nodeproppred import PygNodePropPredDataset
        import torch_geometric.utils as utils

        dataset = PygNodePropPredDataset(root="./Data", name = 'ogbn-arxiv')
        data = dataset[0]
        data.split_idx = dataset.get_idx_split()
        data.y = torch.squeeze(data.y)
        idx_train, idx_val, idx_test = data.split_idx['train'], data.split_idx['valid'], data.split_idx['test']
        # train_val_split={
        #             'train_mask':idx_train,
        #             'val_mask':idx_val,
        #             'test_mask':idx_test
        #         }
        # torch.save(train_val_split,"./splits/"+"ogbn-arxiv.pth")
        # 训练数据只有labeled_rate的labeled nodes
        features, labels = data.x, data.y.numpy()
        adj = utils.to_scipy_sparse_matrix(data.edge_index)
        # 构建无向图
        adj_h = self.process_adj(adj)
        # self.edge_index =  adj_h.coalesce().indices()
        self.n_classes = labels.max() + 1
        self.adj = adj_h.coalesce()
        # self.adj = self.sparse_mx_to_torch_sparse_tensor(adj)
        # self.feats = torch.tensor(np.array(features.todense()))
        self.feats = features
        self.labels = torch.tensor(labels)
        self.train_masks = idx_train
        self.val_masks = idx_val
        self.test_masks = idx_test
        self.n_nodes = self.feats.shape[0]
        self.dim_feats = self.feats.shape[1]
        
        self.feats = self.feats.to(self.device)
        self.labels = self.labels.to(self.device)
        self.adj = self.adj.to(self.device)
        # self.edge_index = self.edge_index.to(self.device)
    # arxiv
    def process_adj(self,adj_h):
        # adj_h = adj
        adj_h.setdiag(1)
        adj_h = adj_h + adj_h.T.multiply(adj_h.T > adj_h) - adj_h.multiply(adj_h.T > adj_h)
        # adj_h = self.normalize_adj(adj_h) 
        adj_h = self.sparse_mx_to_torch_sparse_tensor_arxiv(adj_h)
        # adj = self.sparse_mx_to_torch_sparse_tensor_arxiv(adj)
        return  adj_h
    
    def sparse_mx_to_torch_sparse_tensor_arxiv(self,sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)

    def normalize_adj(self,mx):
        """Row-column-normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1/2).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx).dot(r_mat_inv)
        return mx

DATAPATH = './Data/linkx_data/'
class NCDataset(object):
    def __init__(self, name, root=f'{DATAPATH}'):
        """
        based off of ogb NodePropPredDataset
        https://github.com/snap-stanford/ogb/blob/master/ogb/nodeproppred/dataset.py
        Gives torch tensors instead of numpy arrays
            - name (str): name of the dataset
            - root (str): root directory to store the dataset folder
            - meta_dict: dictionary that stores all the meta-information about data. Default is None, 
                    but when something is passed, it uses its information. Useful for debugging for external contributers.
        
        Usage after construction: 
        
        split_idx = dataset.get_idx_split()
        train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
        graph, label = dataset[0]
        
        Where the graph is a dictionary of the following form: 
        dataset.graph = {'edge_index': edge_index,
                         'edge_feat': None,
                         'node_feat': node_feat,
                         'num_nodes': num_nodes}
        For additional documentation, see OGB Library-Agnostic Loader https://ogb.stanford.edu/docs/nodeprop/
        """

        self.name = name  # original name, e.g., ogbn-proteins
        self.graph = {}
        self.label = None

    def get_idx_split(self, split_type='random', train_prop=.5, valid_prop=.25):
        """
        train_prop: The proportion of dataset for train split. Between 0 and 1.
        valid_prop: The proportion of dataset for validation split. Between 0 and 1.
        """

        if split_type == 'random':
            ignore_negative = False if self.name == 'ogbn-proteins' else True
            train_idx, valid_idx, test_idx = rand_train_test_idx(
                self.label, train_prop=train_prop, valid_prop=valid_prop, ignore_negative=ignore_negative)
            split_idx = {'train': train_idx,
                         'valid': valid_idx,
                         'test': test_idx}
        return split_idx

    def __getitem__(self, idx):
        assert idx == 0, 'This dataset has only one graph'
        return self.graph, self.label

    def __len__(self):
        return 1

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, len(self))

class DS:
    def __init__(self, x, y, edge_index):
        self.x = x
        self.y = y
        self.edge_index = edge_index

class ArxivYearData:
    '''
    Dataset Class.
    This class loads, preprocesses and splits various datasets.

    Parameters
    ----------
    data : str
        The name of dataset.
    feat_norm : bool
        Whether to normalize the features.
    verbose : bool
        Whether to print statistics.
    n_splits : int
        Number of data splits.
    path : str
        Path to save dataset files.
    '''

    def __init__(self, data, split, feat_norm=False, adj_norm=False,
                 add_self_loop=True, device='cuda:0'):
        self.name = data
        self.device = torch.device(device)
        self.self_loop = add_self_loop
        self.split = split

        self.load_arxiv_year()
        # if feat_norm:
        #     self.feats = F.normalize(self.feats, p=2, dim=-1)
        
            
    def load_arxiv_year_dataset(self, nclass=5):
        from ogb.nodeproppred import NodePropPredDataset

        filename = 'arxiv-year'
        dataset = NCDataset(filename)
        ogb_dataset = NodePropPredDataset(root=DATAPATH, name='ogbn-arxiv')
        dataset.graph = ogb_dataset.graph
        dataset.graph['edge_index'] = torch.as_tensor(dataset.graph['edge_index'])
        dataset.graph['node_feat'] = torch.as_tensor(dataset.graph['node_feat'])

        label = self.even_quantile_labels(
            dataset.graph['node_year'].flatten(), nclass, verbose=False)
        dataset.label = torch.as_tensor(label).reshape(-1, 1)
        return dataset

    def even_quantile_labels(self, vals, nclasses, verbose=True):
        """ partitions vals into nclasses by a quantile based split,
        where the first class is less than the 1/nclasses quantile,
        second class is less than the 2/nclasses quantile, and so on
        
        vals is np array
        returns an np array of int class labels
        """
        label = -1 * np.ones(vals.shape[0], dtype=np.int32)
        interval_lst = []
        lower = -np.inf
        for k in range(nclasses - 1):
            upper = np.nanquantile(vals, (k + 1) / nclasses)
            interval_lst.append((lower, upper))
            inds = (vals >= lower) * (vals < upper)
            label[inds] = k
            lower = upper
        label[vals >= lower] = nclasses - 1
        interval_lst.append((lower, np.inf))
        if verbose:
            print('Class Label Intervals:')
            for class_idx, interval in enumerate(interval_lst):
                print(f'Class {class_idx}: [{interval[0]}, {interval[1]})]')
        return label

    def load_arxiv_year(self):
        import torch_geometric.utils as utils

        dataset = self.load_arxiv_year_dataset()
        data = dataset[0]
        x = data[0]['node_feat']
        # y = data[1].squeeze().type(torch.long)
        y = data[1].squeeze().numpy()

        edge_index = data[0]['edge_index']
        
        data = DS(
            x=x, y=y, 
            edge_index=edge_index, 
        )
        train_val_split = torch.load("./splits/"+'arxiv_year_50_25_'+str(self.split)+".pth")
        x = F.normalize(data.x, p=2, dim=-1)
        adj = utils.to_scipy_sparse_matrix(data.edge_index)
        # 构建无向图
        adj = self.process_adj(adj)
        self.adj = adj.coalesce()
        self.n_classes = y.max() + 1
        # breakpoint()
        self.feats = x
        self.labels = torch.tensor(y)
        self.train_masks = np.nonzero(train_val_split['train_mask']).squeeze(1) 
        self.val_masks = np.nonzero(train_val_split['val_mask']).squeeze(1) 
        self.test_masks = np.nonzero(train_val_split['test_mask']).squeeze(1) 
        # todo mask转indices
        self.n_nodes = self.feats.shape[0]
        self.dim_feats = self.feats.shape[1]
        
        self.feats = self.feats.to(self.device)
        self.labels = self.labels.type(torch.int64).to(self.device)
        self.adj = self.adj.to(self.device)
        self.edge_index = data.edge_index.to(self.device)

    def process_adj(self,adj):
        # # adj_h = adj.clone()
        # adj_h = copy.deepcopy(adj)
        # adj_h.setdiag(1)
        
        # adj_h = adj_h + adj_h.T.multiply(adj_h.T > adj_h) - adj_h.multiply(adj_h.T > adj_h)
        # adj_h = self.normalize_adj(adj_h)
        # adj_h = self.sparse_mx_to_torch_sparse_tensor_arxiv(adj_h)
        adj = self.sparse_mx_to_torch_sparse_tensor_arxiv(adj)
        return adj
    
    def sparse_mx_to_torch_sparse_tensor_arxiv(self,sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)

    def normalize_adj(self,mx):
        """Row-column-normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1/2).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx).dot(r_mat_inv)
        return mx
  
    
def pyg_load_dataset(name, path='./data/'):
    dic = {'cora': 'Cora',
           'citeseer': 'CiteSeer',
           'pubmed': 'PubMed',
           'amazoncom': 'Computers',
           'amazonpho': 'Photo',
           'coauthorcs': 'CS',
           'coauthorph': 'Physics',
           'wikics': 'WikiCS',
           'chameleon': 'Chameleon',
           'squirrel': 'Squirrel',
           'cornell': 'Cornell',
           'texas': 'Texas',
           'wisconsin': 'Wisconsin',
           'actor': 'Actor',
           'blogcatalog': 'blogcatalog',
           'flickr': 'flickr',
           'amazon-ratings': 'Amazon-ratings',
           'roman-empire': 'Roman-empire'}
    if name in dic.keys():
        name = dic[name]
    else:
        name = name

    if name in ["Cora", "CiteSeer", "PubMed"]:
        dataset = Planetoid(root=path, name=name)
    elif name in ["Computers", "Photo"]:
        dataset = Amazon(root=path, name=name)
    elif name in ["CS", "Physics"]:
        dataset = Coauthor(root=path, name=name)
    elif name in ['WikiCS']:
        dataset = WikiCS(root=os.path.join(path, name))
    elif name in ['Chameleon', 'Squirrel', 'Crocodile']:
        dataset = WikipediaNetwork(root=path, name=name)
    elif name in ['Cornell', 'Texas', 'Wisconsin']:
        dataset = WebKB(root=path, name=name)
    elif name == 'Actor':
        dataset = Actor(root=os.path.join(path, name))
    elif name in ['blogcatalog', 'flickr']:
        dataset = AttributedGraphDataset(root=path, name=name)
    elif name in ['Amazon-ratings', 'Roman-empire']:
        dataset = HeterophilousGraphDataset(root=path, name=name)
    elif name in ['dblp']:
        dataset = CitationFull(root=path, name=name)
    else:
        dataset = TUDataset(root=path, name=name)
    return dataset
