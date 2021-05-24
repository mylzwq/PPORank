from __future__ import print_function
import torch
import argparse
import torch.nn as nn
import utils
from utils import AverageMeter, TqdmLoggingHandler
from preprocess.preprocess_fts_cl_drug import load_fts_mat_with_inds, load_PQ_WP_embs
import numpy as np
import pandas as pd
import os
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
import logging
import datetime
from torch.nn.parameter import Parameter
from set_log import *
from arguments import *
from os import path
from pathlib import Path

from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler
from pygments.lexers import r

import torch
import numpy as np
import os
from functools import partial
import pandas as pd
from collections import Counter, defaultdict
import time
from torch.utils.data import Dataset, DataLoader


class NoiseCellDrugDataset(Dataset):

    def __init__(self, data_dir, Xtrain_fn, Ytrain_fn, sigma, mu, ess_genes_fn=""):
        super().__init__()
        self.data_dir = data_dir
        self.Xtrain_fn = Xtrain_fn
        self.Ytrain_fn = Ytrain_fn
        self.Xtrain_origin = pd.read_csv(os.path.join(data_dir, Xtrain_fn), index_col=0)
        self.Ytrain = np.array(pd.read_csv(os.path.join(data_dir, Ytrain_fn), index_col=0))
        self.sigma = sigma
        self.mu = mu
        self.dtype = torch.DoubleTensor
        self.N, self.P = self.Xtrain_origin.shape
        self.M = self.Ytrain.shape[1]
        if ess_genes_fn:
            ess_genes = pd.read_csv(ess_genes_fn, index_col=0, header=0, names=["genes"])
            common_genes = list(set(self.Xtrain_origin.columns) & set(ess_genes['genes']))
            self.Xtrain_origin = self.Xtrain_origin[common_genes]
        self.Xtrain_origin = np.array(self.Xtrain_origin)
        self.Xmean = np.mean(self.Xtrain_origin, axis=0)  # (1610,)
        self.Xstd = np.std(self.Xtrain_origin, axis=0)  # (1610,)

    def get_drug_fts(self):
        train_drug_inds = np.arange(self.M).reshape(1, self.M).reshape(1, self.M, 1)
        return train_drug_inds

    def __len__(self):
        return len(self.Xtrain_origin)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        Xsample = self.Xtrain_origin[idx].reshape(1, -1)  # (1,1610) array
        Xnoise_sample = self.get_noise_samples(Xsample).reshape(1, -1)  # (1,1610)
        drug_ind = self.get_drug_fts()  # (1,265,1)
        cell_fts = np.repeat(Xnoise_sample[:, np.newaxis, :], self.M, axis=1)  # (1,265,1610)
        cell_fts = (cell_fts-self.Xmean)/self.Xstd
        train_sample = np.concatenate((cell_fts, drug_ind), axis=2).squeeze()  # (M,P+1)
        true_scores = self.Ytrain[idx]  # (M,)
        train_sample = torch.from_numpy(train_sample)  # (M,P+1)

        true_scores = torch.from_numpy(true_scores)

        return train_sample, true_scores

    def get_noise_samples(self, data):
        u = np.random.uniform()
        if u > 0.5:
            return data + np.random.randn(*(data.shape)) * self.sigma + self.mu
        else:
            return data


def read_data(args):
    if args.analysis == "FULL" and args.prop < 1.0:
        Xtrain, Xtest, Ytrain, Ytest = utils.read_PROP(args.Data, args.prop, "CV", int(args.fold[-1]))

    elif args.analysis == "FULL":
        Xtrain, Xtest, Ytrain, Ytest = utils.read_FULL(args.Data, "CV", int(args.fold[-1]))

    if args.ess_genes_fn:
        ess_genes = pd.read_csv(args.ess_genes_fn, index_col=0, header=0, names=["genes"])
        common_genes = list(set(Xtrain.columns) & set(ess_genes['genes']))
        Xtrain = Xtrain[common_genes]
        Xtest = Xtest[common_genes]

    return Xtrain, Xtest, Ytrain, Ytest


def prepare_loader(data_dir, args, f=None, pretrain=False):

    # Xtrain [N,P],Ytrain[N,M],Ytrain includes Na,
    f = f if f else args.f
    Xtrain, Xtest, Ytrain, Ytest = read_data(args)

    Xtrain, Xtest, Ytrain, Ytest = np.array(Xtrain), np.array(Xtest), np.array(Ytrain), np.array(Ytest)

    # Xtrain and Ytrain are all
    WP, drug_embs = load_PQ_WP_embs(data_dir, f, args.pretrain)  # only drugs_emb in tensor

    ###
    xscaler = StandardScaler()
    Xtrain = xscaler.fit_transform(Xtrain)
    Xtest = xscaler.transform(Xtest)
    ###
    # WP [P,f] f is project dim
    N = Xtrain.shape[0]
    M = Ytrain.shape[1]
    P = Xtrain.shape[1]

    cell_fts_dim = Xtrain.shape[1]  # 985 or 1610
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dtype = torch.DoubleTensor
    train_drug_inds = np.arange(M).reshape(1, M).repeat(N, axis=0).reshape(N, M, 1)
    train_cell_fts = np.repeat(Xtrain[:, np.newaxis, :], M, axis=1)  # (788,223,985)
    train_input = np.concatenate((train_cell_fts, train_drug_inds), axis=2)  # train_input[i] shape is (265,1611)
    train_input = torch.from_numpy(train_input)  # (N,M,P+1)

    # Ytrain shape is (N,M), train_true_scores[i] shape is (265,)
    train_true_scores = torch.from_numpy(Ytrain)
    cell_mean_score = np.nanmean(Ytrain, axis=1)  # (N,)
    drug_mean_score = np.nanmean(Ytrain, axis=0)  # (M,)
    overall_mean_score = np.nanmean(Ytrain)

    train_dataset = TensorDataset(train_input, train_true_scores)

    test_true_scores = torch.from_numpy(Ytest)
    N1 = Xtest.shape[0]
    M1 = Ytest.shape[1]
    test_drug_inds = np.arange(M1).reshape(1, M1).repeat(N1, axis=0).reshape(N1, M1, 1)
    test_cell_fts = np.repeat(Xtest[:, np.newaxis, :], M1, axis=1)  # (197,223,985)

    test_input = np.concatenate((test_cell_fts, test_drug_inds), axis=2)
    test_input = torch.from_numpy(test_input)
    test_dataset = TensorDataset(test_input, test_true_scores)
    # test_loader=DataLoader(test_dataset,batch_size=args.batch_size,shuffle=False)

    return N, M, P, WP, drug_embs, train_dataset, test_dataset, train_input, test_input, Ytest, \
        cell_mean_score, drug_mean_score, overall_mean_score


def prepare_loader_simu(data_dir, args, f=None):

    Xtrain = np.asarray(pd.read_csv(data_dir+"/XtrainDf.csv", index_col=0))
    Xtest = np.asarray(pd.read_csv(data_dir+"/XtestDf.csv", index_col=0))
    if args.analysis == 'sparse':
        Ytrain_fn = 'Ytrain_' + args.analysis+"_miss{}".format(args.miss_rate)+'df.csv'
        Ytest_fn = 'Ytest_' + args.analysis+"_miss{}".format(args.miss_rate)+'df.csv'
        Ynoise_train_fn = 'Ynoise_train_' + args.analysis + 'df.csv'
        Ynoise_test_fn = 'Ynoise_test_' + args.analysis+'df.csv'
    elif args.analysis == 'noise':
        Ytrain_fn = 'Ytrain_' + args.analysis+'_df.csv'
        Ytest_fn = 'Ytest_' + args.analysis+'_df.csv'
        Ynoise_train_fn = Ytrain_fn
        Ynoise_test_fn = Ytest_fn

    Ytrain = np.asarray(pd.read_csv(os.path.join(data_dir, Ytrain_fn), index_col=0))
    Ytest = np.asarray(pd.read_csv(os.path.join(data_dir, Ytest_fn), index_col=0))
    Ynoise_train = np.asarray(pd.read_csv(os.path.join(data_dir, Ynoise_train_fn), index_col=0))
    Ynoise_test = np.asarray(pd.read_csv(os.path.join(data_dir, Ynoise_test_fn), index_col=0))

    N = Xtrain.shape[0]
    M = Ytrain.shape[1]
    P = Xtrain.shape[1]
    f = f if f else args.f

    WP_dir = os.path.join(data_dir, '{}Dim'.format(f))
    if not os.path.exists(WP_dir):
        os.mkdir((WP_dir))

    drug_embs_fn = os.path.join(WP_dir, 'Qmatrix.csv')

    if os.path.exists(drug_embs_fn):
        drug_embs = np.asarray(pd.read_csv(drug_embs_fn, index_col=0))
        drug_embs = torch.from_numpy(drug_embs)
    else:
        drug_embs = None

    cell_fts_dim = Xtrain.shape[1]

    if args.normalize_y:

        if args.scale == "minmax":
            # range to [0,1]
            scaler = MinMaxScaler()
            Ytrain = scaler.fit_transform(Ytrain)
        elif args.scale == "normalize":
            # N(0,1)
            scaler = StandardScaler()
            Ytrain = scaler.fit_transform(Ytrain)
        elif args.scale == "maxabs":
            # [-1,1]
            scaler = MaxAbsScaler()
            Ytrain = scaler.fit_transform(Ytrain)

    dtype = torch.DoubleTensor

    WP_fn = os.path.join(WP_dir, "WPmatrix.csv")

    if os.path.exists(WP_fn):
        WP = torch.from_numpy(np.asarray(pd.read_csv(WP_fn, index_col=0)))
    else:
        # WP = torch.from_numpy(np.identity(cell_fts_dim)).type(dtype)
        # WP = torch.zeros(N, args.f)
        WP = None

    train_drug_inds = np.arange(M).reshape(1, M).repeat(N, axis=0).reshape(N, M, 1)
    train_cell_fts = np.repeat(Xtrain[:, np.newaxis, :], M, axis=1)
    train_input = np.concatenate((train_cell_fts, train_drug_inds), axis=2)
    train_input = torch.from_numpy(train_input)

    train_true_scores = torch.from_numpy(Ytrain)
    train_dataset = TensorDataset(train_input, train_true_scores)
    #train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    if args.normalize_y:
        Ytest = scaler.transform(Ytest)

    test_true_scores = torch.from_numpy(Ytest).type(dtype)
    N1 = Xtest.shape[0]
    M1 = Ytest.shape[1]
    test_drug_inds = np.arange(M1).reshape(1, M1).repeat(N1, axis=0).reshape(N1, M1, 1)
    test_cell_fts = np.repeat(Xtest[:, np.newaxis, :], M1, axis=1)  # (197,223,985)

    test_input = np.concatenate((test_cell_fts, test_drug_inds), axis=2)
    test_input = torch.from_numpy(test_input)
    #test_noise_scores = torch.from_numpy(Ynoise_test).type(dtype)
    test_dataset = TensorDataset(test_input, test_true_scores)
    # test_loader=DataLoader(test_dataset,batch_size=args.batch_size,shuffle=False)

    return N, M, P, WP, drug_embs, train_dataset, test_dataset, train_input, test_input, Ytest, Ynoise_test


def gene_drug_emb(M_drug, clusters, a=0.5, b=1.0):
    # generate cell-line latent fts matrix
    V = np.zeros((M_drug, clusters))
    ind = 0
    mini_size = M_drug//clusters
    for i in range(clusters):
        V[ind:ind+mini_size, i] = (b-a)*np.random.random_sample(mini_size) + a
        ind += mini_size
    noise_mat = np.random.randn(M_drug, clusters)
    V = V + 0.5*noise_mat
    #V_df = pd.DataFrame(V,index=np.arange(V.shape[0]))
    return V


def load_data_simulator(data_dir,
                        cell_num,
                        drug_num,
                        cell_dim,
                        drug_dim,
                        beta=None,
                        training='train',
                        random_beta=False,
                        scenario="Linear"):

    if training == 'test':
        cell_num = int(cell_num/5)
    file_name = '{}/{}/Simu_{}_{}cells_{}drugs_{}celldim_{}drugdim.npz'.format(
        data_dir, training, scenario, cell_num, drug_num, cell_dim, drug_dim)
    if os.path.isfile(file_name):
        data = np.load(file_name)
        feature_mat = data['X']
        y_mat = data['Y']
        y_norm_mat = data['Ynorm']

    N = feature_mat.shape[0]
    M = feature_mat.shape[1]
    P = feature_mat.shape[2]

    feature_mat = feature_mat.reshape(N, 1, M, P)
    dtype = torch.FloatTensor
    input = torch.from_numpy(feature_mat).type(dtype)
    output = torch.from_numpy(y_mat).type(dtype)
    output_norm = torch.from_numpy(y_norm_mat).type(dtype)

    input = torch.tensor(input, requires_grad=False)
    output = torch.tensor(output, requires_grad=False)
    output_norm = torch.tensor(output_norm, requires_grad=False)

    return input, output, output_norm

# CCLE_all_file = "./RL/RL_GDSC/CCLE/CCLE_dose_response_scores.tsv"
# CCLE_all = pd.read_csv(CCLE_all_file, sep='\t')
# pd.unique((CCLE_all['CCLE Cell Line Name']))

# if __name__ == "__main__":
#     prepare_loader_simu('./RL/RL_GDSC/SimuData/linear/CV/noise/Fold0')
