import pandas as pd
import numpy as np
import sys
import os
import time
import pickle
import random
from scipy import stats
import Reward_utils
import torch
import logging
import utils

import argparse
logger = logging.getLogger(__name__)


###########
# the GDSC_data/input is for whole ss_file but the 10Dim 40Dim is the folder for corresponding feature matrix for cell
# and drugs

###########
def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess data")
    parser.add_argument(
        "--ss_name", default="./GDSC_data/input/gdsc_all_abs_ic50_bayesian_sigmoid.csv",
        help="GDSC IC50 response matrix")  # contains a matrix of IC50s where rows are cell lines and columns are drugs
    parser.add_argument("--ss_test_name", default="./GDSC_data/input/gdsc_all_abs_ic50_bayesian_sigmoid.csv",
                        help="GDSC IC50 response matrix for test")
    parser.add_argument("--cl_feature_fname", default="./GDSC_data/input/gdsc_cellline_pcor_ess_genes.csv",
                        help="cell-line features matrix")
    parser.add_argument("--drug_list_fname", default="./GDSC_data/input/gdsc_drugMedianGE0.txt")
    parser.add_argument("--out_dir", default="GDSC_data/output")
    parser.add_argument("--f", type=int, default=10,
                        help="dimension in the projected space")  # Number of dimensions
    parser.add_argument("--iters", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data_dir", default="GDSC_data")
    return parser.parse_args()


def ss_file_save():
    args = parse_args()
    ss_name = args.ss_name
    ss_test_name = args.ss_test_name
    cl_feature_fname = args.cl_feature_fname
    drug_list_fname = args.drug_list_fname

    selected_drugs = list(pd.read_csv(drug_list_fname, header=None)[0].values.astype(str))
    # contains a feature matrix where rows are both testing and training cell lines and columns are features
    cl_features_df = pd.read_csv(cl_feature_fname, index_col=0)

    # Training Data
    ss_df = pd.read_csv(ss_name, index_col=0)  # cellline_drug response matrix, n*m, S matrix
    ss_df.index = ss_df.index.astype(str)  # cell_line names

    # Convert IC50 to sensitivity score, -log(IC50)
    ss_df *= -1

    drug_list = list(ss_df.columns)
    drug_list = [d for d in drug_list if d in selected_drugs]  # 223 drugs

    cl_features_df.index = cl_features_df.index.astype(str)  # (1014,1014)
    cl_list = list(cl_features_df.index.astype(str))

    # select only predictable cell lines and drugs
    # S matrix, the cellline_drug_matrix
    ss_df = ss_df[ss_df.index.isin(cl_list)][drug_list]

    # Test Data
    ss_test_df = pd.read_csv(ss_test_name, index_col=0)
    ss_test_df.index = ss_test_df.index.astype(str)
    ss_test_df = ss_test_df[ss_test_df.index.isin(cl_list)][ss_df.columns]

    #######################################
    # Remove cell lines with no drug info #
    #######################################
    ss_df = ss_df.dropna(how="all")
    ss_test_df = ss_test_df.dropna(how="all")

    cl_features_df = cl_features_df[ss_df.index]

    # Save both train and test data
    # ss_df:(985,223), cl_features_df:(1014,985)
    np.savez_compressed("./GDSC_data/input/GDSC_proprecessed.npz",
                        ss_df=ss_df, ss_test_df=ss_test_df, cl_features_df=cl_features_df, cl_list=cl_list)
    return ss_df, ss_test_df, cl_features_df, cl_list


def Response_decompose(ss_df, cl_features_df, iters, lr, f, out_dir):

    current_err = np.inf
    cell_lines = list(ss_df.index)
    N = len(cell_lines)
    Drugs = list(ss_df.columns)
    M = len(Drugs)

    R = np.matrix(ss_df)
    n_K = np.sum(~np.isnan(R))  # all experiments
    X = np.matrix(cl_features_df.loc[cell_lines])  # cell-line features
    xdim = X.shape[1]  # num of X features
    Y = np.matrix(np.identity(M))  # Use identity matrix as default drug features (aka. learn directly to q_i)
    ydim = Y.shape[1]

    prng = np.random.RandomState(0)
    WP = (np.matrix(prng.rand(xdim, f))-0.5) / 10.
    prng = np.random.RandomState(0)
    WQ = (np.matrix(prng.rand(ydim, f))-0.5) / 10.

    WE = np.matrix(np.ones(R.shape))
    mu = np.nanmean(R)
    b_p = np.zeros(N)
    b_q = np.zeros(M)
    err_list = []

    for epoch in range(iters):
        old_WP = WP
        old_WQ = WQ
        old_mu = mu
        old_b_p = b_p
        old_b_q = b_q

        pred = (((mu + (Y * WQ * WP.T * X.T).T) + b_q).T + b_p).T

        for u in range(N):
            err = np.nansum(np.multiply(R[u, :]-pred[u, :], WE[u, :]))/n_K
            b_p[u] += lr*err

        for i in range(M):
            err = np.nansum(np.multiply(R[:, i] - pred[:, i], WE[:, i])) / n_K
            b_q[i] += lr*err

        # Update WP
        temp = np.matrix(np.zeros(WP.shape))
        for i in range(M):
            q_i = Y[i, :] * WQ
            err_per_i = (np.multiply(R[:, i] - pred[:, i], WE[:, i])).T
            err_per_i[np.isnan(err_per_i)] = 0
            temp += (q_i.T * np.sum(np.multiply(err_per_i.T, X), axis=0)).T
        temp = temp/n_K
        WP += lr*temp

        # Update WQ
        temp = np.matrix(np.zeros(WQ.shape))
        for u in range(N):
            p_u = X[u, :] * WP
            err_per_u = np.multiply(R[u, :] - pred[u, :], WE[u, :])
            err_per_u[np.isnan(err_per_u)] = 0
            temp += (p_u.T * np.sum(np.multiply(err_per_u.T, Y), axis=0)).T
        temp = temp / n_K
        WQ += lr * temp
        new_err = np.sqrt(np.nansum(np.square(R - pred)) / n_K)
        new_ndcg = Reward_utils.cal_exact_avg_ndcg(pred, R)
        print("current loss is {} and current ndcg is {}".format(new_err, new_ndcg))
        err_list += [[epoch, new_err, new_ndcg]]

        if new_err < current_err:
            current_err = new_err
        else:
            WP = old_WP
            WQ = old_WQ
            mu = mu
            b_p = old_b_p
            b_q = old_b_q
            predict_CaDRRes(ss_df, cl_features_df, X, WP, WQ, mu, b_p, b_q, err_list, epoch)
        if epoch % 1000 == 0:
            out_name = os.path.join(out_dir, "{}Dim".format(f))
            if not os.path.exists(out_name):
                os.makedirs(out_name)
            predict_CaDRRes(ss_df,  cl_features_df, X, WP, WQ, mu, b_p, b_q, err_list, epoch, out_name)


def predict_CaDRRes(ss_df, cl_features_df, X, WP, WQ, mu, b_p, b_q, err_list, epoch, out_name):
    f = WP.shape[1]
    P_train = list(ss_df.index)  # cell-line
    X_train = np.matrix(cl_features_df.loc[P_train])
    M = len(list(ss_df.columns))
    Y_train = np.matrix(np.identity(M))

    Q_mat_train = Y_train * WQ
    P_mat_train = X_train * WP

    temp = mu + (Q_mat_train * P_mat_train.T).T
    temp = temp + b_q
    train_pred_mat = (temp.T + b_p).T
    train_pred_df = pd.DataFrame(train_pred_mat, columns=ss_df.columns, index=ss_df.index)
    # convert sensitivity score to IC50
    train_pred_df *= -1
    #new_out_name_train = os.path.join(out_name, 'CaDRReS_pred_end_train.csv')
    # train_pred_df.to_csv(new_out_name_train)
    pd.DataFrame(WP, columns=range(1, f+1)).to_csv(os.path.join(out_name, 'WPmatrix.csv'))
    pd.DataFrame(P_mat_train, index=ss_df.index, columns=range(1, f+1)).to_csv(os.path.join(out_name, 'Pmatrix.csv'))
    pd.DataFrame(Q_mat_train, index=ss_df.columns, columns=range(1, f+1)).to_csv(os.path.join(out_name, 'Qmatrix.csv'))


def Save_GDSC_feautures():
    """
    from the Rec decompose to save features for cell and drug
    """

    args = parse_args()
    data_dir = os.path.join(os.getcwd(), args.data_dir)
    ss_file = os.path.join(os.getcwd(), "./GDSC_data/GDSC_proprecessed.npz")
    ss_data = np.load(ss_file)
    ss_df = ss_data['ss_df']
    P_file_name = data_dir+"/{}Dim/Pmatrix.csv".format(args.f)
    Q_file_name = data_dir+"/{}Dim/Qmatrix.csv".format(args.f)
    WP_file_name = data_dir+"/{}Dim/WPmatrix.csv".format(args.f)
    P = pd.read_csv(P_file_name, index_col=0)
    Q = pd.read_csv(Q_file_name, index_col=0)
    WP_mat = np.asarray(pd.read_csv(WP_file_name, index_col=0))
    P_mat = np.asarray(P)
    Q_mat = np.asarray(Q)
    # all_features=np.zeros([P_mat.shape[0],Q_mat.shape[0],P_mat.shape[1]*Q_mat.shape[1]])
    # all_features_nonan= np.zeros([P_mat.shape[0],Q_mat.shape[0],P_mat.shape[1]*Q_mat.shape[1]])
    # for i in range(P_mat.shape[0]):
    #     cross_features=np.kron(P_mat[i],Q_mat)
    #     all_features[i]=cross_features
    #     all_features_nonan[i]=cross_features
    #     drug_bool=pd.isna(ss_df[i])
    #     all_features_nonan[i][drug_bool]=0
    # np.savez_compressed("./GDSC_data/input/{}Dim/feature_with_cross_mat.npz".format(args.f),
    #             feature_mat=all_features,all_features_nonan=all_features_nonan,
    #             cell_features=P_mat,drug_features=Q_mat,WP = WP_mat)
    np.savez_compressed("./GDSC_data/{}Dim/feature_mat.npz".format(args.f),
                        cell_features=P_mat, drug_features=Q_mat, WP=WP_mat)


def GDSC_inds(args):
    ss_data = np.load("./GDSC_data/input/GDSC_proprecessed.npz")
    ss_df = ss_data['ss_df']

    N = ss_df.shape[0]
    M = ss_df.shape[1]

    inds_cl_drug = -1*np.ones((N, M)).astype(int)

    for i in range(N):
        drug_total = np.sum(~np.isnan(ss_df[i]))
        drug_index = np.argwhere(~np.isnan(ss_df[i]))[:, 0]
        inds_cl_drug[i, :drug_total] = drug_index

    pd.DataFrame(inds_cl_drug).to_csv(os.path.join(args.out_dir, 'drug_cll_index.csv'))


def load_fts_mat_with_inds(data_dir, f):
    # f is the dimension for emb space

    data_file = data_dir+"/{}Dim/feature_mat.npz".format(f)
    if not os.path.exists(data_file):
        Save_GDSC_feautures()
    data = np.load(data_file)
    cl_features = data['cell_features']  # (N,P1)
    drug_features = data['drug_features']
    #feature_mat_nonan = data['all_features_nonan']

    cl_fts_pn = torch.norm(torch.from_numpy(cl_features), p=2, dim=1)+1e-5
    cell_embs = cl_features/cl_fts_pn[:, None]

    drug_fts_pn = torch.norm(torch.from_numpy(drug_features), p=2, dim=1)+1e-5
    drug_embs = drug_features/drug_fts_pn[:, None]

    ss_file = os.path.join(
        os.getcwd(), "./GDSC_data/GDSC_proprecessed.npz")
    ss_data = np.load(ss_file)
    ss_df = ss_data['ss_df']  # (985,223)
    return cell_embs, drug_embs, ss_df


def load_PQ_WP_embs(data_dir, f):
    # data_dir like ./RL_GDSC/GDSC/CV/Fold0
    Xtrain = np.asarray(pd.read_csv(data_dir+"/XtrainDf.csv", index_col=0))  # (788,985)
    Xtest = np.asarray(pd.read_csv(data_dir+"/XtestDf.csv", index_col=0))
    Ytrain = np.asarray(pd.read_csv(data_dir+"/YtrainDf.csv", index_col=0))  # (788,223)
    Ytest = np.asarray(pd.read_csv(data_dir+"/YtestDf.csv", index_col=0))
    MF_dir = os.path.join(data_dir, "{}Dim".format(f))
    WP = np.asarray(pd.read_csv(MF_dir+"/WPmatrix.csv", index_col=0))  # P*f,(985,10)
    Q = pd.read_csv(MF_dir+"/Qmatrix.csv", index_col=0)  # M*f(223,10)
    # P = pd.read_csv(MF_dir+"/Pmatrix.csv",index_col=0)#N*f
    drug_features = np.asarray(Q)

    drug_fts_pn = torch.norm(torch.from_numpy(drug_features), p=2, dim=1)+1e-5
    drug_embs = drug_features/drug_fts_pn[:, None]
    return Xtrain, Xtest, Ytrain, Ytest, WP, drug_embs


if __name__ == "__main__":
    # Xtrain,Xtest,Ytrain,Ytest,WP,drug_embs=load_PQ_WP_embs("./GDSC/CV/Fold0",10)
    #cell_embs,drug_embs,ss_df = load_fts_mat_with_inds('./GDSC_data',10)
    # ss_df,ss_test_df,cl_features_df,cl_list = ss_file_save(args)
    # Response_decompose(ss_df,cl_features_df,args)
    # GDSC_feautures(args)
    # GDSC_inds(args)
    args = parse_args()
    X_train, X_test, Y_train, Y_test = utils.read_FULL("GDSC", "CV", i=0)
    ss_df = pd.DataFrame(X_train)
    cl_fts_df = pd.DataFrame(Y_train)
    Response_decompose(ss_df, cl_fts_df, 1000, 0.01, 10, "GDSC_data/output")
