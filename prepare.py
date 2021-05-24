from __future__ import print_function
import os
import sys
import yaml
import numpy as np
import pandas as pd
import argparse
from utils import create_save_dir
from sklearn.model_selection import KFold
from preprocess.preprocess_fts_cl_drug import Response_decompose
from results import get_result_filename
from set_log import set_logging, initialize_logger_name
from preprocess import prep_genexp
from preprocess import pp_gene_original
import pickle
from Reward_utils import ndcg
from sklearn.preprocessing import StandardScaler
# sys.path.append("../CaDRReS_SC")
####
# Simulator generates all the simulation, but prepare.py for simulation preparation or preprocessing
# CV split and feature preprocessing
####

#####
# Simulation Data
#  NMP(X.csv,W.csv)
#      --exp/linear/quad
#          -- CVfolder && Rsparse.csv, Rnoise.csv, Rtrue.csv
#               -- sparse/noise
#                   -- Fold0/Fold1/Fold2/Fold3/Fold4
#                      --40Dim && Full.csv && Xtrain.csv, Xtest.csv && Ytrain_sparse_miss0.5df.csv
#                      -- Qmatrix.csv && WPmatrix.csv

#####


def parse_args():
    parser = argparse.ArgumentParser(description="prepare data split for CV with both FULL and KEEPK method")
    parser.add_argument("--data_dir", default="GDSC_ALL")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--Source", type=str, default="GDSC", help="data source to analysis")
    parser.add_argument("--gene_type", type=str, default="GEX", help="among GEX, WES, CNV, MET")
    parser.add_argument("--filename", type=str, default="_all_abs_ic50_bayesian_sigmoid.csv",
                        help="full data set file to split")
    parser.add_argument("--gene_list_fn", type=str, default="gdsc_697_genes.csv",
                        help="essential genes data")
    parser.add_argument("--config", default="./configs/configG_FULL.yaml")
    parser.add_argument("--cl_feature_fname", default="_cellline_pcor_ess_genes.csv",
                        help="cellline features matrix")
    parser.add_argument("--drug_list_fname", default="_drugMedianGE0.txt", help="essential genes")
    parser.add_argument("--CV_dir", type=str, default="CV", help="CV data folder")
    parser.add_argument("--nfolds", type=int, default=5, help="num of folds for CV")
    parser.add_argument("--training", default=True, help="whether save the result for CaRRes")
    parser.add_argument("--decompose", default=False, action='store_true', help="whether save the result for CaRRes")
    parser.add_argument("--f", type=int, default=100,
                        help="dimension in the projected space")  # Number of dimensions
    parser.add_argument("--essential_genes", default=True, action="store_false",
                        help="whether to use the essential genes")  # Number of dimensions
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate for Caddres")

    return parser.parse_args()


def keepk_sample(Y_train, Y_test, keepk_ratio, keepk):
    # fraction of most effective drugs(keepk_ratio) used for sampling q (keepk)drugs
    inds_train = Y_train.index
    cols = Y_train.columns
    inds_test = Y_test.index

    Y_train_np = np.array(Y_train)
    Y_test_np = np.array(Y_test)
    Y_train_keepk = Y_train_np.copy()
    N = int(keepk_ratio * Y_train_keepk.shape[1])  # num of drugs to select from
    for i in range(Y_train_keepk.shape[0]):
        y = Y_train_keepk[i]
        notnan = ~np.isnan(y)
        y = y[notnan]
        y_argsort = np.argsort(y)[::-1]  # this is array with y.values
        y_argsort_pos = y_argsort[:N]
        y_argsort_neg = y_argsort[N:]
        pos_permutation = np.random.permutation(y_argsort_pos.shape[0])
        for j in range(keepk, pos_permutation.shape[0]):
            y[y_argsort_pos[pos_permutation[j]]] = np.nan
        neg_permutation = np.random.permutation(y_argsort_neg.shape[0])
        for j in range(0, neg_permutation.shape[0]):
            y[y_argsort_neg[neg_permutation[j]]] = np.nan
        Y_train_keepk[i, notnan] = y
    keep = []
    for i in range(Y_train_keepk.shape[1]):
        y = Y_train_keepk[:, i]
        if y[~np.isnan(y)].shape[0] > 1:
            keep.append(i)
    cols = cols[keep]
    keep = np.array(keep)

    Y_train_keepk = Y_train_keepk[:, keep]
    Y_test_keepk = Y_test_np[:, keep]

    Y_train_keepk = pd.DataFrame(Y_train_keepk, index=inds_train, columns=cols)
    Y_test_keepk = pd.DataFrame(Y_test_keepk, index=inds_test, columns=cols)

    return Y_train_keepk, Y_test_keepk


def Split_Data():
    NAME = "data_spilt_CV"
    logger = set_logging(NAME)
    logger.info('Splitting the data for training and testing, creating folds for cross-validation...')
    args = parse_args()
    config_file = args.config
    with open(config_file, 'r') as f:
        config = yaml.full_load(f)
    seed = args.seed
    nfolds = config['nfolds']

    analysis = config['analysis']
    data_name = config['data']  # ['GDSC','CCLE','GDSC_ALL','SimuData']
    if data_name.startswith("GDSC"):
        keepk_ratios = np.array(config['keepk_ratios_G'], dtype=float)
    elif data_name == "CCLE":
        keepk_ratios = np.array(config['keepk_ratios_C'], dtype=float)

    keepk = config['keepk']

    directory = os.path.join(os.getcwd(), args.data_dir)
    if config['Data_All']:
        data_all_fn = os.path.join(directory, args.Source+"_" + args.gene_type+".npz")
        Data_All = np.load(data_all_fn)
        cl_feature_arr = Data_All['X']  # (962,11737)
        ss_arr = Data_All['Y']  # already normalized -logIC50 (962,265)
        drug_lists = Data_All['drug_ids']  # (265,) # number as 1,2,3..265 string in GDSC, like 211	TL-2-105
        # comes from Cell_line_RMA_proc_basalExp.txt, including all the genes in essential_gens(1856)
        gene_symbols = Data_All['GEX_gene_symbols']
        # (962,) # also corresponding to Cell_line_RMA_proc_basalExp.txt and IC50 file, cell_ids with gene_symbols
        cell_ids = Data_All['cell_ids']  # like ['1240121', '1240122'],Cell line cosmic identifiers
        cell_names = Data_All['cell_names']  # corresponds to IC50 file, like ['BICR22', 'BICR78']
        P = cell_ids
        Y = pd.DataFrame(ss_arr, index=cell_ids, columns=drug_lists)
        X = pd.DataFrame(cl_feature_arr, index=cell_ids, columns=gene_symbols)  # 【962，17737】

        # ess_genes_list = pp_gene_original.get_gene_list(os.path.join(
        #     os.getcwd(),
        #     args.data_dir, args.gene_list_fn))  # 1856 essential gens
        # ess_genes_common = np.intersect1d(ess_genes_list, list(gene_symbols))  # 1610 common essential genes

        ess_genes_list = pd.read_csv(os.path.join(os.getcwd(), args.data_dir, args.gene_list_fn),
                                     index_col=0, header=0, names=["genes"])
        ess_genes_common = list(set(X.columns) & set(ess_genes_list['genes']))

        X = X[ess_genes_common]
        cell_names_to_ids_map = {cell_names[i]: cell_ids[i] for i in range(len(cell_ids))}
        cell_ids_to_names_map = {cell_ids[i]: cell_names[i] for i in range(len(cell_ids))}

        # Doing transformation for X
        X_log2_exp_df = pp_gene_original.log2_exp(X).T  # transform
        logger.info(
            "doing transform of log2 gene exp as X_log2_exp_df with shape {} {}".format(
                X_log2_exp_df.shape[0],
                X_log2_exp_df.shape[1]))
        X_log2_exp_df = X_log2_exp_df.groupby(X_log2_exp_df.index).mean()
        logger.info("Calculate log2 fold-change based on mean")
        X_log2_mean_fc_exp_df, X_mean_exp_df = pp_gene_original.normalize_log2_mean_fc(X_log2_exp_df)
        logger.info("Read essential genes")
        logger.info("X_log2_mean_fc_exp_df shape {} {}".format(
            X_log2_mean_fc_exp_df.shape[0],
            X_log2_mean_fc_exp_df.shape[1]))
        logger.info("calculate kernel features based on pearson correlation")
        kernel_feature_df = pp_gene_original.calculate_kernel_feature(
            X_log2_mean_fc_exp_df, X_log2_mean_fc_exp_df, list(ess_genes_list['genes']))
        kernel_feature_df.to_csv(os.path.join(directory, "kernel_feature.csv"))
        # we have already calculate the overlap int loading data, no need to doubble check list again

    else:
        cl_feature_fname = os.path.join(directory, "{}".format(args.Source)+args.cl_feature_fname)
        cl_features_df = pd.read_csv(cl_feature_fname, index_col=0)  # (1014,1014)

        fn = os.path.join(directory, "{}".format(args.Source)+args.filename)
        ss_df = pd.read_csv(fn, index_col=0)  # cellline_drug response matrix, n*m, S matrix
        ss_df.index = ss_df.index.astype(str)  # cell_line names
        # Convert IC50 to sensitivity score, -log(IC50)
        ss_df *= -1
        drug_list_fname = os.path.join(directory, "{}".format(args.Source)+args.drug_list_fname)
        selected_drugs = list(pd.read_csv(drug_list_fname, header=None)[0].values.astype(str))
        drug_list = list(ss_df.columns)  # 265 drugs, after delete from median 1um, has 223 drugs
        drug_list = [d for d in drug_list if d in selected_drugs]  # 223 drugs

        cl_features_df.index = cl_features_df.index.astype(str)  # (1014,1014)
        cl_list = list(cl_features_df.index.astype(str))
        ss_df = ss_df[ss_df.index.isin(cl_list)][drug_list]  # (985,265)
        ss_df = ss_df.dropna(how="all")

        cl_features_df = cl_features_df[ss_df.index]
        # select only predictable cell lines and drugs
        # cellline_drug response matrix, n*m, S matrix
        P = list(ss_df.index)   # cell lines

        # select only predictable cell lines and drugs
        # S matrix, the cellline_drug_matrix
        Y = ss_df  # (985,223),
        X = cl_features_df.loc[P]  # (985,985)

    # when it is keepk, use 3fold cv
    if args.data_dir.startswith('GDSC') and analysis == "KEEPK":
        if args.CV_dir == "CV":
            args.CV_dir = args.CV_dir+str(args.nfolds)
    CV_dir = os.path.join(directory, args.CV_dir)
    logger.info("CV directory is {}".format((CV_dir)))
    if not os.path.exists(CV_dir):
        os.makedirs(CV_dir)

    if analysis == "FULL":
        np.random.seed(seed)
        kf = KFold(n_splits=args.nfolds, shuffle=True, random_state=args.seed)
        for i, (train_index, test_index) in enumerate(kf.split(X)):
            print("spit fold {}".format(i))
            X_train_full = X.iloc[train_index]  # (788,985)
            X_test = X.iloc[test_index]  # (197,985)
            Y_train_full = Y.iloc[train_index]  # (788,223)
            Y_test = Y.iloc[test_index]  # (197,223)
            X_train_kernel = kernel_feature_df.iloc[train_index, train_index]
            X_test_kernel = kernel_feature_df.iloc[test_index, train_index]
            # '/home/liux3941/RL/RL_GDSC/GDSC_ALL/CV/FULL/Fold0'
            fold_dir = os.path.join(os.getcwd(), args.data_dir, CV_dir, analysis, "Fold{}".format(i))
            if not os.path.exists(fold_dir):
                os.makedirs(fold_dir)
            X_train_full.to_csv(os.path.join(fold_dir, 'Xtrain_rawDf.csv'))
            Y_train_full.to_csv(os.path.join(fold_dir, 'YtrainDf.csv'))
            X_test.to_csv(os.path.join(fold_dir, 'Xtest_rawDf.csv'))
            Y_test.to_csv(os.path.join(fold_dir, 'YtestDf.csv'))
            X_train_kernel.to_csv(os.path.join(fold_dir, "Xtrain_kernel.csv"))
            X_test_kernel.to_csv(os.path.join(fold_dir, "Xtest_kernel.csv"))

            # np.savez_compressed(fold_dir+"/FulltrainDf.npz".format(i), Xtrain=X_train_full, Ytrain=Y_train_full)
            # np.savez_compressed(fold_dir+"/FulltestDf.npz".format(i), Xtest=X_test, Ytest=Y_test)

            if config['tunning']:
                for j, (train_index, val_index) in enumerate(kf.split(X_train_full)):
                    X_train = X.iloc[train_index]  # (630,)
                    X_val = X.iloc[val_index]  # (197,)   taken as validate data
                    Y_train = Y.iloc[train_index]
                    Y_val = Y.iloc[val_index]
                    X_train.to_csv(os.path.join(fold_dir, 'XtrainDf_fold{}.csv'.format(j)))
                    Y_train.to_csv(os.path.join(fold_dir, 'YtrainDf_fold{}.csv'.format(j)))
                    X_val.to_csv(os.path.join(fold_dir, 'XvalDf_fold{}.csv'.format(j)))
                    Y_val.to_csv(os.path.join(fold_dir, 'YvalDf_fold{}.csv'.format(j)))
                    np.savez_compressed(fold_dir+"/trainDf_fold{}.npz".format(j), Xtrain=X_train, Ytrain=Y_train)
                    np.savez_compressed(fold_dir+"/valDf_fold{}.npz".format(j), Xval=X_val, Yval=Y_val)
    elif analysis == "KEEPK":
        np.random.seed(seed)
        kf = KFold(n_splits=args.nfolds, shuffle=True, random_state=args.seed)
        for kr in keepk_ratios:
            for i, (train_index, test_index) in enumerate(kf.split(Y)):
                X_train_full = X.iloc[train_index]  # (788,985)
                X_test = X.iloc[test_index]  # (197,985)
                Y_train = Y.iloc[train_index]
                Y_test = Y.iloc[test_index]
                Y_train_keepk, Y_test_keepk = keepk_sample(Y_train, Y_test, kr, keepk)
                fold_dir = os.path.join(os.getcwd(), args.data_dir, CV_dir, analysis,
                                        "keep_{}_kr_{}".format(keepk, kr), "Fold{}".format(i))
                # eg:'~/RL_GDSC/GDSC/CV/KEEPK/keep_5_kr_1.0/Fold0'
                if not os.path.exists(fold_dir):
                    os.makedirs(fold_dir)
                X_train_full.to_csv(os.path.join(fold_dir, 'XtrainDf.csv'))
                Y_train_keepk.to_csv(os.path.join(fold_dir, 'YtrainDf.csv'))
                X_test.to_csv(os.path.join(fold_dir, 'XtestDf.csv'))
                Y_test_keepk.to_csv(os.path.join(fold_dir, 'YtestDf.csv'))
                np.savez_compressed(fold_dir+"/Full_keepktrainDf.npz", Xtrain=X_train_full, Ytrain=Y_train_keepk)
                np.savez_compressed(fold_dir+"/Full_keepktestDf.npz", Xtest=X_test, Ytest=Y_test_keepk)
                if config['tunning']:
                    for j, (train_index, val_index) in enumerate(kf.split(X_train_full)):
                        X_train = X.iloc[train_index]  # (630,)
                        X_val = X.iloc[val_index]  # (197,)   taken as validate data
                        Y_train = Y.iloc[train_index]
                        Y_val = Y.iloc[val_index]
                        Y_train_keepk, Y_val_keepk = keepk_sample(Y_train, Y_val, kr, keepk)
                        X_train.to_csv(os.path.join(fold_dir, 'XtrainDf_fold{}.csv'.format(j)))
                        Y_train_keepk.to_csv(os.path.join(fold_dir, 'YtrainDf_fold{}.csv'.format(j)))
                        X_val.to_csv(os.path.join(fold_dir, 'XvalDf_fold{}.csv'.format(j)))
                        Y_val_keepk.to_csv(os.path.join(fold_dir, 'YvalDf_fold{}.csv'.format(j)))
                        np.savez_compressed(fold_dir+"/trainDf_fold{}.npz".format(j), Xtrain=X_train, Ytrain=Y_train)
                        np.savez_compressed(fold_dir+"/valDf_fold{}.npz".format(j), Xval=X_val, Yval=Y_val)

    if args.decompose:
        Pretrained_MF_split()


def Load_from_decompose():
    args = parse_args()
    config_file = args.config
    gene_list_fn = args.gene_list_fn
    with open(config_file, 'r') as f:
        config = yaml.full_load(f)

    analysis = config['analysis']
    data_name = config['data']  # ['GDSC','CCLE','GDSC_ALL']
    if data_name.startswith("GDSC"):
        keepk_ratios = np.array(config['keepk_ratios_G'], dtype=float)
    elif data_name == "CCLE":
        keepk_ratios = np.array(config['keepk_ratios_C'], dtype=float)

    keepk = config['keepk']

    nfolds = args.nfolds

    f = args.f

    if analysis == "FULL":
        out_dir = os.path.join(os.getcwd(), args.data_dir, args.CV_dir, analysis)

        for i in range(nfolds):
            out_dir_cv = os.path.join(out_dir, "Fold{}".format(i))
            print(out_dir_cv)
            # train_data = np.load(out_dir_cv+"/FulltrainDf.npz")
            # Xtrain = train_data['Xtrain']
            # Ytrain = train_data['Ytrain']
            Xtrain_df = pd.read_csv(out_dir_cv+'/Xtrain_rawDf.csv', index_col=0)
            Ytrain_df = pd.read_csv(out_dir_cv+'/YtrainDf.csv', index_col=0)
            Ytest_df = pd.read_csv(out_dir_cv+'/YtestDf.csv', index_col=0)
            Xtest_df = pd.read_csv(out_dir_cv+'/Xtest_rawDf.csv', index_col=0)
            Xtrain_kernel = pd.read_csv(out_dir_cv+'/Xtrain_kernel.csv', index_col=0)
            Xtest_kernel = pd.read_csv(out_dir_cv + '/Xtest_kernel.csv', index_col=0)
            cadrres_out_dict = pickle.load(
                open(
                    os.path.join(
                        out_dir_cv, 'cadrres_{}Dim_5f_{}_output_dict.pickle'.format(
                            args.f, i)),
                    'rb'))
            cadrres_model_dict = pickle.load(
                open(
                    os.path.join(
                        out_dir_cv, 'cadrres_{}Dim_5f_{}_param_dict.pickle'.format(
                            args.f, i)),
                    'rb'))

            P_df = cadrres_out_dict['P_df']  # P = tf.matmul(X, W_P) [769,10], X is [769,769]
            Q_df = cadrres_out_dict['Q_df']  # (265,10)
            pred_test_df = cadrres_out_dict['pred_test_df']  # (193,265)
            WP = cadrres_model_dict['W_P']  # array, [769,10]
            WQ = cadrres_model_dict['W_Q']  # Q = tf.matmul(I, W_Q), I is identity(ndrugs),array,[265,10]
            Q = np.matmul(np.identity(Ytrain_df.shape[1]), WQ)  # [265,10]
            b_q = cadrres_model_dict['b_Q']

            out_dir_cv_dim = os.path.join(out_dir_cv, "{}Dim".format(f))
            if not os.path.exists(out_dir_cv_dim):
                os.makedirs(out_dir_cv_dim)

            pd.DataFrame(WP, columns=range(1, f+1)).to_csv(os.path.join(out_dir_cv_dim, 'WPmatrix.csv'))
            pd.DataFrame(P_df, index=Xtrain_df.index, columns=range(1, f+1)
                         ).to_csv(os.path.join(out_dir_cv_dim, 'Pmatrix.csv'))
            pd.DataFrame(Q, index=Ytrain_df.columns, columns=range(1, f+1)
                         ).to_csv(os.path.join(out_dir_cv_dim, 'Qmatrix.csv'))

            pred_model = cadrres_out_dict['pred_test_df']*-1
            pred = np.matmul(np.matmul(np.asarray(Xtest_kernel), WP), WQ.T) + b_q.T

            result_fn = get_result_filename(
                'CaDRRes', analysis, data_name, i, f)
            print("results file name is {}".format(result_fn))
            np.savez(result_fn, Y_true=np.array(Ytest_df), Y_pred=pred)
            ndcg_test = ndcg(np.array(Ytest_df), pred, k=265, full_rank=True)
            print("ndcg test val is {}".format(ndcg_test))


# Response_decompose(ss_df,cl_features_df,iters,lr,f,out_dir)
def Pretrained_MF_split(iters=20000):
    args = parse_args()
    config_file = args.config
    gene_list_fn = args.gene_list_fn
    with open(config_file, 'r') as f:
        config = yaml.full_load(f)

    analysis = config['analysis']
    data_name = config['data']  # ['GDSC','CCLE']
    if data_name == "GDSC":
        keepk_ratios = np.array(config['keepk_ratios_G'], dtype=float)
    elif data_name == "CCLE":
        keepk_ratios = np.array(config['keepk_ratios_C'], dtype=float)

    keepk = config['keepk']

    nfolds = args.nfolds

    if args.data_dir == 'GDSC' and analysis == "KEEPK":
        if args.CV_dir == "CV":
            args.CV_dir = args.CV_dir+"3"

    f = args.f
    lr = args.lr
    if analysis == "FULL":
        out_dir = os.path.join(os.getcwd(), args.data_dir, args.CV_dir, analysis)
        for i in range(nfolds):
            out_dir_cv = os.path.join(out_dir, "Fold{}".format(i))
            # train_data = np.load(out_dir_cv+"/FulltrainDf.npz")
            # Xtrain = train_data['Xtrain']
            # Ytrain = train_data['Ytrain']
            Xtrain_df = pd.read_csv(out_dir_cv+'/Xtrain_rawDf.csv', index_col=0)
            Ytrain_df = pd.read_csv(out_dir_cv+'/YtrainDf.csv', index_col=0)
            Ytest_df = pd.read_csv(out_dir_cv+'/YtestDf.csv', index_col=0)
            Xtest_df = pd.read_csv(out_dir_cv+'/Xtest_rawDf.csv', index_col=0)
            Xtrain_kernel = pd.read_csv(out_dir_cv+'/Xtrain_kernel.csv', index_col=0)
            Xtest_kernel = pd.read_csv(out_dir_cv+'/Xtest_kernel.csv', index_col=0)

            # Xtrain_corrdf, Xtest_corrdf = prep_genexp.pre_kernal_genes(Xtrain_df, Xtest_df, Ytrain_df, Ytest_df, os.path.join(
            #     os.getcwd(),
            #     args.data_dir, gene_list_fn,out_dir_cv)
            ###
            xscaler = StandardScaler()
            Xtrain_df = pd.DataFrame(xscaler.fit_transform(Xtrain_df), columns=Xtrain_df.columns, index=Xtrain_df.index)
            Xtest_df = pd.DataFrame(xscaler.transform(Xtest_df), columns=Xtest_df.columns, index=Xtest_df.index)
            Ypred_mat = Response_decompose(Ytrain_df, Xtrain_df, Ytest_df, Xtest_df, iters,
                                           lr, f, out_dir_cv, i, training=True)
            if Ypred_mat is not None:
                print('storing the Ypred mat from CaDRRes')
                result_fn = get_result_filename('CaDRRes', analysis, data_name, i, args.f)
                np.savez(result_fn, Y_true=np.array(Ytest_df), Y_pred=Ypred_mat)

    elif analysis == "KEEPK":
        for kr in keepk_ratios:
            out_dir = os.path.join(os.getcwd(), args.data_dir, args.CV_dir, analysis, "keep_{}_kr_{}".format(keepk, kr))
            for i in range(nfolds):
                out_dir_cv = os.path.join(out_dir, "Fold{}".format(i))
                Xtrain_df = pd.read_csv(out_dir_cv+'/Xtrain_rawDf.csv', index_col=0)  # (788,985)
                Ytrain_df = pd.read_csv(out_dir_cv+'/YtrainDf.csv', index_col=0)  # (788,215)
                Ytest_df = pd.read_csv(out_dir_cv+'/YtestDf.csv', index_col=0)
                Xtest_df = pd.read_csv(out_dir_cv+'/Xtest_rawDf.csv', index_col=0)
                Ypred_mat = Response_decompose(Ytrain_df, Xtrain_df, Ytest_df, Xtest_df,
                                               iters, lr, f, out_dir_cv, training=args.training)
                if Ypred_mat is not None:
                    print('storing the Ypred mat from CaDRRes for fold {} and kr {}'.format(i, kr))
                    result_fn = get_result_filename('CaDRRes', analysis, data_name, i, args.f, keepk=keepk, ratio=kr)
                    print(result_fn)
                    np.savez(result_fn, Y_true=np.array(Ytest_df), Y_pred=Ypred_mat)


if __name__ == "__main__":
    # Split_Data()
    Pretrained_MF_split()
    # Load_from_decompose()
