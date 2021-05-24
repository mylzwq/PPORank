import os
import sys
import yaml
import numpy as np
import pandas as pd
import argparse
from utils import create_save_dir
from sklearn.model_selection import KFold
from preprocess.preprocess_fts_cl_drug import Response_decompose, Response_decompose2
from results import get_result_filename
from Simulation.utils_simu import noramlize_y, read_simu
import multiprocessing as mp
from multiprocessing import Pool
import time


def parse_args():
    parser = argparse.ArgumentParser(
        description="prepare data split of simulated data for CV with both FULL and KEEPK method")
    parser.add_argument("--data_dir", default="SimuData")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--Source", type=str, default="SimuData", help="data source to analysis")
    parser.add_argument("--scenarios", default="linear,quad,exp", help="can be linear, quad, exp")
    parser.add_argument("--scenario", default="linear", help="can be linear, quad, exp")
    parser.add_argument("--filename", type=str, default="Rsparsedf.csv",
                        help="full data set file to split")
    parser.add_argument("--analysises", default="sparse")

    # parser.add_argument("--config",default="./configs/configG_keepk.yaml")
    parser.add_argument("--cl_feature_fname", default="Xdf.csv",
                        help="cellline features matrix")
    parser.add_argument("--drug_features_fname", default="Vdf.csv")
    parser.add_argument("--CV_dir", type=str, default="CV", help="CV data folder")
    parser.add_argument("--nfolds", type=int, default=5, help="num of folds for CV")
    parser.add_argument("--training", default=False, help="whether save the result for CaRRes")
    parser.add_argument("--decompose", default=True, help="whether save the result for CaRRes")
    parser.add_argument("--f", type=int, default=20, help="clusters")  # Number of dimensions
    parser.add_argument("--analysis", default="sparse", help="noise or sparse data to use")  # Number of dimensions
    parser.add_argument("--norm_y", default=False, action="store_true")  # Number of dimensions
    parser.add_argument("--iters", default=3500, type=int, help="CaD iterations")  # Number of dimensions
    parser.add_argument("--lr", default=0.005, type=int, help="CaD learning rate")  # Number of dimensions
    parser.add_argument("--N", default=10000, type=int)
    parser.add_argument("--M", default=250, type=int)
    parser.add_argument("--P", default=1000, type=int)
    parser.add_argument("--miss_ratio", default=0.95, type=float)
    return parser.parse_args()


def Split_Simu_Data(args=None):
    if not args:
        args = parse_args()
    nfolds = args.nfolds
    analysis = args.analysis
    data_name = args.Source
    N = args.N
    P = args.P
    M = args.M
    f = args.f
    simu_data_dir = os.path.join(os.getcwd(), args.data_dir, "N{}_P{}_M{}".format(N, P, M))
    # ./SimuData/N200_P60_M60/
    X_fn = os.path.join(simu_data_dir, "Xdf.csv")
    W_fn = os.path.join(simu_data_dir, "Wdf.csv")
    # V_fn = os.path.join(directory,"Vdf.csv")

    X_df = pd.read_csv(X_fn, index_col=0)
    W_df = pd.read_csv(W_fn, index_col=0)

    X = X_df.copy()

    scenarios = args.scenarios
    scenarios = scenarios.split(',')
    analysises = args.analysises
    analysises = analysises.split(',')

    for scenario in scenarios:
        for analysis in analysises:
            if analysis == 'sparse':
                fn = 'R'+analysis+"_miss{}".format(args.miss_ratio)+'df.csv'  # Rsparse_miss99df.csv
                Ytrain_fn = 'Ytrain_' + analysis+"_miss{}".format(args.miss_ratio)+'Df.csv'
                Ytest_fn = 'Ytest_' + analysis+"_miss{}".format(args.miss_ratio)+'Df.csv'
            elif analysis == "noise":
                fn = 'R'+analysis+'df.csv'
                Ytrain_fn = 'Ytrain_' + analysis+'_Df.csv'
                Ytest_fn = 'Ytest_' + analysis+'_Df.csv'

            R_fn = os.path.join(simu_data_dir, scenario, fn)

            # ./SimuData/N200_P60_M60/linear/Rnoisedf.csv
            Y = pd.read_csv(R_fn, index_col=0)
            CV_dir = os.path.join(simu_data_dir, scenario, args.CV_dir)
            if not os.path.exists(CV_dir):
                os.makedirs(CV_dir)
            np.random.seed(args.seed)
            kf = KFold(n_splits=args.nfolds, shuffle=True, random_state=args.seed)
            for i, (train_index, test_index) in enumerate(kf.split(X)):
                X_train_full = X.iloc[train_index]  # (788,985)
                X_test = X.iloc[test_index]  # (197,985)

                Y_train_full = Y.iloc[train_index]  # (788,223)
                Y_test = Y.iloc[test_index]  # (197,223)

                # CV_dir :./SimuData/N200_P60_M60/linear/CV
                fold_dir = os.path.join(CV_dir, analysis, "Fold{}".format(i))
                if not os.path.exists(fold_dir):
                    os.makedirs(fold_dir)
                X_train_full.to_csv(os.path.join(fold_dir, 'XtrainDf.csv'))

                Y_train_full.to_csv(os.path.join(fold_dir, Ytrain_fn))
                X_test.to_csv(os.path.join(fold_dir, 'XtestDf.csv'))
                Y_test.to_csv(os.path.join(fold_dir, Ytest_fn))
                np.savez_compressed(fold_dir+"/FulltrainDf.npz".format(i), Xtrain=X_train_full, Ytrain=Y_train_full)
                np.savez_compressed(fold_dir+"/FulltestDf.npz".format(i), Xtest=X_test, Ytest=Y_test)


def nfolds_response_decompose(out_dir, folds, iters, lr, f, analysis, data_name,
                              data_dims_name, miss_ratio,
                              scenario, foldi, norm_y=False):

    out_dir_nfolds = os.path.join(out_dir, "Fold{}".format(foldi))

    if analysis == 'sparse':
        fn = 'R'+analysis+"_miss{}".format(miss_ratio)+'df.csv'  # Rsparse_miss99df.csv
        Ytrain_fn = 'Ytrain_' + analysis+"_miss{}".format(miss_ratio)+'Df.csv'
        Ytest_fn = 'Ytest_' + analysis+"_miss{}".format(miss_ratio)+'Df.csv'
    elif analysis == "noise":
        fn = 'R'+analysis+'df.csv'
        Ytrain_fn = 'Ytrain_' + analysis+'_Df.csv'
        Ytest_fn = 'Ytest_' + analysis+'_Df.csv'

    train_data = np.load(out_dir_nfolds+"/FulltrainDf.npz")
    Xtrain = train_data['Xtrain']
    Ytrain = train_data['Ytrain']

    Xtrain_df = pd.read_csv(out_dir_nfolds+'/XtrainDf.csv', index_col=0)
    Ytrain_df = pd.read_csv(os.path.join(out_dir_nfolds, Ytrain_fn), index_col=0)
    Ytest_df = pd.read_csv(os.path.join(out_dir_nfolds, Ytest_fn), index_col=0)
    Xtest_df = pd.read_csv(out_dir_nfolds+'/XtestDf.csv', index_col=0)

    if norm_y:
        # no need to normalize during split
        inds_train = Ytrain_df.index
        inds_test = Ytest_df.index
        Ytrain, Ytest = noramlize_y(Ytrain_df.values, Ytest_df.values)
        Ytrain_df = pd.DataFrame(Ytrain, index=inds_train)
        Ytest_df = pd.DataFrame(Ytest, index=inds_test)

    Ypred_mat = Response_decompose(
        Ytrain_df, Xtrain_df, Ytest_df, Xtest_df, iters, lr, f, out_dir_nfolds, foldi, Ytest_df=Ytest_df,
        analysis=analysis, data_name=data_name, data_dims_name=data_dims_name,
        miss_ratio=miss_ratio, scenario=scenario, training=True, fixed_WP=False)

    if Ypred_mat is not None:
        print('storing the Ypred mat from CaDRRes for fold {} '.format(foldi))
        result_fn = get_result_filename(
            'CaDRRes', analysis, data_name, foldi, f, scenario=scenario,
            data_dims_name=data_dims_name,
            miss_ratio=miss_ratio)
        if not os.path.isfile(result_fn):
            print(result_fn)
            np.savez(result_fn, Y_true=np.array(Ytest_df), Y_pred=Ypred_mat)


def Pretrained_MF_split_simu():
    args = parse_args()
    # args.N = 200
    # args.P = 60
    # args.miss_ratio = 0.5
    # args.M = 60
    nfolds = args.nfolds
    analysis = args.analysis
    data_name = args.Source
    iters = args.iters
    lr = args.lr
    N = args.N
    M = args.M
    P = args.P
    print('choosing to noramize y {}'.format(args.norm_y))
    directory = os.path.join(os.getcwd(), args.data_dir, "N{}_P{}_M{}".format(N, P, M))
    X_fn = os.path.join(directory, "Xdf.csv")
    W_fn = os.path.join(directory, "Wdf.csv")

    if not os.path.exists(X_fn):
        Split_Simu_Data(args)

    X_df = pd.read_csv(X_fn, index_col=0)
    W_df = pd.read_csv(W_fn, index_col=0)

    X = X_df.copy()

    f = args.f
    if analysis == 'sparse':
        fn = 'R'+analysis+"_miss{}".format(args.miss_ratio)+'df.csv'  # Rsparse_miss99df.csv
        Ytrain_fn = 'Ytrain_' + analysis+"_miss{}".format(args.miss_ratio)+'Df.csv'
        Ytest_fn = 'Ytest_' + analysis+"_miss{}".format(args.miss_ratio)+'Df.csv'

    elif analysis == "noise":
        fn = 'R'+analysis+'df.csv'
        Ytrain_fn = 'Ytrain_' + analysis+'_Df.csv'
        Ytest_fn = 'Ytest_' + analysis+'_Df.csv'

    scenarios = args.scenarios
    scenarios = scenarios.split(',')
    simu_data_dir = os.path.join(os.getcwd(), args.data_dir, "N{}_P{}_M{}".format(N, P, M))
    for scenario in scenarios:
        CV_dir = os.path.join(simu_data_dir, scenario, args.CV_dir)
        R_fn = os.path.join(directory, scenario, fn)
        Y = pd.read_csv(R_fn, index_col=0)
        out_dir = os.path.join(
            CV_dir,
            analysis)
        # out_dir ./SimuData/N200_P60_M60/linear/CV/sparse
        data_dims_name = "N{}_P{}_M{}".format(N, P, M)
        miss_ratio = args.miss_ratio
        for foldi in range(nfolds):
            nfolds_response_decompose(out_dir, nfolds, iters, lr, f, analysis, data_name,
                                      data_dims_name, miss_ratio,
                                      scenario, foldi)


if __name__ == "__main__":
    # Split_Simu_Data()
    # Pretrained_MF_split_simu()
    args = parse_args()
    nfolds = args.nfolds
    analysis = args.analysis
    data_name = args.Source
    iters = args.iters
    lr = args.lr
    N = args.N
    M = args.M
    P = args.P
    scenario = args.scenario
    directory = os.path.join(os.getcwd(), args.data_dir, "N{}_P{}_M{}".format(N, P, M))
    X_fn = os.path.join(directory, "Xdf.csv")
    W_fn = os.path.join(directory, "Wdf.csv")
    if not os.path.exists(X_fn):
        Split_Simu_Data(args)
    X_df = pd.read_csv(X_fn, index_col=0)
    W_df = pd.read_csv(W_fn, index_col=0)
    X = X_df.copy()
    fn = 'R'+analysis+"_miss{}".format(args.miss_ratio)+'df.csv'  # Rsparse_miss99df.csv
    Ytrain_fn = 'Ytrain_' + analysis+"_miss{}".format(args.miss_ratio)+'Df.csv'
    Ytest_fn = 'Ytest_' + analysis+"_miss{}".format(args.miss_ratio)+'Df.csv'

    f = args.f
    simu_data_dir = os.path.join(os.getcwd(), args.data_dir, "N{}_P{}_M{}".format(N, P, M))
    CV_dir = os.path.join(simu_data_dir, scenario, args.CV_dir)
    R_fn = os.path.join(directory, scenario, fn)
    Y = pd.read_csv(R_fn, index_col=0)
    out_dir = os.path.join(
        CV_dir,
        analysis)
    # out_dir ./SimuData/N200_P60_M60/linear/CV/sparse
    data_dims_name = "N{}_P{}_M{}".format(N, P, M)
    miss_ratio = args.miss_ratio

    pool = Pool(processes=5)
    start = time.time()
    params = [(out_dir, nfolds, iters, lr, f, analysis, data_name,
               data_dims_name, miss_ratio, scenario, foldi) for foldi in range(nfolds)]
    pool.starmap(nfolds_response_decompose, params)
    pool.close()
    pool.join()
    print('The cross-validation  on 5 cores took time (s): {}'.format(time.time() - start))
