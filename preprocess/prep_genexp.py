import pandas as pd
import numpy as np
from scipy import stats
import time


def log2_exp(exp_df):
    """Calculate log2 gene expression
    """

    return np.log2(exp_df + 1)


def normalize_log2_mean_fc(log2_exp_df):
    """Calculate gene expression foldchange based on median of each genes. The sample size should be large enough (>10).
    """

    return (log2_exp_df - log2_exp_df.mean(axis=0)), pd.DataFrame(log2_exp_df.mean(axis=0), columns=['median'])


def normalize_log2_mean_fc_with_ref(log2_exp_df, log2_ref_exp_df):

    common_genes = set(log2_ref_exp_df.index).intersection(log2_exp_df.index)
    log2_exp_df = log2_exp_df.loc[common_genes]
    log2_ref_exp_df = log2_ref_exp_df.loc[common_genes]

    return (log2_exp_df - log2_ref_exp_df.mean(axis=0)), pd.DataFrame(log2_ref_exp_df.mean(axis=0), columns=['median'])


# TODO: make this multiprocessor
def calculate_kernel_feature(test_log2_median_fc_exp_df, train_log2_median_fc_exp_df, gene_list):
    common_genes = [g for g in gene_list if (g in list(test_log2_median_fc_exp_df.columns))
                    and (g in list(train_log2_median_fc_exp_df.columns))]  # 1610 common genes in ess genes

    print('Calculating kernel features based on', len(common_genes), 'common genes')

    print(
        "test shape {}, {}".format(test_log2_median_fc_exp_df.shape[0],
                                   test_log2_median_fc_exp_df.shape[1]))
    print("train shape {},{}".format(
          train_log2_median_fc_exp_df.shape[0], train_log2_median_fc_exp_df.shape[1]))

    test_list = list(test_log2_median_fc_exp_df.index)
    train_sample_list = list(train_log2_median_fc_exp_df.index)

    test_exp_mat = np.array(test_log2_median_fc_exp_df.loc[:, common_genes], dtype='float')
    train_exp_mat = np.array(train_log2_median_fc_exp_df.loc[:, common_genes], dtype='float')

    sim_mat = np.zeros((len(test_list), len(train_sample_list)))  # 769,769

    start = time.time()
    for i in range(len(test_list)):
        if (i+1) % 100 == 0:
            print("{} of {} ({:.2f})s".format(i+1, len(test_list), time.time()-start))
            start = time.time()
        for j in range(len(train_sample_list)):
            p_cor, _ = stats.pearsonr(test_exp_mat[i, :], train_exp_mat[j, :])
            sim_mat[i, j] = p_cor

    return pd.DataFrame(sim_mat, columns=train_sample_list, index=test_list)


def get_gene_list(gene_list_fname):
    return list(pd.read_csv(gene_list_fname, header=None)[0].values)


def pre_kernal_genes(XtrainDf, XtestDf, YtrainDf, YtestDf, gene_list_fname, out_dir):
    cl_train_log2_mean_fc_exp_df, cl_train_mean_exp_df = normalize_log2_mean_fc(XtrainDf)
    ess_genes_list = get_gene_list(gene_list_fname)  # 1856
    # this reauire train and test have the same columns
    cl_test_log2_mean_fc_exp_df = XtestDf.values-cl_train_mean_exp_df.T.values
    cl_test_log2_mean_fc_exp_df = pd.DataFrame(
        cl_test_log2_mean_fc_exp_df, index=XtestDf.index, columns=XtestDf.columns)

    test_kernel_feature_df = calculate_kernel_feature(
        cl_test_log2_mean_fc_exp_df, cl_train_log2_mean_fc_exp_df, ess_genes_list)
    print("test kernel feature df shape {},{}".format(test_kernel_feature_df.shape[0], test_kernel_feature_df.shape[1]))
    print("save test_kernel_feature_df")
    test_kernel_feature_df.to_csv(out_dir+"/Xtest_corrDf.csv")
    train_kernel_feature_df = calculate_kernel_feature(
        cl_train_log2_mean_fc_exp_df, cl_train_log2_mean_fc_exp_df, ess_genes_list)
    print("train kernel feature df shape {},{}".format(
        train_kernel_feature_df.shape[0],
        train_kernel_feature_df.shape[1]))
    print("save train_kernel_feature_df")
    train_kernel_feature_df.to_csv(out_dir + "/Xtrain_corrDf.csv")
    return train_kernel_feature_df, test_kernel_feature_df
