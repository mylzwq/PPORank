import argparse
import sys
import itertools
import numpy as np
import traceback
import yaml
import os
from Reward_utils import *
from utils import *
import Reward_utils

# config_file = './configs/configC.yaml'


def parse_args():
    parser = argparse.ArgumentParser(
        description="save fold comparation results for all methods")
    parser.add_argument("--config", type=str,
                        default='./configs/configS_exp.yaml')
    parser.add_argument("--k", type=int,
                        default=None)

    return parser.parse_args()


def float2str(number):
    return '{}'.format(number)


def result_info(data_name, metric_name, results, method, rank_k, analysis, keepk, ratio, debug=False):
    return " ".join(
        [data_name, analysis, str(keepk),
         '{:g}'.format(ratio),
         metric_name, method, "rank_{}".format(rank_k),
         "mean_{}".format(non_nan_mean(results)),
         " ".join(map(float2str, results)) if not debug else ''])


def get_result_filename(
        method, analysis, data_name, fold1, f, debug=False, keepk=None, ratio=None, scenario="", data_dims_name="",
        miss_ratio=None, fold2=-1, params=None):
    # only ppo, dnn, CaDRRes relates with projected dimension f
    if debug is False:
        if method != "KRL":
            if analysis == "FULL":
                # "./results/CCLE/FULL/KRL/KRL_cv1"
                directory_method = "{}/results/{}/{}/{}Dim/{}".format(os.getcwd(), data_name, analysis, f, method)
                if not os.path.exists(directory_method):
                    os.makedirs(directory_method)
                filename = '{}/{}_{}.npz'.format(directory_method, method, fold1)

            elif analysis == 'noise':
                directory_method = "{}/results/{}/{}/{}/{}/{}Dim/{}".format(
                    os.getcwd(), data_name, data_dims_name, analysis, scenario, f, method)

            elif analysis == 'sparse':
                # this must from simulation
                directory_method = "{}/results/{}/{}/{}/{}/{}Dim/{}".format(
                    os.getcwd(), data_name, data_dims_name, analysis, scenario, f, method)

                if not os.path.exists(directory_method):
                    os.makedirs(directory_method)
                filename = '{}/{}_miss{}_{}.npz'.format(directory_method, method, miss_ratio, fold1)
        elif method == "KRL":
            if analysis == "FULL":
                directory_method = "{}/results/{}/{}/{}Dim/{}".format(os.getcwd(), data_name, analysis, f, method)
                if not os.path.exists(directory_method):
                    os.makedirs(directory_method)
                filename = '{}/{}_{}{}_lambda{}_gamma{}.npz'.format(
                    directory_method, method, fold1, ('.{}'.format(fold2)) if fold2 != -1 else '', params[0], params[1])
            elif analysis == 'noise':
                directory_method = "{}/results/{}/{}/{}/{}/{}Dim/{}".format(
                    os.getcwd(), data_name, data_dims_name, analysis, scenario, f, method)

            elif analysis == 'sparse':
                # this must from simulation
                directory_method = "{}/results/{}/{}/{}/{}/{}Dim/{}".format(
                    os.getcwd(), data_name, data_dims_name, analysis, scenario, f, method)

                if not os.path.exists(directory_method):
                    os.makedirs(directory_method)
                filename = '{}/{}_miss{}_{}{}_lambda{}_gamma{}.npz'.format(
                    directory_method, method, miss_ratio, fold1, ('.{}'.format(fold2)) if fold2 != -1 else '', params[0], params[1])
    else:
        if analysis == "FULL":
            # "./results/CCLE/FULL/KRL/Debug/KRL/KRL_0"
            directory_method = "{}/results/{}/{}/{}/{}Dim/{}".format(
                os.getcwd(), data_name, analysis, "Debug", f, method)
            if not os.path.exists(directory_method):
                os.makedirs(directory_method)
            filename = '{}/{}_{}.npz'.format(directory_method, method, fold1)

    return filename


def update_metric(current_value, result, cv_metric, rank_k):
    return (current_value + Reward_utils.non_nan_mean(cv_metric(result['Y_true'], result['Y_pred'], rank_k))) \
        if result is not None and ~np.isnan(current_value) else np.nan


def get_result(
        method, analysis, data_name, fold1, f, cv_metric, rank_ks, keepk, ratio, scenario="", debug=False,
        krl_lambdas=None, krl_gammas=None, data_form="", miss_rate=None):
    results = {}
    for k in rank_ks:
        if method != "KRL":
            filename = get_result_filename(
                method, analysis, data_name, fold1, f, debug=debug, keepk=keepk, ratio=ratio, scenario=scenario,
                data_dims_name=data_form, miss_ratio=miss_rate)

            result = np.load(filename)
            results[k] = result
        elif method == "KRL":
            metric10i = np.zeros([len(krl_lambdas), len(krl_gammas)])
            for i, l in enumerate(krl_lambdas):
                for j, g in enumerate(krl_gammas):
                    filename_i = get_result_filename(method, analysis, data_name, fold1, f, scenario=scenario,
                                                     data_dims_name=data_form, miss_ratio=miss_rate, params=[l, g])

                    result_i = np.load(filename_i)

                    #full_rank = True if ratio == 1.0 else False
                    metric10i[i, j] = ndcg(result_i['Y_true'], result_i['Y_pred'], k)

            param_idx = np.where(metric10i == np.max(metric10i))
            l, g = krl_lambdas[param_idx[0][0]], krl_gammas[param_idx[1][0]]

            filename = get_result_filename(method, analysis, data_name, fold1, f, params=[l, g])
            result = np.load(filename)
            results[k] = result

    return results


def get_result_debug_temp(method, analysis, data_name, fold1, cv_metric, rank_ks, keepk, ratio, debug=False):
    results = {}
    for k in rank_ks:
        result_dir = get_result_filename(method, analysis, data_name, 0, debug=True)
        filename = result_dir[:-4]+"/rank_{}.npz".format(k)
        result = np.load(filename)
        results[k] = result
    return results


def save_exp_baselines_results(config_file, single_k=None):

    with open(config_file, 'r') as f:
        config = yaml.full_load(f)

    Debug = config['Debug']
    data_name = config['data']
    methods = config['methods']
    num_folds = config['nfolds']
    seed = config['seed']
    analysis = config['analysis']
    if data_name.startswith("GDSC"):
        keepk_ratios = np.array(config['keepk_ratios_G'], dtype=float)
        scenarios = [""]
    elif data_name == "CCLE":
        keepk_ratios = np.array(config['keepk_ratios_C'], dtype=float)
        scenarios = [""]
    elif data_name == "SimuData":
        keepk_ratios = [1.0]
        scenarios = config['scenarios']
    keepk = config['keepk']
    f = config['f']
    N = config['N']
    P = config['P']
    M = config['M']

    rank_ks = np.array(config['rank_ks'], dtype=int)

    krl_lambdas = np.array(config['krl_lambdas'], dtype=float)
    krl_gammas = np.array(config['krl_gammas'], dtype=float)
    kbmtl_alphas = np.array(config['kbmtl_alphas'], dtype=float)
    kbmtl_betas = np.array(config['kbmtl_betas'], dtype=float)
    kbmtl_gammas = np.array(config['kbmtl_gammas'], dtype=float)
    cv_metric = config['cv_metric']
    k_max = np.array(config['k_max'], dtype=int)

    miss_rate = config['miss_rate'] if config['analysis'] == 'sparse' else None
    data_form = "N{}_P{}_M{}".format(N, P, M)

    ratio_range = keepk_ratios if analysis == 'KEEPK' else np.array([1.0])
    foldwise = not (analysis == "KEEPK")
    if data_name.startswith("GDSC"):
        max_k = k_max[0]
    elif data_name == 'CCLE':
        max_k = k_max[1]
    elif data_name == 'SimuData':
        max_k = k_max[0]
    all_methods_results = {}

    # abs_ranks,relative_ranks ={},{}

    if single_k is None and config['rank_all']:
        rank_all_ks = list(rank_ks) + [max_k]
    elif single_k is None and not config['rank_all']:
        rank_all_ks = list(rank_ks)
    else:
        rank_all_ks = [single_k]

    rank_all_ks = list(set(rank_all_ks))
    for ratio in ratio_range:
        for scenario in scenarios:
            for method in methods:
                # abs_ranks[method], relative_ranks[method] = [], []

                foldwise_ndcgk = {}
                foldwise_precisionk = {}
                for fold1 in range(num_folds):

                    if Debug and method == 'ppo':
                        results = get_result_debug_temp(method, analysis, data_name, 0,
                                                        cv_metric, rank_all_ks, keepk, ratio)
                    elif method != 'KRL':
                        results = get_result(method, analysis, data_name, fold1, f, cv_metric, rank_all_ks,
                                             keepk, ratio, scenario, data_form=data_form, miss_rate=miss_rate)
                    elif method == "KRL":
                        results = get_result(
                            method, analysis, data_name, fold1, f, cv_metric, rank_all_ks, keepk, ratio, scenario,
                            krl_lambdas=krl_lambdas, krl_gammas=krl_gammas, data_form=data_form, miss_rate=miss_rate)

                    for k in rank_all_ks:
                        Y_true = results[k]['Y_true']
                        Y_pred = results[k]['Y_pred']
                        if len(Y_pred.shape) >= 3:
                            Y_pred = Y_pred.reshape(Y_pred.shape[0], Y_pred.shape[1])
                        if k not in foldwise_precisionk:
                            foldwise_precisionk[k] = []
                        if k not in foldwise_ndcgk:
                            foldwise_ndcgk[k] = []
                        # foldwise_ndcgk[k].append(ndcg(Y_true, Y_pred, k))
                        foldwise_ndcgk[k].append(NDCGk(Y_true, Y_pred, k))
                        ndcg_paper = NDCGk(Y_true, Y_pred, k)
                        ndcg_my = ndcg_tmp(Y_true, Y_pred, k)
                        print("method {} with K of {} has ndcg_paper {} and ndcg_my {}".format(
                            method, k, np.nanmean(ndcg_paper), np.nanmean(ndcg_my)))
                        foldwise_precisionk[k].append(Precision(Y_true, Y_pred, k))

                    # foldwise_ndcgk[max_k].append(ndcg(Y_true, Y_pred, max_k,full_rank=True))
                    # foldwise_precisionk[max_k] = []
                    # foldwise_precisionk[max_k].append(1.0)

                for k in rank_all_ks:
                    if foldwise:
                        if method == 'ppo':
                            ndcg_result = [np.nanmean(fold) for fold in foldwise_ndcgk[k]]
                            precision_result = [np.nanmean(fold) for fold in foldwise_precisionk[k]]
                        else:
                            ndcg_result = [np.nanmean(fold) for fold in foldwise_ndcgk[k]]
                            precision_result = [np.nanmean(fold) for fold in foldwise_precisionk[k]]
                    else:
                        ndcg_result = list(itertools.chain.from_iterable(foldwise_ndcgk[k]))
                        precision_result = list(itertools.chain.from_iterable(foldwise_precisionk[k]))
                    print(result_info(data_name, "NDCG", ndcg_result, method, k, analysis, keepk, ratio, debug=Debug))
                    print(result_info(data_name, "PRECISION", precision_result,
                                      method, k, analysis, keepk, ratio, debug=Debug))
                    if method not in all_methods_results:
                        all_methods_results[method] = {}
                    if k not in all_methods_results[method]:
                        all_methods_results[method][k] = {}
                    if ratio not in all_methods_results[method][k]:
                        all_methods_results[method][k][ratio] = {}
                    if 'NDCG' not in all_methods_results[method][k][ratio]:
                        all_methods_results[method][k][ratio]['NDCG'] = []
                    all_methods_results[method][k][ratio]['NDCG'].extend(ndcg_result)
                    if 'PRECISION' not in all_methods_results[method][k][ratio]:
                        all_methods_results[method][k][ratio]['PRECISION'] = []
                    all_methods_results[method][k][ratio]['PRECISION'].extend(precision_result)

            # if analysis == FULL, then the result_info is by foldwise, and cal foldwise mean, results is [fold1_mean,
            # fold2_mean,...foldn_mean]like
            # [data_name, analysis,keepk, ratio, metric_name, method, "rank_{}".format(rank_k),
            # "mean_{}".format(non_nan_mean(results))," ".join(map(float2str, results))]

            # But when analysis == KEEPK, results is a list of total sample size, here for ccle is 461 and for gdsc is 983

    return all_methods_results


def make_long_dictionary(results, methods=None, k_range=None, r_range=None, r_transform=None):
    long_data = {'rank_k': [], 'ratio': [], 'method': [], 'NDCG': [], "PRECISION": []}
    for method in (results if methods is None else methods):
        for rank_k in (results[method] if k_range is None else k_range):
            for ratio in (results[method][rank_k] if r_range is None else r_range):
                for ndcg, precision in zip(
                        results[method][rank_k][ratio]["NDCG"],
                        results[method][rank_k][ratio]["PRECISION"]):
                    long_data['method'].append(method)
                    long_data['rank_k'].append(rank_k)
                    long_data['ratio'].append(r_transform(ratio) if r_transform is not None else ratio)
                    long_data["NDCG"].append(ndcg)
                    long_data["PRECISION"].append(precision)
    return long_data


if __name__ == "__main__":
    all_methods_results = save_exp_baselines_results("./configs/configS_base.yaml")
    # long_data = make_long_dictionary(all_methods_results,  r_range=[1.0])
