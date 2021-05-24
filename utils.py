import os
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances
import torch
import logging
import shutil
import pickle
import gzip
import math
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
from _testcapi import raise_exception


def check_gpu():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    # Additional Info when using cuda
    if device.type == 'cuda':
        print(torch.cuda.get_device_name(0))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3, 1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_cached(0)/1024**3, 1), 'GB')


def save_pytorch_model(save_dir, model):
    """
    Saves the entire pytorch Module
    """
    torch.save(model, os.path.join(save_dir, "model.pkl"))


def read_PROP(data_name, prop, CV="CV", i=0, j=None):
    """
    Read proportion of data, used in simulation,

    """

    data_dir = os.path.join(os.getcwd(), data_name, CV, "FULL", "Fold{}".format(i))

    if not j:
        if data_name != "GDSC_ALL":
            # only with train and test data
            train_data = np.load(data_dir+"/FulltrainDf.npz")
            Xtrain = train_data["Xtrain"]
            Ytrain = train_data["Ytrain"]
            test_data = np.load(data_dir+"/FulltestDf.npz")
            Xtest = test_data['Xtest']
            Ytest = test_data['Ytest']
        elif data_name == "GDSC_ALL":

            Ytrain_full = pd.read_csv(os.path.join(data_dir, "YtrainDf.csv"), index_col=0)
            Ytest = pd.read_csv(os.path.join(data_dir, "YtestDf.csv"), index_col=0)
            Xtrain_full = pd.read_csv(os.path.join(data_dir, "Xtrain_rawDf.csv"), index_col=0)
            Xtest = pd.read_csv(os.path.join(data_dir, "Xtest_rawDf.csv"), index_col=0)
            print("for GDSC ALL we are using  {} essentail genes".format(Xtrain_full.shape[1]))

            N, P = Xtrain_full.shape
            train_num = math.floor(N*prop)
            inds = np.random.choice(N, train_num)
            Xtrain = Xtrain_full.iloc[inds]
            Ytrain = Ytrain_full.iloc[inds]
    else:
        # train and val data
        train_data = np.load(data_dir + "/trainDf_fold{}.npz".format(j))
        Xtrain = train_data["Xtrain"]
        Ytrain = train_data["Ytrain"]
        test_data = np.load(data_dir + "/valDf_fold{}.npz".format(j))
        Xtest = test_data['Xval']
        Ytest = test_data['Yval']
    return Xtrain, Xtest, Ytrain, Ytest


def read_FULL(data_name, CV="CV", i=0, j=None):
    """
    Read full set of data,
    if j is None using training and val data, returns are all numpy array
    """

    data_dir = os.path.join(os.getcwd(), data_name, CV, "FULL", "Fold{}".format(i))

    if not j:
        if data_name != "GDSC_ALL":
            # only with train and test data
            train_data = np.load(data_dir+"/FulltrainDf.npz")
            Xtrain = train_data["Xtrain"]
            Ytrain = train_data["Ytrain"]
            test_data = np.load(data_dir+"/FulltestDf.npz")
            Xtest = test_data['Xtest']
            Ytest = test_data['Ytest']
        elif data_name == "GDSC_ALL":

            Ytrain = pd.read_csv(os.path.join(data_dir, "YtrainDf.csv"), index_col=0)
            Ytest = pd.read_csv(os.path.join(data_dir, "YtestDf.csv"), index_col=0)
            Xtrain = pd.read_csv(os.path.join(data_dir, "Xtrain_rawDf.csv"), index_col=0)
            Xtest = pd.read_csv(os.path.join(data_dir, "Xtest_rawDf.csv"), index_col=0)
            print("for GDSC ALL we are using  essentail genes")
    else:
        # train and val data
        train_data = np.load(data_dir + "/trainDf_fold{}.npz".format(j))
        Xtrain = train_data["Xtrain"]
        Ytrain = train_data["Ytrain"]
        test_data = np.load(data_dir + "/valDf_fold{}.npz".format(j))
        Xtest = test_data['Xval']
        Ytest = test_data['Yval']
    return Xtrain, Xtest, Ytrain, Ytest


def open_file(filename, mode='r', compresslevel=9):
    if mode not in ['r', 'rb', 'a', 'ab', 'w', 'wb']:
        raise_exception('file mode not supported:', mode)
    if filename.endswith('.gz') or (not os.path.exists(filename) and os.path.exists(filename + '.gz')):
        # gzip automatically adds 'b' to the 'r', 'a', and 'w' modes
        return gzip.open(filename if filename.endswith('.gz') else filename + '.gz', mode, compresslevel)
    else:
        return open(filename, mode)


def line_split_gen(filename, delim='\t', strip='\n', comment='#', skip_rows=0, skip_columns=0):
    with open_file(filename) as f:
        for _ in range(skip_rows):
            f.readline()
        for line in f:
            if comment is not None and line.startswith(comment):
                continue
            line_split = line.strip(strip).split(delim)
            yield line_split[skip_columns:]


def cm_to_inch(value):
    return float(value) / 2.54


def non_nan_statistic(array_like, statistic):
    array_like = np.array(array_like)
    return statistic(array_like[~np.isnan(array_like)])


def non_nan_mean(array_like):
    return non_nan_statistic(array_like, np.mean)


def non_nan_std(array_like):
    return non_nan_statistic(array_like, np.std)


class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.tqdm.write(msg)
            self.flush()
        except (KeyboardInterrupt, SystemExit):
            raise
        except:

            self.handleError(record)


class AverageMeter():
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.count = 0
        self.total = 0
        self.avg = 0

    def update(self, val, n=1):
        self.val = val
        self.total += val * n
        self.count += n
        self.avg = self.total / self.count


def save_checkpoint(state, is_best, saved_dir, filename='checkpoint.pth.tar'):
    torch.save(state, saved_dir+filename)
    if is_best:
        shutil.copy(saved_dir+filename, saved_dir+"/model_best.pth.tar")


def create_save_dir(save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)


def create_model_name(args, ppo=False):

    # if args.analysis == "FULL" and args.fold:
    #     data_dir = os.path.join(data_dir, args.CV_dir, args.analysis, args.fold)
    #     # logger.info("Data {} with analysis {} training with CV folder {}".format(
    #         args.Data, args.analysis, int(args.fold[-1])))
    # elif args.analysis == "KEEPK" and args.fold:
    #     keep_dir="_".join(['keep', str(args.keepk), "kr", str(args.keepk_ratio)])
    #     data_dir=os.path.join(data_dir, args.CV_dir, args.analysis, str(keep_dir), args.fold)
    #     # logger.info("Data {} with analysis {} training with CV folder {}/{}".format(args.Data,
    #                                                                                 args.analysis, keep_dir, args.fold))
    # elif args.analysis == "noise" or args.analysis == 'sparse':
    #     data_dir=os.path.join(
    #         data_dir, "N{}_P{}_M{}".format(args.simu_N, args.simu_P, args.simu_M),
    #         args.scenario, args.CV_dir, args.analysis, args.fold)
    # logger.info("Data {} with scenario {} analysis {} training with CV folder {}".format(
    # args.Data, args.scenario, args.analysis, int(args.fold[-1])))

    if args.analysis == 'KEEPK':
        keepk = args.keepk
    elif args.analysis == 'FULL':
        keepk = ""
    else:
        keepk = args.scenario

    miss_rate = args.miss_rate if args.analysis == "sparse" else ""
    scale = args.scale if args.normalize_y else "raw"

    if not ppo:
        model_name = "_".join(("{}".format(args.algo),
                               "{}".format(args.Data),
                               "{}".format(args.analysis),
                               "f{}".format(args.f),
                               "{}".format(keepk),
                               "S{}".format(scale),  # ('.{}'.format(fold2)) if fold2 != -1 else ''
                               "B{}".format(args.num_processes),
                               "D{}".format(args.nlayers_deep),
                               "C{}".format(args.nlayers_cross),
                               "{}".format(args.fold),
                               "miss{}".format(miss_rate)
                               ))
    else:
        model_name = "_".join(("{}".format(args.algo),
                               "{}".format(args.Data),
                               "{}".format(args.analysis),
                               "{}Dim".format(args.f),
                               "{}".format(keepk),
                               "{}scale".format(scale),
                               "VF{}".format(args.value_loss_coef),
                               "B{}".format(args.num_processes),
                               "gam{}".format(args.gamma),
                               "la{}".format(args.gae_lambda),
                               "{}D".format(args.nlayers_deep),
                               "{}C".format(args.nlayers_cross),
                               "{}".format(args.fold),
                               "miss{}".format(miss_rate)
                               ))

    if not args.pretrain:
        model_name = model_name + "noPretrain"

    return model_name


def create_train_data_dir(data_dir, args):
    if args.analysis == "FULL" and args.fold:
        data_dir = os.path.join(data_dir, args.CV_dir, args.analysis, args.fold)
        # logger.info("Data {} with analysis {} training with CV folder {}".format(
        #     args.Data, args.analysis, int(args.fold[-1])))
    elif args.analysis == "KEEPK" and args.fold:
        keep_dir = "_".join(['keep', str(args.keepk), "kr", str(args.keepk_ratio)])
        data_dir = os.path.join(data_dir, args.CV_dir, args.analysis, str(keep_dir), args.fold)
        # logger.info("Data {} with analysis {} training with CV folder {}/{}".format(args.Data,
        #                                                                             args.analysis, keep_dir, args.fold))
    elif args.analysis == "noise" or args.analysis == 'sparse':
        data_dir = os.path.join(
            data_dir, "N{}_P{}_M{}".format(args.simu_N, args.simu_P, args.simu_M),
            args.scenario, args.CV_dir, args.analysis, args.fold)
        # logger.info("Data {} with scenario {} analysis {} training with CV folder {}".format(
        #     args.Data, args.scenario, args.analysis, int(args.fold[-1])))
    return data_dir


def generate_k(analysis, keepk_ratio, keepk, scenario, k):
    if analysis == 'KEEPK':
        kr = keepk_ratio
        k = keepk
    else:
        kr = 1.0

    if analysis == 'KEEPK':
        keepk = keepk
    elif analysis == 'FULL':
        keepk = ""
    else:
        keepk = scenario
    return k, kr, keepk


def compare_models_by_metric(model_1, model_2, model_hist_1, model_hist_2, metric, epochs):
    '''
    Function to compare a metric between two models

    Parameters:
        model_hist_1 : training history of model 1
        model_hist_2 : training history of model 2
        metrix : metric to compare, loss, acc, val_loss or val_acc

    Output:
        plot of metrics of both models
    '''
    metric_model_1 = model_hist_1.history[metric]
    metric_model_2 = model_hist_2.history[metric]
    e = range(1, epochs)

    metrics_dict = {
        'ndcg': 'Training NDCG',
        'loss': 'Training Loss',
        'val_acc': 'Validation accuracy',
        'val_loss': 'Validation loss'
    }

    metric_label = metrics_dict[metric]
    plt.plot(e, metric_model_1, 'bo', label=model_1.name)
    plt.plot(e, metric_model_2, 'b', label=model_2.name)
    plt.xlabel('Epoch number')
    plt.ylabel(metric_label)
    plt.title('Comparing ' + metric_label + ' between models')
    plt.legend()
    plt.show()


def update_linear_schedule(optimizer, epoch, total_num_epochs, initial_lr):
    """Decreases the learning rate linearly"""
    lr = initial_lr - (initial_lr * (epoch / float(total_num_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def precision(y, f, k):
    return (1.0 * np.intersect1d(np.argsort(y)[::-1][:k], np.argsort(f)[::-1][:k]).shape[0] / k) if k > 0 else np.nan


def Precision(Y, F, k):
    n = Y.shape[0]
    precisionk = []
    for i in range(n):
        f = F[i]
        y = Y[i]
        f = f[~np.isnan(y)]
        y = y[~np.isnan(y)]
        precisionk.append(precision(y, f, min(k, y.shape[0])))
    return np.array(precisionk)


def rank(pool, best, which=0):
    assert which >= 0
    return list(np.argsort(pool)[::-1]).index(np.argsort(best)[::-1][which]) if pool.shape[0] > 0 else np.nan


def Rank(Y, F, rank_type):
    assert rank_type in ['best_drug', 'best_prediction']
    n = Y.shape[0]
    ranks = []
    for i in range(n):
        y = Y[i]
        f = F[i]
        f = f[~np.isnan(y)]
        y = y[~np.isnan(y)]
        ranks.append(rank(f, y) if rank_type == 'best_drug' else rank(y, f))
    return np.array(ranks)


def percentile(y, f, which):
    assert which >= 0
    return (rank(y, f, which) / float(y.shape[0])) if (y.shape[0] > 0 and y.shape[0] > which) else np.nan


def Percentile(Y, F, k):
    n = Y.shape[0]
    percentiles = []
    for i in range(n):
        y = Y[i]
        f = F[i]
        f = f[~np.isnan(y)]
        y = y[~np.isnan(y)]
        percentiles.append([percentile(y, f, which) for which in range(k)])
    return np.array(percentiles)


def average_rank(Y, F, k):
    n = Y.shape[0]
    ar = []
    for i in range(n):
        f = F[i]
        y = Y[i]
        pi = np.argsort(f)[::-1]
        y_pi = y[pi]
        tmp = np.where(y_pi == k)[0]
        ar.extend(tmp)
    return np.array(ar)


if __name__ == "__main__":
    X, Y, Y_norm = load_data_simulator("./Simu_Data", 1000, 20, 10, 5)

#    input,output,output_norm=load_data_simulator("./Simu_Data",2000,50)
