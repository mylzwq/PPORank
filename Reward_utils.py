import numpy as np
import torch
# from sklearn.metrics import ndcg_score


def dcg_from_cur_pos(true_scores, pre_scores, filter_masks, cur_pos, k=None):
    # the default is full rank
    if k:
        assert k <= cur_pos.max()

    B, M = true_scores.shape
    dcg_cut = np.zeros(B)
    for cell in range(B):
        candidates_bool = filter_masks != float('-inf')
        drug_bool = ~np.isnan(true_scores[cell])
        candidates_s_u = true_scores[cell][candidates_bool]
        candidates_r_u = pre_scores[cell][candidates_bool]
        can_pre_rank = np.argsort(candidates_r_u)[::-1]
        if k:
            remain = k-cur_pos[cell]
            dcg_cut[cell] = np.sum((2**candidates_s_u[can_pre_rank[:remain]
                                                      ]-1)/np.log2(range(2+cur_pos[cell], 2+remain)))
        else:
            dcg_cut[cell] = np.sum((2**candidates_s_u[can_pre_rank]-1
                                    )/np.log2(range(2+cur_pos[cell], 2+remain.shape[0])))

    return dcg_cut


def ndcg_from_cur_pos(true_scores, pre_scores, filter_masks, cur_pos, k=None):
    ndcg_val_cut = dcg_from_cur_pos(true_scores, pre_scores, filter_masks, cur_pos, k=k)\
        / dcg_from_cur_pos(true_scores, true_scores, filter_masks, cur_pos, k=k)
    return np.mean(ndcg_val_cut)


def cal_immediate_reward(scores, pre_ranks):
    nominators = 2**scores-1
    reward = nominators/np.log2(pre_ranks+2)
    return reward


def dcg_general(true_scores, pre_scores, k, full_rank=False):
    dcg = np.zeros(true_scores.shape[0])
    # true_scores=np.asarray(true_scores)
    # pre_scores=np.asarray(pre_scores)
    for cell in range(true_scores.shape[0]):

        test_drug_bool = ~np.isnan(true_scores[cell])
        if np.sum(test_drug_bool) == 0:
            continue
        s_u = true_scores[cell][test_drug_bool]
        r_u = pre_scores[cell][test_drug_bool]

        pre_rank = np.argsort(r_u)[::-1]
        # if k > pre_rank.shape[0]:
        #     k = pre_rank.shape[0]
        k_new = min(k, s_u.shape[0])
        if not full_rank:
            dcg[cell] = np.sum((2**s_u[pre_rank[:k_new]]-1)/np.log2(range(2, 2+k_new)))
        else:
            dcg[cell] = np.sum((2**s_u[pre_rank]-1)/np.log2(range(2, 2+pre_rank.shape[0])))
    return dcg


def ndcg(true_scores, pre_scores, k, full_rank=False):
    dcg = dcg_general(true_scores, pre_scores, k, full_rank=full_rank)
    idcg = dcg_general(true_scores, true_scores, k, full_rank=full_rank)
    ndcg_val = dcg/idcg
    idcg_mask = idcg == 0
    ndcg_val[idcg_mask] = np.nan  # if idcg == 0 , set ndcg to 0

    assert (ndcg_val < 0.0).sum() >= 0, "every ndcg should be non-negative"
    return ndcg_val


def ndcg_tmp(true_scores, pre_scores, k, full_rank=False):
    dcg = dcg_general(true_scores, pre_scores, k, full_rank=full_rank)
    idcg = dcg_general(true_scores, true_scores, k, full_rank=full_rank)
    ndcg_val = dcg/idcg
    idcg_mask = idcg == 0
    ndcg_val[idcg_mask] = np.nan  # if idcg == 0 , set ndcg to 0

    assert (ndcg_val < 0.0).sum() >= 0, "every ndcg should be non-negative"
    return ndcg_val


def dcg_general_Rec(true_scores, pre_scores, k, full_rank=False):
    dcg = np.zeros(true_scores.shape[0])
    # true_scores=np.asarray(true_scores)
    # pre_scores=np.asarray(pre_scores)
    for cell in range(true_scores.shape[0]):

        test_drug_bool = ~np.isnan(true_scores[cell])[0]
        s_u = np.asarray(true_scores[cell][test_drug_bool])[0]
        r_u = np.asarray(pre_scores[cell][test_drug_bool])[0]
        pre_rank = np.argsort(r_u)[::-1]
        if k > pre_rank.shape[0]:
            k = pre_rank.shape[0]
        if not full_rank:
            dcg[cell] = np.sum((2**s_u[pre_rank[:k]]-1)/np.log2(range(2, 2+k)))
        else:
            dcg[cell] = np.sum((2**s_u[pre_rank]-1)/np.log2(range(2, 2+pre_rank.shape[0])))
    return dcg


def ndcg_Rec(true_scores, pre_scores, k=5, full_rank=True):

    ndcg_val = dcg_general_Rec(true_scores, pre_scores, k,
                               full_rank=full_rank)/dcg_general_Rec(true_scores, true_scores, k, full_rank=full_rank)
    return np.mean(ndcg_val)


def discount_sum_rewards(discount, rewards_mat, drug_num_mat, max_drug_num):
    """
        returns: q_n_mat: [N,M], MC estimation for reward_to_go matrix
    """
    N = rewards_mat.size()[0]
    q_n_mat = torch.zeros([N, max_drug_num])
    q_n_mat_normal = torch.zeros([N, max_drug_num])
    for cell in range(N):
        re_n = rewards_mat[cell]
        q_n = discount_reward(discount, re_n, drug_num_mat[cell], max_drug_num)
        q_n_mat[cell] = torch.Tensor(q_n)
        mean_q_n = torch.sum(q_n_mat[cell])/drug_num_mat[cell]
        q_n_mat_normal[cell][:drug_num_mat[cell]] = torch.Tensor(q_n[:drug_num_mat[cell]]-mean_q_n.data.item())
    return q_n_mat_normal


def discount_reward(discount, one_cell_rewards, drug_num, max_drug_num):
    q_n = np.zeros(max_drug_num)
    q_n[drug_num-1] = one_cell_rewards[drug_num-1]
    # q_n[-max_drug_num+drug_num-1]=one_cell_rewards[-max_drug_num+drug_num-1]
    for t in range(drug_num-2, -1, -1):
        q_n[t] = one_cell_rewards[t]+discount*q_n[t+1]
    return q_n


def precision(y, pred, k):
    return (1.0 * np.intersect1d(np.argsort(y)[::-1][:k], np.argsort(pred)[::-1][:k]).shape[0] / k) if k > 0 else np.nan


def Precision(true_scores, pre_scores, k):
    n = true_scores.shape[0]
    precisionk = []
    for i in range(n):
        f = pre_scores[i]
        y = true_scores[i]
        f = f[~np.isnan(y)]
        y = y[~np.isnan(y)]
        precisionk.append(precision(y, f, min(k, y.shape[0])))
    return np.array(precisionk)

# Same as ndcg, two methods,same results


def score_to_exact_rank(s):
    return (-1*s).argsort().argsort()


def cal_exact_avg_ndcg(pred, R):
    all_ndcg = []
    for u in range(R.shape[0]):  # R.shape=(n,m), for each cell-line iteration
        test_drug_bool = ~np.isnan(R[u, :])
        s_u = R[u, :][test_drug_bool]
        r_u = score_to_exact_rank(s_u)
        s_u_pred = pred[u, :][test_drug_bool]
        r_u_pred = score_to_exact_rank(s_u_pred)
        G_u_max = np.sum((np.power(2, s_u)-1) / np.log(r_u + 2))
        G_u = np.sum((np.power(2, s_u)-1) / np.log(r_u_pred + 2))
        # print G_u, G_u_max, G_u / G_u_max
        if np.isnan(G_u_max) or G_u_max == 0.:
            all_ndcg += [0.]
        else:
            all_ndcg += [G_u / G_u_max]
    return np.nanmean(all_ndcg)


def NDCGk(Y, F, k):
    # F is the predicted scores
    # also calculate cell by cell
    n = Y.shape[0]
    ndcgk = []
    for i in range(n):
        f = F[i]
        y = Y[i]
        f = f[~np.isnan(y)]
        y = y[~np.isnan(y)]
        ndcgk.append(ndcg_single(y, np.argsort(f)[::-1], min(k, y.shape[0])))
    return np.array(ndcgk)


def ndcg_single(y, pi, k):
    return dcg_single(y, pi, k) / dcg_single(y, np.argsort(y)[::-1], k) if k > 0 else np.nan


def dcg_single(y, pi, k):
    return ((2**y[pi[:k]]-1) / np.log2(range(2, 2+k))).sum() if k > 0 else np.nan


def MSEloss(y_pred, y_true):
    padded_mask = torch.isnan(y_true)
    N = torch.sum(~torch.isnan((y_true)))
    y_true[padded_mask] = 0.0
    y_pred[padded_mask] = 0.0
    mse = torch.nn.MSELoss(reduction="sum")

    return mse(y_pred, y_true)/N


def approxNDCGLoss(y_pred, y_true, device, padded_value_indicator=-1, eps=1e-9, alpha=10.):
    """
    Loss based on approximate NDCG introduced in "A General Approximation Framework for Direct Optimization of
    Information Retrieval Measures". Please note that this method does not implement any kind of truncation.
    :param y_pred: predictions from the model, shape [batch_size, slate_length]
    :param y_true: ground truth labels, shape [batch_size, slate_length]
    :param eps: epsilon value, used for numerical stability
    :param padded_value_indicator: an indicator of the y_true index containing a padded item, e.g. -1
    :param alpha: score difference weight used in the sigmoid function
    :return: loss value, a torch.Tensor
    """
    # device = y_pred.device
    # will not change the original input
    y_pred = y_pred.clone()
    y_true = y_true.clone()

    padded_mask = torch.isnan(y_true)

    # padded_mask = y_true == padded_value_indicator
    y_pred[padded_mask] = float("-inf")
    y_true[padded_mask] = float("-inf")

    # Here we sort the true and predicted relevancy scores.
    y_pred_sorted, indices_pred = y_pred.sort(descending=True, dim=-1)
    y_true_sorted, _ = y_true.sort(descending=True, dim=-1)

    # After sorting, we can mask out the pairs of indices (i, j) containing index of a padded element.
    true_sorted_by_preds = torch.gather(y_true, dim=1, index=indices_pred)
    true_diffs = true_sorted_by_preds[:, :, None] - true_sorted_by_preds[:, None, :]
    padded_pairs_mask = torch.isfinite(true_diffs)
    padded_pairs_mask.diagonal(dim1=-2, dim2=-1).zero_()

    # Here we clamp the -infs to get correct gains and ideal DCGs (maxDCGs)
    true_sorted_by_preds.clamp_(min=0.)
    y_true_sorted.clamp_(min=0.)

    # Here we find the gains, discounts and ideal DCGs per slate.
    pos_idxs = torch.arange(1, y_pred.shape[1] + 1).to(device)
    D = torch.log2(1. + pos_idxs.float())[None, :]
    maxDCGs = torch.sum((torch.pow(2, y_true_sorted) - 1) / D, dim=-1).clamp(min=eps)
    G = (torch.pow(2, true_sorted_by_preds) - 1) / maxDCGs[:, None]

    # Here we approximate the ranking positions according to Eqs 19-20 and later approximate NDCG (Eq 21)
    scores_diffs = (y_pred_sorted[:, :, None] - y_pred_sorted[:, None, :])
    scores_diffs[~padded_pairs_mask] = 0.
    approx_pos = 1. + torch.sum(padded_pairs_mask.float()
                                * (torch.sigmoid(-alpha * scores_diffs).clamp(min=eps)), dim=-1)
    approx_D = torch.log2(1. + approx_pos)
    approx_NDCG = torch.sum((G / approx_D), dim=-1)

    return -torch.mean(approx_NDCG)
