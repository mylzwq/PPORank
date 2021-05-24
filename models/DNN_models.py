from __future__ import print_function
import sys
#from models.Policy import build_mlp, CrossLayer
from sklearn.preprocessing import StandardScaler
from torch.nn.utils import clip_grad_norm_
import torch.optim as optim
from torch.nn.parameter import Parameter
import logging
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
import os
import numpy as np
import Reward_utils
import torch.nn as nn
from torch.nn import Sequential
import argparse
import torch
import utils
from utils import AverageMeter, TqdmLoggingHandler
# sys.path.insert(0, "..")


logger = logging.getLogger(__name__)


def build_mlp(input_size, output_size, n_layers, hidden_sizes, dropout_rate=0.5):
    """
    returns:an instance of nn.Sequential which contains the feedforward neural network
    """
    layers = []
    for i in range(n_layers):
        layers += [nn.Linear(input_size, hidden_sizes[i]), nn.ReLU(True), nn.Dropout(p=dropout_rate)]
        input_size = hidden_sizes[i]
    layers += [nn.Linear(hidden_sizes[-1], output_size)]
    return nn.Sequential(*layers).apply(weights_init)


# def weights_init(m):
#     if hasattr(m, 'weight'):
#         torch.nn.init.xavier_uniform_(m.weight)
#     if hasattr(m, 'bias'):
#         m.bias.data.fill_(0)


def weights_init(m):
    if not hasattr(m, 'affine'):
        if hasattr(m, 'weight'):
            nn.init.orthogonal_(m.weight)


class CrossNet(nn.Module):
    """
     Cross net used in dnn, different initialization as in RL cross net
      Input shape
        - 2D tensor with shape: ``(batch_size, units)``.
      Output shape
        - 2D tensor with shape: ``(batch_size, units)``.
      Arguments
        - **input_feature_num**: Positive integer, shape(Input tensor)[-1]
        - **layer_num**: Positive integer, the cross layer number
        # - **l2_reg**: float between 0 and 1. L2 regularizer strength applied to the kernel weights matrix
    """

    def __init__(self, input_feature_num, layer_num=2, drug_mean=0.0):
        super(CrossNet, self).__init__()
        self.layer_num = layer_num
        # self.BN = torch.nn.LayerNorm(input_feature_num)
        self.kernels = torch.nn.ParameterList(
            [nn.Parameter(nn.init.xavier_normal_(torch.empty(input_feature_num, 1))) for i in range(self.layer_num)])
        self.bias = torch.nn.ParameterList(
            [nn.Parameter(nn.init.zeros_(torch.empty(input_feature_num, 1))) for i in range(self.layer_num)])

    def forward(self, inputs):

        # x_0 = self.BN(inputs)
        x_0 = inputs
        x_l = x_0
        for i in range(self.layer_num):
            xl_w = x_0*torch.mm(x_l, self.kernels[i])
            # xl_w = torch.tensordot(x_l, self.kernels[i], dims=([1], [0]))
            # dot_ = torch.matmul(x_0, xl_w)
            x_l = xl_w + self.bias[i].squeeze(-1) + x_l
            # x_l = self.BN(x_l)
            # x_l = nn.Dropout(0.5)
            # self.kernels[i] = self.kernels[i]/torch.norm(self.kernels[i])
            # self.bias[i] = self.bias[i]/torch.norm(self.bias[i])
        # x_l = torch.squeeze(x_l, dim=2)
        return x_l


def DeepNet(input_size, output_size, n_layers, hidden_sizes, dropout_rates=0.8):
    layers = []

    for i in range(n_layers):
        layers += [nn.Linear(input_size, hidden_sizes[i], bias=False), nn.ReLU(True), nn.Dropout(p=dropout_rates)]
        input_size = hidden_sizes[i]
    if len(hidden_sizes) > 2:
        layers += [nn.BatchNorm1d(hidden_sizes[-2])]
    layers += [nn.Dropout(p=0.5)]
    layers += [nn.Linear(hidden_sizes[-1], output_size, bias=False)]
    return nn.Sequential(*layers).apply(weights_init)


class CrossLayer(nn.Module):
    def __init__(self, num_input):
        super(CrossLayer, self).__init__()
        self.num_input = num_input
        self.weight = Parameter(torch.Tensor(num_input))
        self.weight.data.uniform_(-1, 1)
        self.bias = Parameter(torch.Tensor(num_input))
        self.bias.data.uniform_(-1, 1)

    def forward(self, x_0, x_l):
        return x_0*torch.mm(x_l, self.weight.unsqueeze(1)) + self.bias + x_l


class RLCrossNet(nn.Module):
    """
     Cross net used in dnn, different initialization as in RL cross net

    """

    def __init__(self, input_feature_num, cross_layer_num=2):
        super(RLCrossNet, self).__init__()
        self.layer_num = cross_layer_num
        self.kernels = torch.nn.ParameterList(
            [nn.Parameter(nn.init.orthogonal_(torch.empty(input_feature_num, 1))) for i in range(self.layer_num)])
        self.bias = torch.nn.ParameterList(
            [nn.Parameter(nn.init.zeros_(torch.empty(input_feature_num, 1))) for i in range(self.layer_num)])
        # self.to(device)

    def forward(self, inputs):
        x_0 = inputs.unsqueeze(1)
        x_l = x_0
        for i in range(self.layer_num):
            xl_w = torch.tensordot(x_l, self.kernels[i], dims=([1], [0]))
            dot_ = torch.matmul(x_0, xl_w)
            x_l = dot_ + self.bias[i] + x_l
        x_l = torch.squeeze(x_l, dim=2)
        return x_l


class PG_Rank_NN_Cross(nn.Module):
    def __init__(self, N, M, drug_dim, drug_pretrained_weight, cell_dim, cell_pretrained_weight,
                 cross_layers, deep_layers, hidden_sizes, deep_out_size,
                 train_cell=False, train_drug=False):
        super(PG_Rank_NN_Cross, self).__init__()
        self.N = N
        self.M = M
        self.drug_dim = drug_dim
        self.cell_dim = cell_dim

        self.drug_embedding = nn.Embedding(M, drug_dim, max_norm=1)
        self.drug_embedding.weight.data.copy_(drug_pretrained_weight)
        self.drug_embedding.weight.requires_grad = train_drug

        self.cell_embedding = nn.Embedding(N, cell_dim, max_norm=1)
        self.cell_embedding.weight.data.copy_(cell_pretrained_weight)
        self.cell_embedding.weight.requires_grad = train_cell

        self.total_dim = drug_dim+cell_dim+1

        self.deep_classifier = build_mlp(self.total_dim, deep_out_size, deep_layers, hidden_sizes)

        #self.cross_classifier = build_cross(self.total_dim,cross_layers)
        self.classifierC0 = CrossLayer(self.total_dim)
        self.classifierC1 = CrossLayer(self.total_dim)

        out_dim = deep_out_size + self.total_dim + 1  # 1 is for cosine similarity
        self.classifierF = nn.Linear(out_dim, 1, bias=True)

    def forward(self, input):
        B = input.size()[0]
        cell_index = input[:, 0]
        drug_index = input[:, 1:]
        cell_emb = self.cell_embedding(cell_index).view(B, 1, self.cell_dim)  # (N,1,P2)
        cell_embs = cell_emb.repeat(1, self.M, 1)  # (N,M,P2)

        drug_emb = self.drug_embedding(drug_index)  # (N,M,P1)

        cos1 = (cell_embs * drug_emb).sum(2).view(B, self.M, 1)
        x0 = torch.cat((cell_embs, drug_emb, cos1), 2)  # (N,M,P1+P2+1)

        D_in = self.deep_classifier(x0)  # 64 (N,M,64)
        total_dim = self.cell_dim + self.drug_dim+1

        x0 = x0.view(-1, total_dim)
        C0_out = self.classifierC0(x0, x0)
        C1_out = self.classifierC1(x0, C0_out)

        C_out = C1_out.view(B, self.M, -1)  # (N,M,P)

        in_final = torch.cat((D_in, C_out, cos1), 2)
        out = self.classifierF(in_final)
        return out


def CellNet(input_sizes, output_size, WP_pretrained=None):
    if len(input_sizes) == 2:  # (N,P)
        N, P = input_sizes
        if WP_pretrained:
            layers = nn.Linear(P, output_size, bias=False)
        else:
            layers = nn.Sequential(
                nn.Linear(P, 256, bias=False),
                # nn.ReLU(True),
                nn.Tanh(),
                nn.BatchNorm1d(256),
                nn.Dropout(p=0.8),
                nn.Linear(256, output_size, bias=True),
                nn.BatchNorm1d(output_size),
                # nn.ReLU(True)
            )
        return layers.apply(weights_init)
    elif len(input_sizes) == 3:
        N, M, P = input_sizes
        if WP_pretrained is not None:
            layers = nn.Linear(P, output_size, bias=False)
        else:
            layers = nn.Sequential(
                nn.Linear(P, 256, bias=False),
                # nn.ReLU(True),
                nn.Tanh(),
                nn.BatchNorm1d(M),
                nn.Dropout(p=0.8),
                nn.Linear(256, output_size, bias=True),
                nn.BatchNorm1d(M),
                # nn.ReLU(True)
            )
        return layers.apply(weights_init)


class CrossNet(nn.Module):
    """
     Cross net used in dnn, different initialization as in RL cross net
      Input shape
        - 2D tensor with shape: ``(batch_size, units)``.
      Output shape
        - 2D tensor with shape: ``(batch_size, units)``.
      Arguments
        - **input_feature_num**: Positive integer, shape(Input tensor)[-1]
        - **layer_num**: Positive integer, the cross layer number
        # - **l2_reg**: float between 0 and 1. L2 regularizer strength applied to the kernel weights matrix
    """

    def __init__(self, input_feature_num, layer_num=2, drug_mean=0.0):
        super(CrossNet, self).__init__()
        self.layer_num = layer_num
        # self.BN = torch.nn.LayerNorm(input_feature_num)
        self.kernels = torch.nn.ParameterList(
            [nn.Parameter(nn.init.xavier_normal_(torch.empty(input_feature_num, 1))) for i in range(self.layer_num)])
        self.bias = torch.nn.ParameterList(
            [nn.Parameter(nn.init.zeros_(torch.empty(input_feature_num, 1))) for i in range(self.layer_num)])
        # self.bias.require_grad=False
        # self.to(device)
        # self.add_regularization_loss(
        #     filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.dnn.named_parameters()), l2_reg_dnn)
        # self.add_regularization_loss(self.dnn_linear.weight, l2_reg_linear)
        # self.add_regularization_loss(self.crossnet.kernels, l2_reg_cross)

    def forward(self, inputs):

        # x_0 = self.BN(inputs)
        x_0 = inputs
        x_l = x_0
        for i in range(self.layer_num):
            xl_w = x_0*torch.mm(x_l, self.kernels[i])
            # xl_w = torch.tensordot(x_l, self.kernels[i], dims=([1], [0]))
            # dot_ = torch.matmul(x_0, xl_w)
            x_l = xl_w + self.bias[i].squeeze(-1) + x_l
            # x_l = self.BN(x_l)
            # x_l = nn.Dropout(0.5)
            # self.kernels[i] = self.kernels[i]/torch.norm(self.kernels[i])
            # self.bias[i] = self.bias[i]/torch.norm(self.bias[i])
        # x_l = torch.squeeze(x_l, dim=2)
        return x_l


# def DeepNet(input_size, output_size, n_layers, hidden_sizes, dropout_rates=0.8):
#     layers = []

#     for i in range(n_layers):
#         layers += [nn.Linear(input_size, hidden_sizes[i], bias=False), nn.ReLU(True), nn.Dropout(p=dropout_rates)]
#         input_size = hidden_sizes[i]
#     layers += [nn.BatchNorm1d(hidden_sizes[-1])]
#     layers += [nn.Dropout(p=dropout_rates)]
#     layers += [nn.Linear(hidden_sizes[-1], output_size, bias=False)]
#     return nn.Sequential(*layers).apply(weights_init)


# def cellNet(input_size, output_size):

#     layers = nn.Sequential(
#         nn.Linear(input_size, 256, bias=False),
#         # nn.ReLU(True),
#         nn.Tanh(),
#         nn.BatchNorm1d(256),
#         nn.Dropout(p=0.8),
#         nn.Linear(256, output_size, bias=True),
#         nn.BatchNorm1d(output_size),
#         # nn.ReLU(True)
#     )
#     return layers.apply(weights_init)


class LinearModel(nn.Module):
    def __init__(self, N, M, gene_dim, drug_dim, cell_dim, nlayers_cross, nlayers_deep,
                 hidden_sizes, deep_out_size, drug_embedding=None,
                 cell_WPmat=None, train_cell=False, train_drug=False):
        super(LinearModel, self).__init__()
        self.N = N
        self.M = M
        self.drug_dim = drug_dim
        self.cell_dim = cell_dim
        self.gene_dim = gene_dim

        self.nlayers_cross = nlayers_cross
        self.nlayers_deep = nlayers_deep

        self.drug_embedding = nn.Embedding(M, drug_dim, max_norm=1).double()
        nn.init.orthogonal_(self.drug_embedding.weight.data)

        self.cell_layer = nn.Linear(self.gene_dim, self.cell_dim).double()

        self.drug_embedding.weight.requires_grad = True
        nn.init.orthogonal_(self.cell_layer.weight.data)

        self.cell_layer.weight.requires_grad = True

    def forward(self, input):
        # input:[B,M,cell_dim+1]
        B = input.size()[0]
        cell_fts = input.clone()[:, :, :-1]  # direct slice is sharing memory, if cell_fts changes,input also changes
        drug_index = input.clone()[:, :, -1].long()

        cell_emb = self.cell_layer(cell_fts)  # .to(self.device)#(N,M,P1)
        drug_emb = self.drug_embedding(drug_index)  # .to(self.device) #(N,M,P2)

        # cos1 = (cell_emb * drug_emb).sum(2).view(B, self.M, 1)
        cos_sim = nn.CosineSimilarity(dim=2, eps=1e-6)
        cos1 = cos_sim(cell_emb, drug_emb)
        cos1 = cos1.view(B, self.M, 1)

        return cos1

    # return nn.Sequential(*layers).apply(weights_init)


class DeepCrossModel(nn.Module):
    def __init__(self, N, M, gene_dim, drug_dim, cell_dim, nlayers_cross, nlayers_deep,
                 hidden_sizes, deep_out_size, drug_embedding=None,
                 cell_WPmat=None, train_cell=False, train_drug=False, dropout_rates=0.5,
                 cell_mean=None, drug_mean=None, overall_mean=None):
        super(DeepCrossModel, self).__init__()
        # gene_dim is original cell dimension
        # cell dim if projected cell dim
        self.N = N
        self.M = M
        self.drug_dim = drug_dim
        self.cell_dim = cell_dim
        self.gene_dim = gene_dim

        self.nlayers_cross = nlayers_cross
        self.nlayers_deep = nlayers_deep

        self.drug_embedding = nn.Embedding(M, drug_dim, max_norm=1).double()
        #self.drug_embedding = nn.Embedding(M, drug_dim).double()
        if drug_embedding:
            self.drug_embedding.weight.data.copy_(drug_embedding)
            self.drug_embedding.weight.requires_grad = train_drug
        else:
            nn.init.orthogonal_(self.drug_embedding.weight.data)
            self.drug_embedding.weight.requires_grad = True
            # self.drug_embedding.weight.data.uniform_(0, 1)

        input_sizes = [self.N, self.M, self.gene_dim]

        self.cell_layer = nn.Linear(self.gene_dim, self.cell_dim).double()

        # if cell_mean is not None:
        #     self.cell_layer.bias.data.copy_(cell_mean)

        #self.cell_layer = cellNet(self.gene_dim, self.cell_dim).double()

        if cell_WPmat:
            self.cell_layer.weight.data.copy_(torch.transpose(cell_WPmat, 0, 1))
            self.cell_layer.weight.requires_grad = train_cell
        # else:
        #     # nn.init.orthogonal_(self.cell_layer.weight.data)
        #     # self.cell_layer.weight.requires_grad = True

        # self.cell_BN = nn.BatchNorm1d(self.M).double()

        self.x0_dim = drug_dim+cell_dim+1

        self.cross_classifier = CrossNet(self.x0_dim, self.nlayers_cross, drug_mean).double()
        # self.deep_classifier = build_mlp(self.x0_dim, deep_out_size, nlayers_deep, hidden_sizes).double()
        self.deep_classifier = DeepNet(self.x0_dim, deep_out_size, nlayers_deep,
                                       hidden_sizes, dropout_rates=dropout_rates).double()

        in_final_dim = deep_out_size + self.x0_dim + 1  # 1 is for cosine similarity

        # self.BN = nn.BatchNorm1d(self.M).double()

        #
        self.activation = nn.ReLU()
        self.BN = nn.BatchNorm1d(in_final_dim).double()
        #self.drop_out = nn.Dropout(p=0.2)
        # input 3-D, so output_size=1 in the 3rd dim
        self.classifierF = nn.Linear(in_final_dim, 1, bias=True).double()
        nn.init.constant_(self.classifierF.bias, 10.0)
        if overall_mean:
            self.classifierF.bias.data.copy_(overall_mean)

        #self.classifierF = nn.Linear(in_final_dim, self.M, bias=True).double()

    def forward(self, input):
        # input:[B,M,gene_dim+1]
        B, M, gene_dim = input.size()
        gene_dim -= 1

        cell_fts = input.clone()[:, :, :-1]  # direct slice is sharing memory, if cell_fts changes,input also changes
        drug_index = input.clone()[:, :, -1].long()  # (B,M)

        # expand cell_feats
        cell_fts = cell_fts.view(-1, gene_dim)  # (B*M,gene_dim)
        cell_emb = self.cell_layer(cell_fts)  # #(B*M,f)
        # cell_emb_pn = torch.norm(cell_emb, p=2, dim=1)+1e-5
        # cell_emb = cell_emb/cell_emb_pn[:, None]
        # cell_emb = self.cell_BN(cell_proj)

        drug_emb = self.drug_embedding(drug_index)  # .to(self.device) #(N,M,f)
        f = drug_emb.size()[-1]
        drug_emb = drug_emb.view(-1, f)
        # drug_fts_pn = torch.norm(torch.from_numpy(drug_features), p=2, dim=1)+1e-5
        # drug_embs = drug_features/drug_fts_pn[:, None]

        # cos1 = (cell_emb * drug_emb).sum(2).view(B, self.M, 1)
        cos_sim = nn.CosineSimilarity(dim=1, eps=1e-6)
        cos1 = cos_sim(cell_emb, drug_emb)  # (B*M)
        cos1 = cos1.view(-1, 1)
        #cos1 = cos1.view(B, self.M, 1)

        x0 = torch.cat((cell_emb, drug_emb, cos1), 1)  # (N,M,P1+P2+1)

        x0_dim = self.cell_dim + self.drug_dim+1

        D_out = self.deep_classifier(x0)  # hidden (B*M,hidden_out)

        #x0 = x0.view(-1, x0_dim)
        # C0_out = self.classifierC0(x0, x0)

        C_out = self.cross_classifier(x0)  # .view(B, self.M, -1)  # (N,M,P)

        in_final = torch.cat((D_out, C_out, cos1), 1)

        in_final = self.activation(in_final)
        #in_final = self.BN(in_final)
        #in_final = self.drop_out(in_final)

        out = self.classifierF(in_final).view(-1, M, 1)  # [B*M,1]

        return out

    # ###################original [B,M,P] module###########

    # def forward(self, input):
    #     # input:[B,M,gene_dim+1]
    #     B, M, gene_dim = zip(input.size())
    #     gene_dim -= 1

    #     cell_fts = input.clone()[:, :, :-1]  # direct slice is sharing memory, if cell_fts changes,input also changes
    #     drug_index = input.clone()[:, :, -1].long()

    #     cell_emb = self.cell_layer(cell_fts)  # .to(self.device)#(N,M,P1)
    #     # cell_emb = self.cell_BN(cell_proj)

    #     drug_emb = self.drug_embedding(drug_index)  # .to(self.device) #(N,M,P2)
    #     # drug_fts_pn = torch.norm(torch.from_numpy(drug_features), p=2, dim=1)+1e-5
    #     # drug_embs = drug_features/drug_fts_pn[:, None]

    #     #cos1 = (cell_emb * drug_emb).sum(2).view(B, self.M, 1)
    #     cos_sim = nn.CosineSimilarity(dim=2, eps=1e-6)
    #     cos1 = cos_sim(cell_emb, drug_emb)
    #     cos1 = cos1.view(B, self.M, 1)

    #     x0 = torch.cat((cell_emb, drug_emb, cos1), 2)  # (N,M,P1+P2+1)

    #     D_out = self.deep_classifier(x0)  # hidden (N,M,hidden)
    #     x0_dim = self.cell_dim + self.drug_dim+1

    #     x0 = x0.view(-1, x0_dim)
    #     #C0_out = self.classifierC0(x0, x0)

    #     C_out = self.cross_classifier(x0).view(B, self.M, -1)  # (N,M,P)

    #     in_final = torch.cat((D_out, C_out, cos1), 2)

    #     out_BN = self.BN(in_final)
    #     #out_dp = self.drop_out(out_BN)
    #     out_act = self.activation(out_BN)
    #     out = self.classifierF(out_act)  # [B,M,1]
    #     return out
