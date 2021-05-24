import torch
import argparse
import torch.nn as nn
from torch.autograd import Variable
import utils
import Reward_utils
import numpy as np
import os
import time

import torch.optim as optim
import torch.utils.data

def parse_args():
    parser=argparse.ArgumentParser(description="Policy Gradient Ranking with Pytorch")
    #parser.add_argument("config",help="config file")
    parser.add_argument("--Data_Dir",type=str,default="./Simu_Data")
    parser.add_argument("--epochs",type=int,default=10000,metavar='N',help="number of epochs to train")
    parser.add_argument("--batch_size",type=int,default=64)
    parser.add_argument("--saved_path",type=str,default="./Saved")
    parser.add_argument("--evalations",type=str,default="./Eval")
    parser.add_argument('--k',type=int,default=5,help="top k drugs recommended")
    parser.add_argument('--discount', type=float, default=0.99, metavar='discount',
                    help='discount rate')
    parser.add_argument('--load_model', type=bool, default=True, metavar='S',
                    help='whether to load pre-trained model?')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate')
    parser.add_argument('--cell_dim',type=int,default=100,help="cell-line feature dimension")
    parser.add_argument('--drug_dim',type=int,default=5,help="drug feature dimension")
    parser.add_argument('--cell_num',type=int,default=100,help="cell-line number")
    parser.add_argument('--drug_num',type=int,default=10,help="cell-line number")
    
    return parser.parse_args()


class PolicyGradientRank(nn.Module):
    def __init__(self,P): # m is the feature dimension except x0 term
        super(PolicyGradientRank, self).__init__()
        self.conv2=nn.Conv2d(1,1,kernel_size=(1,P),stride=(1,P),bias=True)
        self.conv2.weight.data.zero_()
        self.softmax=nn.Softmax()
    
    def forward(self,input):# online ranking
        return self.conv2(input)
    
class PG_Rank_NN(nn.Module):
    def __init__(self,N,M,P):
        super(PG_Rank_NN,self).__init__()
        self.N=N
        self.M=M
        self.P=P
        #self.l1=nn.Linear(self.P,1,bias=True)
        self.conv2=nn.Conv2d(1,1,kernel_size=(1,P),stride=(1,P),bias=True)
        self.conv2.weight.data.fill_(0.1)
        #self.l2=nn.Linear(16,1,bias=True)
    # def forward(self,input):
    #     model=torch.nn.Sequential(
    #         self.l1,
    #         #nn.Dropout(p=0.6),
    #         nn.ReLU(),
    #         self.l2,
    #         nn.Softmax(dim=1)
    #     )
    def forward(self,input):
        model=self.conv2
        return model(input)




def sample_episode(model,input,scores,true_scores):
    
    #scores=model.forward(input) #(N,M)
    M = scores.size()[1]
    order_as_scores=Variable(torch.zeros(scores.squeeze().data.size()))#(N,M)

    log_pi=Variable(torch.zeros(scores.squeeze().data.size()))#(N,M)

    selected_drug=Variable(torch.zeros(log_pi.size()[0],log_pi.size()[1]))#(N,M)
    
    for cell in range(scores.size()[0]):
        
        score_exp=scores[cell].squeeze().exp()#[M]
        masks=Variable(torch.ones(M).unsqueeze(0).repeat(M,1),requires_grad=False)
        #masks=Variable(torch.cat((torch.ones(M), torch.zeros(n - M))).unsqueeze(0).repeat(M,1),requires_grad=False)
        selected=[]

        for t in range(M):
            selected_drug_id=torch.multinomial(score_exp*masks[t],1).numpy()[0]
            selected+=[selected_drug_id]
            order_as_scores[cell,selected_drug_id]=M-t-1
            if t <M-1:
                masks[t+1]=Variable(torch.from_numpy(masks[t].data.clone().numpy()))
                masks[t+1,selected_drug_id]=0
        
        masks_clone=masks.clone()
        for t in range(M):
            log_pi[cell,t]=torch.log(score_exp[selected[t]]/torch.sum(score_exp*masks_clone[t]))#(N,M)
        

    rewards=Reward_utils.cal_immediate_reward(true_scores.squeeze(),order_as_scores)#(N,M)
    discount_rewards=Reward_utils.discount_sum_rewards(args.discount,rewards,M)#(N,M)

    return rewards,discount_rewards,log_pi 
    
def MDP_gradient(model,input,output):
    rewards,discount_rewards,log_pi = sample_episode(model,input,output)
    Q_n=Variable(torch.Tensor(1))
    model.zero_grad()
    N=output.size()[0]
    M=output.size()[1]

    for cell in range(N):
        
        Gt=Reward_utils.discount_sum_of_rewards(args.discount,rewards,M)
        Q_n+=np.multiply(Gt,log_pi)

    Q_n.backward()

    for param in model.parameters():
        param.data += args.lr * param.grad.data
        

def train(model,input,true_scores,args):
    # switch to train mode
    model.train()
    P=input.size()[3]
    output=model(input).squeeze() #(10000,1,200,1)(N,1,M,M)

    rewards,Gt,log_pi = sample_episode(model,input,output,true_scores)
    #Gt=Reward_utils.discount_sum_of_rewards(args.discount,rewards,M)
    b=torch.mean(rewards)
    loss=-torch.mean(torch.mul(log_pi,Gt))
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adagrad(parameters, lr=0.01)

    # compute gradient and do SGD step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print("current loss is {}".format(loss))

    ndcg_val= Reward_utils.ndcg(true_scores.numpy(),output.squeeze().detach().numpy(),args.k)

    #print("current ndcg is {}".format(ndcg_val))
    return ndcg_val

def main():
    global args
    args=parse_args()

    simu_data_dir="./Simu_Data"
    
    cell_dim=args.cell_dim # cell -line feature
    drug_dim=args.drug_dim # drug features
    beta0=1
    beta1=np.ones(cell_dim)*0.1
    theta=np.ones(cell_dim*drug_dim)*0.1
    input,true_scores,true_scores_norm=utils.load_data_simulator(simu_data_dir,
                                                                    args.cell_num,
                                                                    args.drug_num,
                                                                    args.cell_dim,
                                                                    args.drug_dim,
                                                                    beta0=beta0,
                                                                    beta1=beta1,
                                                                    theta=theta)
    # input is (N,1,M,P)
    N=input.size()[0]
    M=input.size()[2]
    P=input.size()[3]
    #model=PolicyGradientRank(P)
    model=PG_Rank_NN(N,M,P)
    best_ndcg=0

    for epo in range(args.epochs):

        ndcg_val=train(model,input,true_scores_norm,args)
        is_best=ndcg_val>best_ndcg
        
        best_ndcg=max(ndcg_val,best_ndcg)
        print("current ndcg is {}, and best ndcg is {}".format(ndcg_val,best_ndcg))

if __name__=="__main__":
    main()










    

            



        
        





    

