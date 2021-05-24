import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from models.Policy import *

# python main.py --env-name "Reacher-v2" --algo ppo --use_gae(true) --log-interval 1 --num-steps 2048 --num-processes 1
# --lr 3e-4 --entropy-coef 0 --value-loss-coef 0.5 --ppo-epoch 10 --num-mini-batch 32
#  --gamma 0.99 --gae-lambda 0.95 --num-env-steps 1000000 --use_linear_lr_decay(true)
# --use-proper-time-limits(true)


class PPO():
    def __init__(self, args, N, M, P, f, WP, drug_embs, device, drug_mean=None, overall_mean=None):
        self.clip_param = args.clip_param
        self.ppo_epoch = args.ppo_epoch
        self.num_mini_batch = args.num_mini_batch
        self.value_loss_coef = args.value_loss_coef
        self.lr = args.lr
        self.eps = args.eps
        self.max_grad_norm = args.max_grad_norm
        self.use_clipped_value_loss = True
        self.N = N
        self.M = M
        self.gene_dim = P
        self.cell_dim = f
        self.drug_dim = f
        self.total_dim = self.drug_dim + self.cell_dim + 1  # 1 is for cosine similarity
        self.entropy_coef = args.entropy_coef
        self.value_loss_coef = args.value_loss_coef
        self.shared_params = args.shared_params
        if not self.shared_params:
            # PPO_Policy comes from nn.module
            self.actor_critic = PPO_Policy(
                N, M, self.gene_dim, self.drug_dim, drug_embs, self.cell_dim,
                WP, args.nlayers_cross, args.nlayers_deep, args.deep_hidden_sizes, args.deep_out_size,
                args.nlayers_value, args.value_hidden_sizes, device,
                train_cell=args.train_cell, train_drug=args.train_drug,
                drug_mean=drug_mean, overall_mean=overall_mean)

        else:
            self.actor_critic = PPO_Shared_Policy(
                N, M, self.gene_dim, self.drug_dim, drug_embs, self.cell_dim,
                WP, args.nlayers_cross, args.nlayers_deep, args.deep_hidden_sizes, args.deep_out_size,
                args.nlayers_value, args.value_hidden_sizes,
                train_cell=args.train_cell, train_drug=args.train_drug)
        # self.critic_size = self.actor_critic.critic_size

        # self.actor_critic = nn.DataParallel(self.actor_critic)

        # self.parameters = self.actor_critic.parameters
        self.optimizer = optim.AdamW(self.actor_critic.parameters, lr=args.lr, eps=args.eps)
        self.actor_critic.to(device)

    def update(self, rollouts):
        advantages = rollouts.returns - rollouts.value_preds
        advantages = (advantages - advantages.mean()) / (
            advantages.std() + 1e-5)

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0
        losses_epoch = 0
        ppo_update_ind = 0
        tot_ppo_update = self.ppo_epoch * (rollouts.steps.item()//self.num_mini_batch)

        for e in range(self.ppo_epoch):
            data_generator = rollouts.feed_forward_generator(
                advantages, self.num_mini_batch)

            for sample in data_generator:
                # old_action_log_probs_batch requr_grad false
                obs_actor_batch, filter_masks_batch, actions_batch, value_preds_batch,\
                    return_batch, masks_batch, old_action_log_probs_batch, adv_targ = sample

                # Reshape to do in a single forward pass for all steps, all requr_grad true
                values, action_log_probs, dist_entropy = self.actor_critic.evaluate_actions(
                    obs_actor_batch, filter_masks_batch, actions_batch)

                ratio = torch.exp(action_log_probs -
                                  old_action_log_probs_batch)  # [64,1]
                surr1 = ratio * adv_targ  # [64,1]

                cur_lrmult = 1-ppo_update_ind/tot_ppo_update
                clip_param = self.clip_param * cur_lrmult
                surr2 = torch.clamp(ratio, 1.0 - clip_param,
                                    1.0 + clip_param) * adv_targ
                action_loss = -torch.min(surr1, surr2).mean()

                ppo_update_ind += 1

                if self.use_clipped_value_loss:
                    value_pred_clipped = value_preds_batch + \
                        (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
                    value_losses = (values - return_batch).pow(2)
                    value_losses_clipped = (
                        value_pred_clipped - return_batch).pow(2)
                    value_loss = 0.5 * torch.max(value_losses,
                                                 value_losses_clipped).mean()
                else:
                    value_loss = 0.5 * (return_batch - values).pow(2).mean()

                self.optimizer.zero_grad()
                (value_loss * self.value_loss_coef + action_loss - dist_entropy * self.entropy_coef).backward()

                nn.utils.clip_grad_norm_(self.actor_critic.parameters,
                                         self.max_grad_norm)
                self.optimizer.step()

                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()
                losses_epoch += value_loss.item() * self.value_loss_coef + action_loss.item() - dist_entropy.item() * self.entropy_coef

        num_updates = self.ppo_epoch * self.num_mini_batch

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates
        losses_epoch /= num_updates

        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch, losses_epoch
