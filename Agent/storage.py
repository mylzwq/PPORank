import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import numpy as np
import scipy


class RolloutStorage():
    def __init__(self, num_steps, num_processes, obs_actor_shape, obs_critic_shape, M):
        self.num_steps = num_steps  # which is min(num_steps,M), first only consider num_steps>M
        self.num_processes = num_processes
        self.obs_critic_shape = obs_critic_shape
        self.obs_actor_shape = obs_actor_shape
        self.obs_critic = torch.zeros(num_steps*num_processes, *obs_critic_shape)  # obs for the value net
        self.obs_actor = torch.zeros(num_steps*num_processes, obs_actor_shape)
        self.rewards = torch.zeros(num_steps*num_processes, 1)
        self.value_preds = torch.zeros(num_steps*num_processes, 1)
        self.returns = torch.zeros(num_steps*num_processes, 1)
        self.action_log_probs = torch.zeros(num_steps*num_processes, 1)
        self.actions = torch.zeros(num_steps*num_processes, 1).long()
        #self.actions = self.actions.long()
        self.masks = torch.ones(num_steps*num_processes, 1)  # used to indicate wheter comes to a trajectory end
        self.filter_masks = torch.zeros(num_steps*num_processes, num_steps)
        self.end_masks_ind = torch.zeros(num_processes)
        self.dist_entropys = torch.zeros(num_steps*num_processes, 1)
        self.steps = 0

    def to(self, device):
        # self.obs_critic = self.obs_critic.to(device)
        self.obs_actor = self.obs_actor.to(device)
        self.rewards = self.rewards.to(device)
        self.value_preds = self.value_preds.to(device)
        self.returns = self.returns.to(device)
        self.action_log_probs = self.action_log_probs.to(device)
        self.actions = self.actions.to(device)
        self.masks = self.masks.to(device).to(device)
        self.filter_masks = self.filter_masks.to(device)
        self.dist_entropys = self.dist_entropys.to(device)

    def sample_episodes(self, agent, input, scores, true_scores, critic_inputs):
        drug_bool1 = torch.isnan(true_scores)
        num_process = scores.size()[0]
        filter_masks1 = torch.zeros_like(scores)
        filter_masks1 = filter_masks1.masked_fill(drug_bool1, float('-inf')).clone()  # (B,M)
        drug_masks = torch.ones_like(true_scores)
        drug_masks = drug_masks.masked_fill(drug_bool1, 0.0).clone()
        M0 = torch.sum(~drug_bool1, dim=1).max().item()
        # M1 = torch.sum(~drug_bool1, dim=1).min().item()
        #inds = torch.sort(torch.sum(~drug_bool1, axis=1))
        # M3 = inds[0][2].item()

        acs = []
        rewards = []
        log_probs = []
        dist_entropys = []
        obs_actor = []
        obs_critic = []
        value_preds = []
        end_masks = torch.ones(num_process, 1)  # is zero indicate done
        filter_masks = []
        dist_entropys = []
        filter_masks_n = [filter_masks1]
        scores_t = scores.clone()

        for step in range(M0):
            # self.filter_masks[step+self.steps].copy_(filter_masks)
            selected_drug_ids, end_masks = agent.actor_critic.sample_action(
                scores_t, filter_masks1)  # (B)

            reward = (2**torch.gather(scores_t[~end_masks], 1, selected_drug_ids[~end_masks].unsqueeze(-1)
                                      )-1)/np.log2(2+step)  # * end_masks.double()  # (B,1)

            log_prob, dist_entropy = agent.actor_critic.get_log_prob(
                scores_t[~end_masks], filter_masks1[~end_masks], selected_drug_ids[~end_masks])  # (B,1)

            # ob_critic = agent.actor_critic.get_fts_vecs(input[~end_masks], filter_masks1[~end_masks])  # (B,Pcritic)
            # if step < M0-1:
            # filter_masks[selected_drug_id] = float('-inf')
            filter_masks1.scatter_(1, selected_drug_ids.unsqueeze(-1), float("-inf"))
            drug_masks.scatter_(1, selected_drug_ids.unsqueeze(-1), 0.0)

            scores_t, critic_input_t = agent.actor_critic(input, drug_masks.unsqueeze(-1))
            scores_t = scores_t.squeeze()
            value_pred = agent.actor_critic.get_value_from_actor(critic_input_t[~end_masks])
            # value_pred = agent.actor_critic.get_value_from_actor(critic_inputs_t[~end_masks])  # (B,1,1)

            # value_preds.append(value_pred[0].fill_(end_masks, float("nan")))
            acs.append(selected_drug_ids[~end_masks])  # (B)
            filter_masks.append(filter_masks1[~end_masks])  # (B,M)
            value_preds.append(value_pred.view(-1, 1))  # (B,1)
            log_probs.append(log_prob)  # (B,1)
            dist_entropys.append(dist_entropy)  # (B,1)
            obs_critic.append(critic_input_t[~end_masks])
            obs_actor.append(input[~end_masks, 0, :-1])
            # filter_masks_n.append(filter_masks1)
            rewards.append(reward)

        self.steps += torch.sum(~torch.isnan(true_scores)).item()
        time_steps_this_batch = self.steps
        rewards = torch.cat([reward for reward in rewards])  # (T,1)
        log_probs = torch.cat([log_pi for log_pi in log_probs])  # (T,1)
        acs = torch.cat([ac for ac in acs])  # T
        obs_critic = torch.cat([ob_critic for ob_critic in obs_critic]).squeeze()  # (T,M,P)
        obs_actor = torch.cat([ob_actor for ob_actor in obs_actor]).squeeze()  # (T,P)
        dist_entropys = torch.cat([dist_entropy for dist_entropy in dist_entropys])
        value_preds = torch.cat([value_pred for value_pred in value_preds])

        self.end_masks_ind = torch.sum(~torch.isnan(true_scores), axis=1)

        self.obs_actor[:time_steps_this_batch].copy_(obs_actor)
        self.obs_critic[:time_steps_this_batch].copy_(obs_critic)
        self.rewards[:time_steps_this_batch].copy_(rewards.view(-1, 1))
        self.value_preds[:time_steps_this_batch].copy_(value_preds.view(-1, 1))
        self.action_log_probs[:time_steps_this_batch].copy_(log_probs.view(-1, 1))
        self.actions[:time_steps_this_batch].copy_(acs.view(-1, 1))
        self.dist_entropys[:time_steps_this_batch].copy_(dist_entropys.view(-1, 1))

        self.obs_critic = self.obs_critic[:time_steps_this_batch].clone()
        self.obs_actor = self.obs_actor[:time_steps_this_batch].clone()
        self.rewards = self.rewards[:time_steps_this_batch].clone()
        self.value_preds = self.value_preds[:time_steps_this_batch].clone()
        self.action_log_probs = self.action_log_probs[:time_steps_this_batch].clone()
        self.actions = self.actions[:time_steps_this_batch].clone()
        self.returns = self.returns[:time_steps_this_batch].clone()
        self.filter_masks = self.filter_masks[:time_steps_this_batch].clone()
        self.dist_entropys = self.dist_entropys[:time_steps_this_batch].clone()

        return

    def sample_num_steps(self, agent, input, scores, true_scores, critic_inputs):
        #input : [num_process,M,cell_dim+1]
        # scores: [num_process,M] ts, true_scores:[num_process,M] ts
        num_process = scores.size()[0]
        M = scores.size()[1]
        paths = []
        time_steps_this_batch = 0

        for cell in range(num_process):
            rewards_single = []
            obs_actor_single = []
            acs_single = []
            log_pi_single = []
            dist_entropy_single = []
            # obs_single = []
            value_pred_single = []
            drug_bool = torch.isnan(true_scores[cell])
            M0 = torch.sum(~drug_bool)
            if M0.item() == 0:
                continue
            filter_masks = torch.zeros_like(scores[0])
            filter_masks = filter_masks.masked_fill_(drug_bool, float('-inf')).clone()
            for step in range(M0):
                self.filter_masks[step+self.steps].copy_(filter_masks)
                selected_drug_id, _ = agent.actor_critic.sample_action(scores[cell], filter_masks)  # is a tensor
                acs_single.append(selected_drug_id)
                rewards_single.append((2**true_scores[cell][selected_drug_id]-1)/np.log2(step+2))  # tensor
                log_prob, dist_entropy = agent.actor_critic.get_log_prob(scores[cell], filter_masks, selected_drug_id)
                log_pi_single.append(log_prob)
                dist_entropy_single.append(dist_entropy)
                ob_critic = agent.actor_critic.get_fts_vecs(input[cell], filter_masks)
                if step < M0-1:
                    filter_masks[selected_drug_id] = float('-inf')

                #value_pre = agent.actor_critic.get_value(input[cell], filter_masks)
                value_pre = agent.actor_critic.get_value_from_actor(critic_inputs[cell].unsqueeze(0))
                value_pred_single.append(value_pre[0])
                # obs_single.append(ob_critic)
                obs_actor_single.append(input[cell][0][:-1].unsqueeze(0).clone())
            self.end_masks_ind[cell] = M0
            self.masks[M0+self.steps-1] = 0.0
            self.steps += M0
            path = {"rewards": torch.stack(rewards_single),
                    "log_pi": log_pi_single,  # log_pi_single each element requires grad, so can't be used as np
                    "dist_entropy": dist_entropy_single,  # try to consider torch.stack
                    "actions": torch.stack(acs_single),
                    # "obs_critic": torch.stack(obs_single, axis=0),
                    "value_pred": value_pred_single,
                    "obs_actor": torch.stack(obs_actor_single, axis=0),
                    }
            time_steps_this_batch += M0
            paths.append(path)

        time_steps_this_batch = self.steps

        # concatenate

        # self.rewards[:self.steps].copy_(torch.cat(reward))

        return paths  # this is a batch for training, total steps of M*num_process

    def return_of_rewards(self, rewards_paths, value_paths, use_gae, gamma, gae_lambda):
        """
        re_n: length: num_paths. Each element in re_n is a numpy array 
                    containing the rewards for the particular path
        """
        if use_gae:
            gae = 0
            steps = 0
            for i in range(len(rewards_paths)):
                re = rewards_paths[i]
                val = value_paths[i]
                re_np = re.cpu().detach().numpy()
                M0 = re_np.shape[0]
                self.returns[M0-1+steps] = re_np[-1]
                for step in reversed(range(M0-1)):
                    delta = re_np[step] + gamma * val[step+1] - val[step]
                    gae = delta + gamma * gae_lambda * gae
                    self.returns[step+steps] = gae + val[step]
                steps += M0
        else:
            steps = 0
            for re in rewards_paths:
                re_np = re.cpu().numpy().clone()
                M0 = re_np.shape[0]
                q_n = scipy.signal.lfilter(b=[1], a=[1, -gamma], x=re[::-1])[::-1]
                self.returns[self.steps:self.steps+M0] = torch.from_numpy(q_n).clone()
                steps += re_np.shape[0]

    def sample_concatenate(self, paths):

        time_steps_this_batch = self.steps
        re_n = torch.cat([path["rewards"] for path in paths])
        log_pis = [path["log_pi"] for path in paths]
        log_pi_n = torch.stack([y for x in log_pis for y in x])  # a single tensor [35]
        ac_n = torch.cat([path["actions"] for path in paths])
        # ob_critic_n = torch.cat([path["obs_critic"] for path in paths]).squeeze()  # np ()
        ob_actor_n = torch.cat([path["obs_actor"] for path in paths]).squeeze()  # np ()
        d_ns = [path["dist_entropy"] for path in paths]
        dist_entropy_n = torch.stack([y for x in d_ns for y in x])
        v_ns = [path["value_pred"] for path in paths]
        value_pred_n = torch.stack([y for x in v_ns for y in x]).squeeze()  # (35,1,1)

        #assert self.steps == re_n.shape[0]
        # self.obs_critic[:time_steps_this_batch].copy_(ob_critic_n)
        self.obs_actor[:time_steps_this_batch].copy_(ob_actor_n)
        self.rewards[:time_steps_this_batch].copy_(re_n.view(-1, 1))
        self.value_preds[:time_steps_this_batch].copy_(value_pred_n.view(-1, 1))
        self.action_log_probs[:time_steps_this_batch].copy_(log_pi_n.view(-1, 1))
        self.actions[:time_steps_this_batch].copy_(ac_n.view(-1, 1))
        self.dist_entropys[:time_steps_this_batch].copy_(dist_entropy_n.view(-1, 1))

        # self.obs_critic = self.obs_critic[:time_steps_this_batch].clone()
        self.obs_actor = self.obs_actor[:time_steps_this_batch].clone()
        self.rewards = self.rewards[:time_steps_this_batch].clone()
        self.value_preds = self.value_preds[:time_steps_this_batch].clone()
        self.action_log_probs = self.action_log_probs[:time_steps_this_batch].clone()
        self.actions = self.actions[:time_steps_this_batch].clone()
        self.returns = self.returns[:time_steps_this_batch].clone()
        self.filter_masks = self.filter_masks[:time_steps_this_batch].clone()
        self.dist_entropys = self.dist_entropys[:time_steps_this_batch].clone()

    def compute_returns(self,
                        use_gae,
                        gamma,
                        gae_lambda,
                        use_proper_time_limits=False):
        if use_proper_time_limits:
            if use_gae:
                gae = 0
                steps = 0
                for cell in range(self.num_processes):
                    M0 = int(self.end_masks_ind[cell].item())
                    delta = self.rewards[M0+steps-1]
                    gae = delta + gamma * gae_lambda * self.masks[M0-1+steps] * gae
                    self.returns[M0-1+steps] = gae + self.value_preds[M0-1+steps]
                    for step in reversed(range(M0-1)):
                        delta = self.rewards[step+steps] + gamma * self.value_preds[step + 1+steps]\
                            * self.masks[step + 1] - self.value_preds[step+steps]
                        gae = delta + gamma * gae_lambda * self.masks[step + 1+steps] * gae
                        self.returns[step+steps] = gae + self.value_preds[step+steps]
            else:
                steps = 0
                for cell in range(self.num_processes):
                    M0 = int(self.end_masks_ind[cell].item())
                    self.returns[M0-1+steps] = self.rewards[M0-1+steps]
                    for step in reversed(range(M0-1)):
                        self.returns[step+steps] = self.returns[step + 1+steps] * \
                            gamma * self.masks[step + 1+steps] + self.rewards[step+steps]
                    steps += M0
        else:
            if use_gae:
                gae = 0
                steps = 0
                for cell in range(self.num_processes):
                    M0 = int(self.end_masks_ind[cell].item())
                    self.returns[M0-1+steps] = self.rewards[M0+steps-1].clone()

                    for step in reversed(range(M0-1)):
                        delta = self.rewards[step+steps] + gamma * \
                            self.value_preds[step + 1 + steps] - self.value_preds[step+steps]

                        gae = delta + gamma * gae_lambda * gae

                        self.returns[step+steps] = gae + self.value_preds[step+steps]
                    steps += M0
            else:
                steps = 0
                for cell in range(self.num_processes):
                    M0 = int(self.end_masks_ind[cell].item())

                    self.returns[M0-1+steps] = self.rewards[M0-1+steps]

                    for step in reversed(range(M0-1)):
                        self.returns[step+steps] = self.returns[step + 1+steps] * gamma + self.rewards[step+steps]
                    steps += M0

    def feed_forward_generator(self, advantages, num_mini_batch=None, mini_batch_size=None):

        batch_size = self.steps.item()
        if mini_batch_size is None:
            assert batch_size >= num_mini_batch, (
                "PPO requires the number of processes ({}) "
                "* number of steps ({}) = {} "
                "to be greater than or equal to the number of PPO mini batches ({})."
                "".format(self.num_processes, self.num_steps, self.num_processes * self.num_steps,
                          num_mini_batch))
        mini_batch_size = batch_size // num_mini_batch

        sampler = BatchSampler(SubsetRandomSampler(range(batch_size)), mini_batch_size, drop_last=True)
        for indices in sampler:
            # obs_critic_batch = self.obs_critic.view(-1, self.obs_critic.size()[1])[indices]  # this is after filtering
            # only include cell line without filtering
            obs_actor_batch = self.obs_actor.view(-1, self.obs_actor.size()[1])[indices]
            filter_masks_batch = self.filter_masks.view(-1, self.filter_masks.size()[1])[indices]
            actions_batch = self.actions.view(-1, 1)[indices]
            value_preds_batch = self.value_preds.view(-1, 1)[indices]
            return_batch = self.returns.view(-1, 1)[indices]
            masks_batch = self.masks.view(-1, 1)[indices]
            old_action_log_probs_batch = self.action_log_probs.view(-1, 1)[indices]

            if advantages is None:
                adv_targ = None
            else:
                adv_targ = advantages.view(-1, 1)[indices]

            # yield obs_critic_batch, obs_actor_batch, filter_masks_batch, actions_batch,\
            #     value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, adv_targ
            yield obs_actor_batch, filter_masks_batch, actions_batch,\
                value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, adv_targ

    def after_update(self):
        num_steps = self.num_steps
        num_processes = self.num_processes

        # self.obs_critic = torch.zeros(num_steps*num_processes, self.obs_critic_shape)  # obs for the value net
        self.obs_actor = torch.zeros(num_steps*num_processes, self.obs_actor_shape)
        self.rewards = torch.zeros(num_steps*num_processes, 1)
        self.value_preds = torch.zeros(num_steps*num_processes, 1)
        self.returns = torch.zeros(num_steps*num_processes, 1)
        self.action_log_probs = torch.zeros(num_steps*num_processes, 1)
        self.actions = torch.zeros(num_steps*num_processes, 1)
        self.actions = self.actions.long()
        self.masks = torch.ones(num_steps*num_processes, 1)  # used to indicate wheter comes to a trajectory end
        self.filter_masks = torch.zeros(num_steps*num_processes, num_steps)
        self.end_masks_ind = torch.zeros(num_processes)
        self.dist_entropys = torch.zeros(num_steps*num_processes, 1)
        self.steps = 0
