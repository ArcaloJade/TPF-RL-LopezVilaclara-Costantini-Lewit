import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal, Categorical
from torch.optim import Adam
import numpy as np

from my_networks import ActorNN, CriticNN

class PPO_Clip:
    def __init__(self, env):
        # Extract environment information
        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        if hasattr(env.action_space, "n"):
            # Entorno DISCRETO (ej: Flappy Bird)
            self.is_discrete = True
            self.act_dim = env.action_space.n
        else:
            # Entorno CONTINUO (ej: Pendulum, BipedalWalker)
            self.is_discrete = False
            self.act_dim = env.action_space.shape[0]

        # Step 1 algorithm
        self.actor = ActorNN(self.obs_dim, self.act_dim)
        self.critic = CriticNN(self.obs_dim, 1)
        self._init_hyperparameters()

        if not self.is_discrete:
            self.cov_var = torch.full(size=(self.act_dim,), fill_value = 0.5)
            self.cov_mat = torch.diag(self.cov_var)

        self.actor_optim = Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)

        self.obs_mean = np.zeros(self.obs_dim, dtype=np.float32)
        self.obs_var = np.zeros(self.obs_dim, dtype=np.float32)
        self.obs_count = 1e-4 # Para evitar división por cero

    def get_action(self, obs):
        obs = torch.tensor(obs, dtype=torch.float32)

        mean = self.actor(obs)

        if self.is_discrete:
            # Distribución categórica (acciones discretas)
            dist = Categorical(logits=mean)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            return action.item(), log_prob.detach()
        else:
            # Distribución normal multivariada (acciones continuas)
            dist = MultivariateNormal(mean, self.cov_mat)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            return action.detach().numpy(), log_prob.detach()
    
    def rollout(self):
        # Batch data
        batch_obs = []              # batch observations
        batch_acts = []             # batch actions
        batch_log_probs = []        # log probs of each action
        batch_rews = []             # batch rewards
        batch_rtgs = []             # batch rewards-to-go
        batch_lens = []             # episodic lengths in batch
        batch_vals = []
        batch_dones = []

        t_so_far = 0
        while t_so_far < self.timesteps_per_batch:
            ep_rews = []
            ep_vals = []
            ep_dones = []

            obs, _ = self.env.reset()

            done = False

            for ep_t in range(self.max_timesteps_per_episode):
                t_so_far += 1

                batch_obs.append(obs)
                val = self.critic(torch.tensor(obs, dtype=torch.float32)).detach().item()

                action, log_prob = self.get_action(obs)
                obs, rew, terminated, truncated, _ = self.env.step(action)

                ep_rews.append(rew)
                ep_vals.append(val)
                batch_acts.append(action)
                batch_log_probs.append(log_prob)

                done = terminated or truncated
                ep_dones.append(done)

                if done: break


            batch_lens.append(ep_t + 1)
            batch_rews.append(ep_rews)
            batch_vals.append(ep_vals)
            batch_dones.append(ep_dones)

        batch_obs = torch.tensor(np.array(batch_obs), dtype=torch.float32)
        if self.is_discrete:
            batch_acts = torch.tensor(np.array(batch_acts), dtype=torch.long)
        else:
            batch_acts = torch.tensor(np.array(batch_acts), dtype=torch.float32)
        batch_log_probs = torch.tensor(np.array(batch_log_probs), dtype=torch.float32)
        # ALG STEP #4
        batch_rtgs = self.compute_rtgs(batch_rews)

        # Return the batch data
        return batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens, batch_rews, batch_vals, batch_dones
    
    def _init_hyperparameters(self):
        self.timesteps_per_batch = 4096
        self.max_timesteps_per_episode = 5000
        self.gamma = 0.99
        self.n_updates_per_iteration = 5
        self.clip = 0.15
        self.lr = 5e-5
        self.ent_coef = 0.01
        self.lambda_value = 0.95

    # G_T = r_t + gamma * G_T+1
    def compute_rtgs(self, batch_rews):
        batch_rtgs = []
        for ep_rews in reversed(batch_rews):
            discounted_reward = 0
            for rew in reversed(ep_rews):
                discounted_reward = rew + discounted_reward * self.gamma
                batch_rtgs.insert(0, discounted_reward)
        batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float)
        return batch_rtgs
    
    def evaluate(self, batch_obs, batch_acts):
        mean = self.actor(batch_obs)

        if self.is_discrete:
            dist = Categorical(logits=mean)
            log_probs = dist.log_prob(batch_acts.long())
        else:
            mean = mean.squeeze(0) if mean.dim() > 1 else mean
            dist = MultivariateNormal(mean, self.cov_mat)
            log_probs = dist.log_prob(batch_acts)

        V = self.critic(batch_obs).squeeze()
        entropy = dist.entropy()

        return V, log_probs, entropy
    
    def calculate_gae(self, rewards, values, dones):
        batch_advantages = []
        for ep_rews, ep_vals, ep_dones in zip(rewards, values, dones):
            advantages = []
            last_advantage = 0

            for t in reversed(range(len(ep_rews))):
                if t + 1 < len(ep_rews):
                    delta = ep_rews[t] + self.gamma * ep_vals[t+1] * (1 - ep_dones[t+1]) - ep_vals[t]
                else:
                    delta = ep_rews[t] - ep_vals[t]
                # A_t = δ_t + γλ (1 - donet) A_{t+1}
                advantage = delta +self.gamma * self.lambda_value * (1 - ep_dones[t]) * last_advantage
                last_advantage = advantage
                advantages.insert(0, advantage)

            batch_advantages.extend(advantages)

        return torch.tensor(batch_advantages, dtype=torch.float)

    
    def learn(self, total_timesteps):
        t_so_far = 0
        all_ep_rewards = []
        while t_so_far < total_timesteps:
            batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens, batch_rews, batch_vals, batch_dones = self.rollout()

            batch_reward = np.mean([np.sum(ep) for ep in batch_rews])
            all_ep_rewards.append(batch_reward)
            print(f"Timestep: {t_so_far} | Reward promedio: {batch_reward:.2f}")
            # ALG STEP 5
            # Calculate advantage
            A_k = self.calculate_gae(batch_rews, batch_vals, batch_dones)
            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

            for _ in range(self.n_updates_per_iteration):
                # Learning Rate Annealing
                frac = (t_so_far - 1.0) / total_timesteps
                new_lr = self.lr * (1.0 - frac)
                new_lr = max(new_lr, 0.0)
                self.actor_optim.param_groups[0]["lr"] = new_lr
                self.critic_optim.param_groups[0]["lr"] = new_lr

                V, curr_log_probs, entropy = self.evaluate(batch_obs, batch_acts)

                # Calculate Importance Sampling
                ratios = torch.exp(curr_log_probs - batch_log_probs)

                surr1 = ratios * A_k
                surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k

                actor_loss = (-torch.min(surr1, surr2)).mean()

                # Entropy
                entropy_loss = entropy.mean()
                actor_loss = actor_loss - self.ent_coef * entropy_loss


                self.actor_optim.zero_grad()
                actor_loss.backward()
                self.actor_optim.step()

                V, curr_log_probs, _ = self.evaluate(batch_obs, batch_acts)
                critic_loss = nn.MSELoss()(V, batch_rtgs)


                # Calculate gradients and perform backward propagation for critic network
                self.critic_optim.zero_grad()
                critic_loss.backward()
                self.critic_optim.step()

            t_so_far += np.sum(batch_lens)

        return all_ep_rewards