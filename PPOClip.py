import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class PPO_Clip:
    def __init__(self, epochs, batch_size, policy_net, value_net, clip_param, lr, discount_factor, gae_lambda):
        '''
        epochs: El maximo numero de epochs, osea el maximo k
        batch_size: El tamaño del batch D de trayectorias
        policy_net: La red neuronal de la politica
        value_net: La red neuronal del valor
        clip_param: El parametro de clipping epsilon
        lr: La tasa de aprendizaje para los optimizadores

        '''
        self.epochs = epochs
        self.batch_size = batch_size
        self.policy_net = policy_net
        self.value_net = value_net
        self.clip_param = clip_param
        self.discount_factor = discount_factor
        self.gae_lambda = gae_lambda
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=lr)
        self.mse_loss = nn.MSELoss()
    

    def collect_trajectories(self, env):
        trajectories = []
        for _ in range(self.batch_size):
            reward_avg = 0
            state, _ = env.reset()
            done = False
            trajectory = []
            while not done:
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                with torch.no_grad():
                    logits = self.policy_net(state_tensor)              # [1, n_actions]
                    dist = torch.distributions.Categorical(logits=logits)
                    action_t = dist.sample()                            # tensor scalar
                    log_prob_t = dist.log_prob(action_t)                # para PPO
                    action = action_t.item()
                next_state, reward, terminated, truncated, _ = env.step(action)
                reward_avg += reward
                done = terminated or truncated
                trajectory.append((state, action, reward, next_state, done))
                state = next_state
            reward_avg /= len(trajectory)
            trajectories.append(trajectory)
        return trajectories, reward_avg
    
    def estimate_advantages(self, trajectories):
        all_advantages = []
        all_returns = []
        for trajectory in trajectories:
            dones = [step[4] for step in trajectory]
            rewards = [step[2] for step in trajectory]
            states = [step[0] for step in trajectory]
            values = self.value_net(torch.FloatTensor(states)).detach().squeeze().numpy()
            advantages = np.zeros(len(rewards))
            returns = np.zeros(len(rewards))
            gae = 0
            next_value = 0
            for t in reversed(range(len(rewards))):
                delta = rewards[t] + self.discount_factor * next_value * (1 - dones[t]) - values[t]
                gae = delta + self.discount_factor * self.gae_lambda * (1 - dones[t]) * gae
                advantages[t] = gae
                returns[t] = gae + values[t]
                next_value = values[t]
            all_advantages.extend(advantages)
            all_returns.extend(returns)
        return torch.FloatTensor(all_advantages), torch.FloatTensor(all_returns)
    
    def maximize_objective(self, trajectories, advantages):
        states = []
        actions = []
        for trajectory in trajectories:
            for step in trajectory:
                states.append(step[0])
                actions.append(step[1])
        states_tensor = torch.FloatTensor(states)
        actions_tensor = torch.LongTensor(actions)
        
        old_action_probs = self.policy_net(states_tensor).gather(1, actions_tensor.unsqueeze(1)).detach()
        
        for _ in range(self.epochs):
            action_probs = self.policy_net(states_tensor).gather(1, actions_tensor.unsqueeze(1))
            ratios = action_probs / old_action_probs
            surr1 = ratios * advantages.unsqueeze(1)
            surr2 = torch.clamp(ratios, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages.unsqueeze(1)
            policy_loss = -torch.min(surr1, surr2).mean()
            
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()

    def fit_value_function(self, trajectories, returns):
        states = []
        for trajectory in trajectories:
            for step in trajectory:
                states.append(step[0])
        states_tensor = torch.FloatTensor(states)
        
        for _ in range(self.epochs):
            value_preds = self.value_net(states_tensor).squeeze()
            value_loss = self.mse_loss(value_preds, returns)
            
            self.value_optimizer.zero_grad()
            value_loss.backward()
            self.value_optimizer.step()

    def train(self, env):   
        rewards = []
        for k in range(self.epochs):

            trajectories, reward = self.collect_trajectories(env)
            advantages, returns = self.estimate_advantages(trajectories)
            self.maximize_objective(trajectories, advantages)
            self.fit_value_function(trajectories, returns)
            rewards.append(reward)
            if (k+1) % 10 == 0:
                print(f"Epoch {k+1}/{self.epochs} completed.")
        return rewards
    

    def evaluate(self, env):
        rewards = []
        state, _ = env.reset()
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        while True:
            with torch.no_grad():
                logits = self.policy_net(state_tensor)
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample().item()

            # Processing:
            obs, reward, terminated, truncated, info = env.step(action)
            rewards.append(reward)

            state_tensor = torch.FloatTensor(obs).unsqueeze(0)
            # Checking if the player is still alive
            if terminated:
                break

        env.close()

        return rewards

       
        


# Example usage:
# Define your policy_net and value_net as instances of nn.Module
# env = YourEnvironment()
# ppo = PPO_Clip(epochs=10, batch_size=5, policy_net=policy_net, value_net=value_net, clip_param=0.2, lr=3e-4, discount_factor=0.99, gae_lambda=0.95)
# ppo.train(env)


