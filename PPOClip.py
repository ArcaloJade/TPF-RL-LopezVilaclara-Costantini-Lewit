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
            state = env.reset()
            done = False
            trajectory = []
            while not done:
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                with torch.no_grad():
                    action_probs = self.policy_net(state_tensor)
                action = np.random.choice(len(action_probs.squeeze().numpy()), p=action_probs.squeeze().numpy())
                next_state, reward, done, _ = env.step(action)
                trajectory.append((state, action, reward, next_state, done))
                state = next_state
            trajectories.append(trajectory)
        return trajectories
    
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
        for k in range(self.epochs):
            trajectories = self.collect_trajectories(env)
            advantages, returns = self.estimate_advantages(trajectories)
            self.maximize_objective(trajectories, advantages)
            self.fit_value_function(trajectories, returns)

# Example usage:
# Define your policy_net and value_net as instances of nn.Module
# env = YourEnvironment()
# ppo = PPO_Clip(epochs=10, batch_size=5, policy_net=policy_net, value_net=value_net, clip_param=0.2, lr=3e-4, discount_factor=0.99, gae_lambda=0.95)
# ppo.train(env)


