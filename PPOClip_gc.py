import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
from google.cloud import storage

BUCKET_NAME = "mi-bucket-rl-mateo"
def upload_to_gcs(local_path, bucket_name, destination_blob_name):
    """
    Sube un archivo local a un bucket de GCS.
    local_path: ruta del archivo en el runtime de Colab Enterprise (ej: 'weights.pth')
    bucket_name: nombre del bucket en GCS (ej: 'mi-bucket-rl-mateo')
    destination_blob_name: ruta/filename dentro del bucket (ej: 'checkpoints/weights.pth')
    """
    client = storage.Client()  # usa credenciales por defecto del entorno
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(local_path)
    print(f"Archivo subido a gs://{bucket_name}/{destination_blob_name}")

#self.policy_net.state_dict()
def save_in_gc(k, policy_net_params, value_net_params, rewards, loss, entropy):
  BUCKET_NAME = "mi-bucket-rl-mateo"  # reemplazá por el nombre real
  
  local_path = f"policy_weights_epoch_{k}.pth"
  torch.save(policy_net_params, local_path)
  upload_to_gcs(
      local_path=local_path,
      bucket_name=BUCKET_NAME,
      destination_blob_name=f"checkpoints/{local_path}"
  )

  local_path = f"value_weights_epoch_{k}.pth"
  torch.save(value_net_params, local_path)
  upload_to_gcs(
      local_path=local_path,
      bucket_name=BUCKET_NAME,
      destination_blob_name=f"checkpoints/{local_path}"
  )


  metrics = {
    "rewards": rewards,          # pueden ser listas o np.array
    "loss": loss,
    "entropy": entropy,
  }

  local_metrics_path = f"metrics_epoch_{k}.pt"
  torch.save(metrics, local_metrics_path)

  upload_to_gcs(
      local_path=local_metrics_path,
      bucket_name=BUCKET_NAME,
      destination_blob_name=f"checkpoints/metrics_epoch_{k}.pt"
  )

class PPO_Clip:
    def __init__(self, hiperparams):
        '''
        epochs: El maximo numero de epochs, osea el maximo k
        batch_size: El tamaño del batch D de trayectorias
        policy_net: La red neuronal de la politica
        value_net: La red neuronal del valor
        clip_param: El parametro de clipping epsilon
        lr: La tasa de aprendizaje para los optimizadores

        '''
        self.epochs = hiperparams["epochs"]
        self.K = hiperparams["K"]
        self.batch_size = hiperparams["batch_size"]
        self.policy_net = hiperparams["policy_net"]
        self.value_net = hiperparams["value_net"]
        self.clip_param = hiperparams["clip_param"]
        self.lr = hiperparams["lr"]
        self.discount_factor = hiperparams["discount_factor"]
        self.gae_lambda = hiperparams["gae_lambda"]
        self.max_length = hiperparams["max_length"]
        self.entropy_coeficient = hiperparams["entropy_coeficient"]
        #self.device = device
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=self.lr)
        self.mse_loss = nn.MSELoss()

        self.policy_net#.to(self.device)
        self.value_net#.to(self.device)
    

    def collect_trajectories(self, env):
        trajectories = []
        rewards = []
        for _ in range(self.batch_size):
            reward_sum = 0
            state, _ = env.reset()
            state = np.array(state, dtype=np.float32)
            done = False
            trajectory = []
            index = 0
            while not done:
                state_tensor = torch.FloatTensor(state).unsqueeze(0)#.to(self.device) 
                with torch.no_grad():
                    logits = self.policy_net(state_tensor)              # [1, n_actions]
                    dist = torch.distributions.Categorical(logits=logits)
                    action_t = dist.sample()                            # tensor scalar
                    log_prob_t = dist.log_prob(action_t)                # para PPO
                    action = action_t.item()
                next_state, reward, terminated, truncated, _ = env.step(action)
                next_state = np.array(next_state, dtype=np.float32)
                if index > self.max_length:
                    truncated = True
                index += 1
                reward_sum += reward
                done = terminated or truncated
                trajectory.append((state, action, reward, next_state, done))
                state = next_state
            rewards.append(reward_sum)
            trajectories.append(trajectory)
        return trajectories, rewards
    
    # def estimate_advantages(self, trajectories):
    #     all_advantages = []
    #     all_returns = []
    #     for trajectory in trajectories:
    #         dones = [step[4] for step in trajectory]
    #         rewards = [step[2] for step in trajectory]
    #         states = [step[0] for step in trajectory]
    #         values = self.value_net(torch.FloatTensor(states)).detach().squeeze().numpy()
    #         advantages = np.zeros(len(rewards))
    #         returns = np.zeros(len(rewards))
    #         gae = 0
    #         next_value = 0
    #         for t in reversed(range(len(rewards))):
    #             delta = rewards[t] + self.discount_factor * next_value * (1 - dones[t]) - values[t]
    #             gae = delta + self.discount_factor * self.gae_lambda * (1 - dones[t]) * gae
    #             advantages[t] = gae
    #             returns[t] = gae + values[t]
    #             next_value = values[t]
    #         all_advantages.extend(advantages)
    #         all_returns.extend(returns)
    #     return torch.FloatTensor(all_advantages), torch.FloatTensor(all_returns)

    def estimate_advantages(self, trajectories):
        all_advantages = []
        all_returns = []
        for trajectory in trajectories:
            dones = [step[4] for step in trajectory]
            rewards = [step[2] for step in trajectory]
            states = [step[0] for step in trajectory]

            # asegurar device correcto
            states = np.array(states)      # junta la lista de ndarrays en un solo array
            states_tensor = torch.from_numpy(states)
            with torch.no_grad():
                values = self.value_net(states_tensor).squeeze()  # (T,)
                values_np = values.detach().cpu().numpy()

            advantages = np.zeros(len(rewards), dtype=np.float32)
            returns = np.zeros(len(rewards), dtype=np.float32)
            gae = 0.0
            next_value = 0.0

            for t in reversed(range(len(rewards))):
                delta = rewards[t] + self.discount_factor * next_value * (1 - dones[t]) - values_np[t]
                gae = delta + self.discount_factor * self.gae_lambda * (1 - dones[t]) * gae
                advantages[t] = gae
                returns[t] = gae + values_np[t]
                next_value = values_np[t]

            all_advantages.extend(advantages)
            all_returns.extend(returns)

        # devolver ya en el device correcto
        return (torch.as_tensor(all_advantages),
                torch.as_tensor(all_returns))

    
    # def maximize_objective(self, trajectories, advantages):
    #     states = []
    #     actions = []
    #     for trajectory in trajectories:
    #         for step in trajectory:
    #             states.append(step[0])
    #             actions.append(step[1])
    #     states_tensor = torch.FloatTensor(states).to(self.device)
    #     actions_tensor = torch.LongTensor(actions).to(self.device)
        
    #     old_action_probs = self.policy_net(states_tensor).gather(1, actions_tensor.unsqueeze(1)).detach()
        
    #     for _ in range(self.epochs):
    #         action_probs = self.policy_net(states_tensor).gather(1, actions_tensor.unsqueeze(1))
    #         ratios = action_probs / old_action_probs
    #         surr1 = ratios * advantages.unsqueeze(1)
    #         surr2 = torch.clamp(ratios, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages.unsqueeze(1)
    #         policy_loss = -torch.min(surr1, surr2).mean()
            
    #         self.policy_optimizer.zero_grad()
    #         policy_loss.backward()
    #         self.policy_optimizer.step()
    def maximize_objective(self, trajectories, advantages):
        loss_list, entropy_list = [], []
        states, actions = [], []
        for trajectory in trajectories:
            for s, a, *_ in trajectory:
                states.append(s); actions.append(a)

        states_np = np.array(states)      # junta todo en un solo array
        states_tensor = torch.from_numpy(states_np)
        actions_tensor = torch.as_tensor(actions, dtype=torch.long)

        # si mantienes “probabilities”, al menos asegúrate de consistencia de device
        with torch.no_grad():
            old_logits = self.policy_net(states_tensor)
            old_dist = torch.distributions.Categorical(logits=old_logits)
            old_log_probs = old_dist.log_prob(actions_tensor)

        for _ in range(self.K):
            logits = self.policy_net(states_tensor)
            dist = torch.distributions.Categorical(logits=logits)
            new_log_probs = dist.log_prob(actions_tensor)
            entropy = dist.entropy().mean()
            entropy_list.append(entropy.item())

            # ratio correcto en PPO
            ratios = torch.exp(new_log_probs - old_log_probs)

            adv = advantages  # ya está en self.device por el cambio anterior
            surr1 = ratios * adv
            surr2 = torch.clamp(ratios, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv
            policy_loss = -torch.min(surr1, surr2).mean()
            loss = policy_loss - self.entropy_coeficient * entropy
            loss_list.append(loss.item())  

            self.policy_optimizer.zero_grad()
            loss.backward()
            self.policy_optimizer.step()
        
        return loss_list, entropy_list


    # def fit_value_function(self, trajectories, returns):
    #     returns = returns.to(self.device)
    #     states = []
    #     for trajectory in trajectories:
    #         for step in trajectory:
    #             states.append(step[0])
    #     states_tensor = torch.FloatTensor(states).to(self.device)
        
    #     for _ in range(self.epochs):
    #         value_preds = self.value_net(states_tensor).squeeze()
    #         value_loss = self.mse_loss(value_preds, returns)
            
    #         self.value_optimizer.zero_grad()
    #         value_loss.backward()
    #         self.value_optimizer.step()

    def fit_value_function(self, trajectories, returns):
        #returns = returns.to(self.device)
        states = []
        for trajectory in trajectories:
            for s, *_ in trajectory:
                states.append(s)
        states_np = np.asarray(states)
        states_tensor = torch.from_numpy(states_np)

        for _ in range(self.epochs):
            value_preds = self.value_net(states_tensor).squeeze()
            value_loss = self.mse_loss(value_preds, returns)
            self.value_optimizer.zero_grad()
            value_loss.backward()
            self.value_optimizer.step()


    def train(self, env, save_file_policy, save_file_value):   
        for path in [save_file_policy, save_file_value]:
            if path is not None:
                directory = os.path.dirname(path)
                if directory != "":
                    os.makedirs(directory, exist_ok=True)
        rewards = []
        loss_all = []
        entropy_all = []
        for k in range(self.epochs):
            trajectories, reward = self.collect_trajectories(env)
            advantages, returns = self.estimate_advantages(trajectories)
            loss, entropy = self.maximize_objective(trajectories, advantages)
            self.fit_value_function(trajectories, returns)
            loss_all.append(loss[-1])
            entropy_all.append(entropy[-1])
            rewards.append(sum(reward))
            #if (k+1) % 10 == 0:
            print(f"Epoch {k+1}/{self.epochs} -- reward: {sum(reward):.2f}")
            if k % 20 == 0:
              local_path = f"policy_weights_epoch_{k}.pth"
              torch.save(self.policy_net.state_dict(), local_path)
              upload_to_gcs(
                  local_path=local_path,
                  bucket_name=BUCKET_NAME,
                  destination_blob_name=f"checkpoints/{local_path}"
              )

              local_path = f"value_weights_epoch_{k}.pth"
              torch.save(self.value_net.state_dict(), local_path)
              upload_to_gcs(
                  local_path=local_path,
                  bucket_name=BUCKET_NAME,
                  destination_blob_name=f"checkpoints/{local_path}"
              )

              metrics = {
                "rewards": rewards,          # pueden ser listas o np.array
                "loss": loss,
                "entropy": entropy,
              }

              local_metrics_path = f"metrics_epoch_{k}.pt"
              torch.save(metrics, local_metrics_path)

              upload_to_gcs(
                  local_path=local_metrics_path,
                  bucket_name=BUCKET_NAME,
                  destination_blob_name=f"checkpoints/metrics_epoch_{k}.pt"
              )
              #save_in_gc(k, self.policy_net.state_dict(), self.value_net.state_dict(), rewards, loss_all, entropy_all)
            torch.save(self.policy_net.state_dict(), save_file_policy)
            torch.save(self.value_net.state_dict(), save_file_value)
        return rewards, loss_all, entropy_all
    

    # def evaluate(self, env):
    #     rewards = []
    #     state, _ = env.reset()
    #     state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
    #     while True:
    #         with torch.no_grad():
    #             logits = self.policy_net(state_tensor)
    #             dist = torch.distributions.Categorical(logits=logits)
    #             action = dist.sample().item()

    #         # Processing:
    #         obs, reward, terminated, truncated, info = env.step(action)
    #         rewards.append(reward)

    #         state_tensor = torch.FloatTensor(obs).unsqueeze(0)
    #         # Checking if the player is still alive
    #         if terminated:
    #             break

    #     env.close()

    #     return rewards

    def evaluate(self, env, num_episodes=1):
        rewards = []
        for _ in range(num_episodes):
            ep_rewards = []
            i = 0
            state, _ = env.reset()
            state_tensor = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
            while True:
                with torch.no_grad():
                    logits = self.policy_net(state_tensor)
                    dist = torch.distributions.Categorical(logits=logits)
                    action = dist.sample().item()

                obs, reward, terminated, truncated, info = env.step(action)
                rewards.append(reward)
                ep_rewards.append(reward)
                i += 1

                state_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
                if terminated or truncated:
                    print(f"reward: {sum(ep_rewards)} en {i} pasos")
                    break

        env.close()
        return rewards


       
        


# Example usage:
# Define your policy_net and value_net as instances of nn.Module
# env = YourEnvironment()
# ppo = PPO_Clip(epochs=10, batch_size=5, policy_net=policy_net, value_net=value_net, clip_param=0.2, lr=3e-4, discount_factor=0.99, gae_lambda=0.95)
# ppo.train(env)


