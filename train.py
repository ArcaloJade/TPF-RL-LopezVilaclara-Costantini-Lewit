from PPOClip_Completo import PPO_Clip_Completo
from PPOClip_basic import PPO_Clip_Basic
from PPOClip_entropy import PPO_Clip_Entropy
from PPOClip_GAE import PPO_Clip_GAE
from PPOClip_Annealing import PPO_Clip_Annealing

from ValueNet import ValueNet
from PolicyNet import PolicyNet
import matplotlib.pyplot as plt 
import torch
import flappy_bird_gymnasium
import gymnasium
import numpy as np

import argparse

# sys.argv es una lista de strings:
# sys.argv[0] = nombre del script
# sys.argv[1] = primer argumento
# sys.argv[2] = segundo, etc.




def plot_metrics(modelo_entrenado, hiperparams):
    reward = torch.load(f"trained_net/{modelo_entrenado}/metrics.pt")["rewards"]
    plt.plot(np.linspace(0, hiperparams["epochs"], len(reward)),reward)
    plt.title("Recompensa entrenamiento")
    plt.show()

    entropy = torch.load(f"trained_net/{modelo_entrenado}/metrics.pt")["entropy"]
    if entropy is not None:
        plt.plot(np.linspace(0, hiperparams["epochs"], len(entropy)),entropy)
        plt.title("Entropía entrenamiento")
        plt.show()

    loss = torch.load(f"trained_net/{modelo_entrenado}/metrics.pt")["loss"]
    plt.plot(np.linspace(0, hiperparams["epochs"], len(loss)),loss)
    plt.title("Loss entrenamiento")
    plt.show()




def main():
    parser = argparse.ArgumentParser(description="Entrenamiento PPO")

    parser.add_argument("--modelo", type=str, default="All",
                        help="Modelo a entrenar: 1-Basic, 2-GAE, 3-Entropy, 4-Annealing, 5-Completo, 6-All")
    parser.add_argument("--K", type=int, default=80,
                        help="Iteraciones de optimización por epoch")
    parser.add_argument("--epochs", type=int, default=500,
                        help="Cantidad de epochs de entrenamiento")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Tamaño del batch de trayectorias")
    parser.add_argument("--clip_param", type=float, default=0.13,
                        help="Parámetro de clipping epsilon")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Learning rate")
    parser.add_argument("--discount_factor", type=float, default=0.95,
                        help="Factor de descuento gamma")
    parser.add_argument("--gae_lambda", type=float, default=3e-4,
                        help="Parámetro lambda para GAE")
    parser.add_argument("--entropy_coeficient", type=float, default=0.01,
                        help="Coeficiente de entropía")
    parser.add_argument("--max_length", type=int, default=7000,
                        help="Maxima longitud de una trayectoria")
    parser.add_argument("--modelo_entrenado", type=str, default="PPOClip",
                        help="Maxima longitud de una trayectoria")

    args = parser.parse_args()

    print("Configuración:")
    print(f"  epochs     = {args.epochs}")
    print(f"  lr         = {args.lr}")
    print(f"  max_length = {args.max_length}")

    env = gymnasium.make("FlappyBird-v0", use_lidar=False)
    policy_net = PolicyNet(12, 2)
    value_net = ValueNet(12, 1)
    hiperparams = {
        "epochs": args.epochs, #dejar 1500,
        "K": args.K,
        "batch_size": args.batch_size,
        "policy_net": policy_net, 
        "value_net": value_net,
        "clip_param": args.clip_param, 
        "lr": args.lr, 
        "discount_factor": args.discount_factor, 
        "gae_lambda": args.gae_lambda,
        "entropy_coeficient": args.entropy_coeficient,
        "max_length": args.max_length
        
    }
    if args.modelo == "Basic":
        PPO = PPO_Clip_Basic(hiperparams.copy() )
        PPOs = [PPO]
    elif args.modelo == "GAE":
        PPO = PPO_Clip_GAE(hiperparams.copy() )
        PPOs = [PPO]
    elif args.modelo == "Entropy":
        PPO = PPO_Clip_Entropy(hiperparams.copy() )
        PPOs = [PPO]
    elif args.modelo == "Annealing":
        PPO = PPO_Clip_Annealing(hiperparams.copy() )
        PPOs = [PPO]
    elif args.modelo == "Completo":
        PPO = PPO_Clip_Completo(hiperparams.copy() )
        PPOs = [PPO]
    elif args.modelo == "All":
        PPO_basic = PPO_Clip_Basic(hiperparams.copy() )
        PPO_gae = PPO_Clip_GAE(hiperparams.copy() )
        PPO_entropy = PPO_Clip_Entropy(hiperparams.copy() )
        PPO_annealing = PPO_Clip_Annealing(hiperparams.copy() )
        PPO_completo = PPO_Clip_Completo(hiperparams.copy() )
        PPOs = [PPO_basic, PPO_gae, PPO_entropy, PPO_annealing, PPO_completo]
        
    else:
        raise ValueError("Modelo no reconocido. Usar: Basic, GAE, Entropy, Annealing, Completo")
    
    for i, PPO in enumerate(PPOs):
            save_policy_file = f"trained_net/PPO_Model_{args.modelo}_{i}/flappy_actor.pth"
            save_value_file = f"trained_net/PPO_Model_{args.modelo}_{i}/flappy_critic.pth"
            save_metric_file = f"trained_net/PPO_Model_{args.modelo}_{i}/metrics.pt"
            rewards, loss, entropy = PPO.train(env, save_file_policy=save_policy_file, save_file_value=save_value_file, save_metrics_file=save_metric_file)
    return
    # acá iría tu código de entrenamiento usando args.*
    # train(env_name=args.env, epochs=args.epochs, lr=args.lr, batch_size=args.batch_size)

if __name__ == "__main__":
    main()