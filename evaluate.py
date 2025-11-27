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



def eval(args, hiperparams, load_file):
    modelos_disponibles = {
        "Basic": PPO_Clip_Basic,
        "GAE": PPO_Clip_GAE,
        "Entropy": PPO_Clip_Entropy,
        "Annealing": PPO_Clip_Annealing,
        "Completo": PPO_Clip_Completo,
    }

    if args.view == "human":  
        env = gymnasium.make("FlappyBird-v0", render_mode = "human" ,use_lidar=False)
    else:
        env = gymnasium.make("FlappyBird-v0" ,use_lidar=False)
    policy_net = PolicyNet(12, 2)
    value_net = ValueNet(12, 1)

    PPO = modelos_disponibles[args.modelo](hiperparams)

    policy_net.load_state_dict(torch.load(f"{load_file}/flappy_actor.pth"))
    policy_net.eval()

    PPO.policy_net = policy_net

    PPO.evaluate(env, 1)
    env.close()






def main():
    parser = argparse.ArgumentParser(description="Entrenamiento PPO")

    parser.add_argument("--modelo", type=str, default="All",
                        help="Modelo a entrenar: Basic, GAE, Entropy, Annealing, Completo, All")
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
    parser.add_argument("--gae_lambda", type=float, default=0.95,
                        help="Parámetro lambda para GAE")
    parser.add_argument("--entropy_coeficient", type=float, default=0.01,
                        help="Coeficiente de entropía")
    parser.add_argument("--max_length", type=int, default=7000,
                        help="Maxima longitud de una trayectoria")
    parser.add_argument("--folder_name", type=str, default="PPOClip",
                        help="Nombre base del modelo entrenado")
    parser.add_argument("--view", type=str, default="none",
                        help="Modo de visualización del entorno")

    args = parser.parse_args()

    print("Configuración:")
    print(f"  epochs     = {args.epochs}")
    print(f"  modelo     = {args.modelo}")
    print(f"  max_length = {args.max_length}")

    
    hiperparams = {
        "epochs": args.epochs,
        "K": args.K,
        "batch_size": args.batch_size,
        "policy_net": PolicyNet(12, 2), # 12 entradas, 2 acciones
        "value_net": ValueNet(12, 1),   # 12 entradas, 1 valor
        "clip_param": args.clip_param,
        "lr": args.lr,
        "discount_factor": args.discount_factor,
        "gae_lambda": args.gae_lambda,
        "max_length": args.max_length,
        "entropy_coeficient": args.entropy_coeficient,
    }
    load_file = f"trained_net/{args.folder_name}"

    eval(args, hiperparams, load_file)

    return

if __name__ == "__main__":
    main()