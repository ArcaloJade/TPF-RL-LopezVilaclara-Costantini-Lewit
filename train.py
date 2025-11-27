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
    parser.add_argument("--modelo_entrenado", type=str, default="PPOClip",
                        help="Nombre base del modelo entrenado")

    args = parser.parse_args()

    print("Configuración:")
    print(f"  epochs     = {args.epochs}")
    print(f"  modelo     = {args.modelo}")
    print(f"  max_length = {args.max_length}")

    # Mapeo de nombre → clase
    modelos_disponibles = {
        "Basic": PPO_Clip_Basic,
        "GAE": PPO_Clip_GAE,
        "Entropy": PPO_Clip_Entropy,
        "Annealing": PPO_Clip_Annealing,
        "Completo": PPO_Clip_Completo,
    }

    if args.modelo == "All":
        modelos_a_entrenar = list(modelos_disponibles.keys())
    else:
        if args.modelo not in modelos_disponibles:
            raise ValueError("Modelo no reconocido. Usar: Basic, GAE, Entropy, Annealing, Completo, All")
        modelos_a_entrenar = [args.modelo]

    for i, nombre_modelo in enumerate(modelos_a_entrenar):
        print(f"\n=== Entrenando modelo: {nombre_modelo} (índice {i}) ===")

        # Nuevo entorno y nuevas redes PARA CADA MODELO
        env = gymnasium.make("FlappyBird-v0", use_lidar=False)
        policy_net = PolicyNet(12, 2)
        value_net = ValueNet(12, 1)

        hiperparams = {
            "epochs": args.epochs,
            "K": args.K,
            "batch_size": args.batch_size,
            "policy_net": policy_net,
            "value_net": value_net,
            "clip_param": args.clip_param,
            "lr": args.lr,
            "discount_factor": args.discount_factor,
            "gae_lambda": args.gae_lambda,
            "entropy_coeficient": args.entropy_coeficient,
            "max_length": args.max_length,
        }

        PPO_Class = modelos_disponibles[nombre_modelo]
        PPO = PPO_Class(hiperparams)

        save_policy_file = f"trained_net/PPO_Model_{nombre_modelo}_{i}/flappy_actor.pth"
        save_value_file = f"trained_net/PPO_Model_{nombre_modelo}_{i}/flappy_critic.pth"
        save_metric_file = f"trained_net/PPO_Model_{nombre_modelo}_{i}/metrics.pt"

        rewards, loss, entropy = PPO.train(
            env,
            save_file_policy=save_policy_file,
            save_file_value=save_value_file,
            save_metrics_file=save_metric_file,
        )

    return

    # acá iría tu código de entrenamiento usando args.*
    # train(env_name=args.env, epochs=args.epochs, lr=args.lr, batch_size=args.batch_size)

if __name__ == "__main__":
    main()