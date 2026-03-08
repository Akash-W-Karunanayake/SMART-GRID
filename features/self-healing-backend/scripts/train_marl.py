"""
train_marl.py
=============
Online RL training loop for the 5-agent tie-switch MARL system
with GNN-enhanced topology-aware state representations.

The GNN encoder processes the grid graph each step, producing per-node
embeddings that give agents topology-aware context for Q-value estimation.

Usage:
  python -m scripts.train_marl --episodes 2000 --max-steps 10
  python -m scripts.train_marl --episodes 2000 --no-gnn    # flat-only baseline
"""
import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from loguru import logger
from config import settings

# Suppress noisy DEBUG logs from grid_graph (fires ~20x per episode)
logger.remove()
logger.add(sys.stderr, level="INFO")
from app.ml.environment.grid_env import GridRestorationEnv
from app.ml.agents.marl_agent import MARLAgent

TRAIN_FAULTS = [
    "F06_Node3", "F07_Node2", "F08_Node3", "F09_Node4",
    "F10_Node3", "F11_Node3", "F12_Node2",
    "F06_Node4", "F09_Node2", "F11_Node4",
]
FAULT_TYPES = ["lg", "ll", "3phase", "llg"]


def train(args):
    env = GridRestorationEnv(
        master_dss_path=str(settings.master_dss_path),
        max_steps=args.max_steps,
    )

    obs, info = env.reset(options={"fault_bus": "F09_Node4", "fault_type": "lg"})
    obs_dim = len(list(obs.values())[0])
    graph_data = info.get("graph_data")

    use_gnn = args.use_gnn
    logger.info(
        f"Obs dim per agent (flat): {obs_dim}, Agents: {env.n_agents}, "
        f"GNN: {'enabled' if use_gnn else 'disabled'}"
    )

    agent = MARLAgent(
        obs_dim=obs_dim,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        gamma=args.gamma,
        batch_size=args.batch_size,
        device=args.device,
        use_gnn=use_gnn,
        gnn_embed_dim=args.gnn_embed_dim,
        gnn_hidden_dim=args.gnn_hidden_dim,
    )

    if use_gnn and agent.use_gnn:
        logger.info(
            f"GNN encoder: in_features=1, hidden={args.gnn_hidden_dim}, "
            f"embed={args.gnn_embed_dim}, agent_gnn_dim={args.gnn_embed_dim * 3}"
        )

    best_avg = -float("inf")
    history = []

    for ep in range(1, args.episodes + 1):
        fb = random.choice(TRAIN_FAULTS)
        ft = random.choice(FAULT_TYPES)

        obs, info = env.reset(options={"fault_bus": fb, "fault_type": ft})
        graph_data = info.get("graph_data")
        ep_reward = 0.0

        # Pre-compute GNN embeddings for initial state
        gnn_embed = None
        if agent.use_gnn and graph_data:
            gnn_embed = agent.compute_gnn_embeddings(graph_data)

        for _ in range(args.max_steps):
            actions = agent.select_actions(obs, graph_data=graph_data)
            nxt, reward, terminated, truncated, info = env.step(actions)
            done = terminated or truncated

            next_graph_data = info.get("graph_data")
            next_gnn_embed = None
            if agent.use_gnn and next_graph_data:
                next_gnn_embed = agent.compute_gnn_embeddings(next_graph_data)

            agent.store(
                obs, actions, reward, nxt, done,
                gnn_embed=gnn_embed, next_gnn_embed=next_gnn_embed,
            )
            agent.update()

            ep_reward += reward
            obs = nxt
            graph_data = next_graph_data
            gnn_embed = next_gnn_embed

            if done:
                break

        history.append(ep_reward)

        if ep % 50 == 0:
            avg = np.mean(history[-50:])
            logger.info(
                f"Ep {ep}/{args.episodes} | R={ep_reward:.1f} | "
                f"Avg50={avg:.1f} | eps={agent.epsilon:.3f}"
            )
            if avg > best_avg:
                best_avg = avg
                Path(args.output_dir).mkdir(parents=True, exist_ok=True)
                agent.save(f"{args.output_dir}/marl_checkpoint.pt")

    agent.save(f"{args.output_dir}/marl_final.pt")
    logger.info(f"Done. Best avg50: {best_avg:.1f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=2000)
    p.add_argument("--max-steps", type=int, default=10)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--seed", type=int, default=42)
    # GNN hyperparameters
    p.add_argument("--use-gnn", action="store_true", default=True,
                    help="Enable GNN topology encoder (default: True)")
    p.add_argument("--no-gnn", dest="use_gnn", action="store_false",
                    help="Disable GNN, use flat observations only (baseline)")
    p.add_argument("--gnn-embed-dim", type=int, default=32)
    p.add_argument("--gnn-hidden-dim", type=int, default=64)
    p.add_argument("--output-dir", type=str, default="models",
                    help="Directory to save checkpoints")
    a = p.parse_args()
    # Resolve output_dir relative to project root (not CWD, which OpenDSS changes)
    a.output_dir = str(Path(__file__).resolve().parent.parent / a.output_dir)
    random.seed(a.seed)
    np.random.seed(a.seed)
    train(a)
