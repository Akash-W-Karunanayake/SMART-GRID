"""
evaluate.py
===========
Evaluate trained MARL agent on HELD-OUT fault buses (not in training set).
Supports both GNN-enhanced and flat-only models.

Usage:
  python -m scripts.evaluate --model models/marl_checkpoint.pt --scenarios 100
"""
import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from loguru import logger
from config import settings, CRITICAL_LOADS
from app.ml.environment.grid_env import GridRestorationEnv
from app.ml.agents.marl_agent import MARLAgent

# HELD-OUT: different nodes than training set
TEST_FAULTS = [
    "F06_Node2", "F07_Node3", "F08_Node2", "F09_Node3",
    "F10_Node2", "F11_Node2", "F12_Node3",
]


def evaluate(args):
    env = GridRestorationEnv(
        master_dss_path=str(settings.master_dss_path),
        max_steps=args.max_steps,
    )

    settings.TRAINED_MODEL_PATH = args.model
    agent = MARLAgent.load_latest()
    if not agent:
        logger.error(f"No model at {args.model}")
        sys.exit(1)

    logger.info(f"Model loaded (use_gnn={agent.use_gnn})")

    results = []
    for i in range(args.scenarios):
        fb = random.choice(TEST_FAULTS)
        ft = random.choice(["lg", "ll", "3phase"])

        obs, info = env.reset(options={"fault_bus": fb, "fault_type": ft})
        graph_data = info.get("graph_data")
        total_r = 0.0

        for _ in range(args.max_steps):
            acts = agent.select_actions(obs, graph_data=graph_data, greedy=True)
            obs, r, done, _, info = env.step(acts)
            graph_data = info.get("graph_data")
            total_r += r
            if done:
                break

        de = info.get("de_energized", [])
        results.append({
            "fault_bus": fb,
            "fault_type": ft,
            "reward": total_r,
            "offline": len(de),
            "violations": len(info.get("violations", [])),
            "overloads": len(info.get("overloads", [])),
            "critical_ok": all(cl not in de for cl in CRITICAL_LOADS),
            "converged": info.get("converged", False),
            "num_toggled": info.get("num_toggled", 0),
        })

    rewards = [r["reward"] for r in results]
    offline = [r["offline"] for r in results]
    violations = [r["violations"] for r in results]
    overloads = [r["overloads"] for r in results]
    crit = sum(1 for r in results if r["critical_ok"]) / len(results) * 100
    converged = sum(1 for r in results if r["converged"]) / len(results) * 100
    toggled = [r["num_toggled"] for r in results]

    logger.info("=" * 60)
    logger.info(f"Evaluation: {args.scenarios} scenarios (held-out buses)")
    logger.info(f"  Model:              {args.model} (GNN={'yes' if agent.use_gnn else 'no'})")
    logger.info(f"  Avg reward:         {np.mean(rewards):.2f} +/- {np.std(rewards):.2f}")
    logger.info(f"  Avg loads offline:  {np.mean(offline):.1f}")
    logger.info(f"  Avg violations:     {np.mean(violations):.1f}")
    logger.info(f"  Avg overloads:      {np.mean(overloads):.1f}")
    logger.info(f"  Avg switches used:  {np.mean(toggled):.1f}")
    logger.info(f"  Hospital restored:  {crit:.0f}%")
    logger.info(f"  Convergence rate:   {converged:.0f}%")
    logger.info("=" * 60)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="models/marl_checkpoint.pt")
    p.add_argument("--scenarios", type=int, default=100)
    p.add_argument("--max-steps", type=int, default=10)
    p.add_argument("--seed", type=int, default=123)
    a = p.parse_args()
    random.seed(a.seed)
    np.random.seed(a.seed)
    evaluate(a)
