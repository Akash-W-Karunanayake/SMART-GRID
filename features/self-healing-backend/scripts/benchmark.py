"""
benchmark.py
============
Phase 3: Comprehensive evaluation & benchmarking of all baselines.

Compares 4 strategies on held-out fault scenarios:
  1. Random    — uniformly random tie switch actions each step
  2. Heuristic — greedy: try each tie switch, keep if it restores load
  3. MARL Flat — trained DQN without GNN (flat observations only)
  4. MARL+GNN  — trained DQN with GNN topology encoder

Metrics collected per scenario:
  - Total episode reward
  - Number of loads restored (from initial de-energized)
  - Restoration percentage
  - Voltage violations remaining
  - Overloaded lines remaining
  - Critical loads restored (hospital)
  - Power flow convergence
  - Number of switching operations

Usage:
  python -m scripts.benchmark --scenarios 100
  python -m scripts.benchmark --scenarios 50 --seed 999
"""
import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from loguru import logger
from config import settings, CRITICAL_LOADS, TIE_SWITCH_NAMES, ALL_LOAD_NAMES
from app.ml.environment.grid_env import GridRestorationEnv
from app.ml.agents.marl_agent import MARLAgent
from app.services.grid_graph import build_grid_graph, is_radial

# Suppress noisy DEBUG logs
logger.remove()
logger.add(sys.stderr, level="INFO")

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Held-out fault buses (NOT in training set)
TEST_FAULTS = [
    "F06_Node2", "F07_Node3", "F08_Node2", "F09_Node3",
    "F10_Node2", "F11_Node2", "F12_Node3",
]
FAULT_TYPES = ["lg", "ll", "3phase", "llg"]


def run_episode(env, action_fn, max_steps):
    """Run one episode with a given action function. Returns metrics dict."""
    obs, info = env.reset(
        options={"fault_bus": env.fault_bus, "fault_type": env.fault_type}
    )
    graph_data = info.get("graph_data")
    initial_de_energized = list(env._initial_de_energized)
    total_reward = 0.0
    total_toggled = 0

    for step in range(max_steps):
        actions = action_fn(obs, graph_data, step)
        obs, r, terminated, truncated, info = env.step(actions)
        graph_data = info.get("graph_data")
        total_reward += r
        total_toggled += info.get("num_toggled", 0)
        if terminated or truncated:
            break

    de_energized = info.get("de_energized", [])
    energized_loads = set(ALL_LOAD_NAMES) - set(de_energized)

    # Loads restored from the initial outage
    restored = [ld for ld in initial_de_energized if ld in energized_loads]
    n_initial_off = len(initial_de_energized)
    restoration_pct = (len(restored) / n_initial_off * 100) if n_initial_off > 0 else 100.0

    return {
        "reward": total_reward,
        "loads_restored": len(restored),
        "loads_initially_off": n_initial_off,
        "restoration_pct": restoration_pct,
        "loads_still_off": len(de_energized),
        "violations": len(info.get("violations", [])),
        "overloads": len(info.get("overloads", [])),
        "critical_ok": all(cl not in de_energized for cl in CRITICAL_LOADS),
        "converged": info.get("converged", False),
        "switches_used": total_toggled,
    }


# ── Action Functions ──────────────────────────────────────────────────────

def make_random_action_fn():
    """Random: each agent independently picks 0 or 1."""
    def fn(obs, graph_data, step):
        return {sw: random.randint(0, 1) for sw in TIE_SWITCH_NAMES}
    return fn


def make_heuristic_action_fn(env):
    """Greedy heuristic: try closing each tie switch, keep if it adds load."""
    def fn(obs, graph_data, step):
        actions = {}
        current_states = env.engine.get_tie_switch_states()
        for sw in TIE_SWITCH_NAMES:
            actions[sw] = 1 if current_states.get(sw, False) else 0

        # Try closing each open tie switch
        for sw in TIE_SWITCH_NAMES:
            if not current_states.get(sw, False):
                # Test closing this switch
                actions[sw] = 1
        return actions
    return fn


def make_marl_action_fn(agent):
    """Trained MARL agent (flat or GNN)."""
    def fn(obs, graph_data, step):
        return agent.select_actions(obs, graph_data=graph_data, greedy=True)
    return fn


def make_do_nothing_fn():
    """Do nothing: keep all tie switches open (post-isolation baseline)."""
    def fn(obs, graph_data, step):
        return {sw: 0 for sw in TIE_SWITCH_NAMES}
    return fn


# ── Benchmark Runner ──────────────────────────────────────────────────────

def run_benchmark(env, action_fn, label, scenarios, max_steps, fault_list):
    """Run all scenarios for one strategy, return list of result dicts."""
    results = []
    for i, (fb, ft) in enumerate(fault_list):
        env.fault_bus = fb
        env.fault_type = ft
        metrics = run_episode(env, action_fn, max_steps)
        metrics["fault_bus"] = fb
        metrics["fault_type"] = ft
        metrics["strategy"] = label
        results.append(metrics)

        if (i + 1) % 25 == 0:
            avg_r = np.mean([r["reward"] for r in results])
            avg_rest = np.mean([r["restoration_pct"] for r in results])
            logger.info(f"  [{label}] {i+1}/{len(fault_list)} | AvgR={avg_r:.1f} | Rest%={avg_rest:.1f}%")

    return results


def print_summary(label, results):
    """Print summary statistics for one strategy."""
    rewards = [r["reward"] for r in results]
    rest_pcts = [r["restoration_pct"] for r in results]
    violations = [r["violations"] for r in results]
    overloads = [r["overloads"] for r in results]
    switches = [r["switches_used"] for r in results]
    crit_rate = sum(1 for r in results if r["critical_ok"]) / len(results) * 100
    conv_rate = sum(1 for r in results if r["converged"]) / len(results) * 100

    logger.info(f"  {'Metric':<25} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
    logger.info(f"  {'-'*65}")
    logger.info(f"  {'Reward':<25} {np.mean(rewards):>10.1f} {np.std(rewards):>10.1f} {np.min(rewards):>10.1f} {np.max(rewards):>10.1f}")
    logger.info(f"  {'Restoration %':<25} {np.mean(rest_pcts):>10.1f} {np.std(rest_pcts):>10.1f} {np.min(rest_pcts):>10.1f} {np.max(rest_pcts):>10.1f}")
    logger.info(f"  {'Voltage violations':<25} {np.mean(violations):>10.1f} {np.std(violations):>10.1f} {np.min(violations):>10.0f} {np.max(violations):>10.0f}")
    logger.info(f"  {'Overloaded lines':<25} {np.mean(overloads):>10.1f} {np.std(overloads):>10.1f} {np.min(overloads):>10.0f} {np.max(overloads):>10.0f}")
    logger.info(f"  {'Switches toggled':<25} {np.mean(switches):>10.1f} {np.std(switches):>10.1f} {np.min(switches):>10.0f} {np.max(switches):>10.0f}")
    logger.info(f"  {'Critical load OK':<25} {crit_rate:>10.0f}%")
    logger.info(f"  {'Convergence rate':<25} {conv_rate:>10.0f}%")

    return {
        "strategy": label,
        "reward_mean": float(np.mean(rewards)),
        "reward_std": float(np.std(rewards)),
        "restoration_pct_mean": float(np.mean(rest_pcts)),
        "restoration_pct_std": float(np.std(rest_pcts)),
        "violations_mean": float(np.mean(violations)),
        "overloads_mean": float(np.mean(overloads)),
        "switches_mean": float(np.mean(switches)),
        "critical_load_pct": crit_rate,
        "convergence_pct": conv_rate,
    }


def main(args):
    env = GridRestorationEnv(
        master_dss_path=str(settings.master_dss_path),
        max_steps=args.max_steps,
    )

    # Generate deterministic fault list (same for all strategies)
    rng = random.Random(args.seed)
    fault_list = [
        (rng.choice(TEST_FAULTS), rng.choice(FAULT_TYPES))
        for _ in range(args.scenarios)
    ]

    strategies = {}

    # 1. Do-Nothing baseline
    logger.info("=" * 60)
    logger.info("Strategy: Do-Nothing (post-isolation, no restoration)")
    random.seed(args.seed); np.random.seed(args.seed)
    strategies["do_nothing"] = run_benchmark(
        env, make_do_nothing_fn(), "Do-Nothing", args.scenarios, args.max_steps, fault_list
    )

    # 2. Random baseline
    logger.info("=" * 60)
    logger.info("Strategy: Random")
    random.seed(args.seed); np.random.seed(args.seed)
    strategies["random"] = run_benchmark(
        env, make_random_action_fn(), "Random", args.scenarios, args.max_steps, fault_list
    )

    # 3. Heuristic baseline
    logger.info("=" * 60)
    logger.info("Strategy: Heuristic (close all open ties)")
    random.seed(args.seed); np.random.seed(args.seed)
    strategies["heuristic"] = run_benchmark(
        env, make_heuristic_action_fn(env), "Heuristic", args.scenarios, args.max_steps, fault_list
    )

    # 4. MARL Flat
    flat_path = str(PROJECT_ROOT / "models" / "flat_v3" / "marl_checkpoint.pt")
    if Path(flat_path).exists():
        logger.info("=" * 60)
        logger.info(f"Strategy: MARL-Flat (from {flat_path})")
        random.seed(args.seed); np.random.seed(args.seed)
        settings.TRAINED_MODEL_PATH = flat_path
        flat_agent = MARLAgent.load_latest()
        if flat_agent:
            strategies["marl_flat"] = run_benchmark(
                env, make_marl_action_fn(flat_agent), "MARL-Flat", args.scenarios, args.max_steps, fault_list
            )
    else:
        logger.warning(f"Flat model not found at {flat_path}, skipping")

    # 5. MARL+GNN
    gnn_path = str(PROJECT_ROOT / "models" / "gnn_v3" / "marl_checkpoint.pt")
    if Path(gnn_path).exists():
        logger.info("=" * 60)
        logger.info(f"Strategy: MARL+GNN (from {gnn_path})")
        random.seed(args.seed); np.random.seed(args.seed)
        settings.TRAINED_MODEL_PATH = gnn_path
        gnn_agent = MARLAgent.load_latest()
        if gnn_agent:
            strategies["marl_gnn"] = run_benchmark(
                env, make_marl_action_fn(gnn_agent), "MARL+GNN", args.scenarios, args.max_steps, fault_list
            )
    else:
        logger.warning(f"GNN model not found at {gnn_path}, skipping")

    # ── Print Summary ─────────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("BENCHMARK RESULTS SUMMARY")
    logger.info("=" * 60)

    summaries = []
    for name, results in strategies.items():
        logger.info(f"\n--- {name.upper()} ---")
        s = print_summary(name, results)
        summaries.append(s)

    # ── Comparison Table ──────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("COMPARISON TABLE")
    logger.info("=" * 60)
    header = f"  {'Strategy':<15} {'Reward':>10} {'Rest%':>8} {'Viol':>6} {'Over':>6} {'Crit%':>7} {'Conv%':>7}"
    logger.info(header)
    logger.info(f"  {'-'*60}")
    for s in summaries:
        logger.info(
            f"  {s['strategy']:<15} {s['reward_mean']:>10.1f} {s['restoration_pct_mean']:>8.1f} "
            f"{s['violations_mean']:>6.1f} {s['overloads_mean']:>6.1f} "
            f"{s['critical_load_pct']:>7.0f} {s['convergence_pct']:>7.0f}"
        )

    # ── Statistical Tests ─────────────────────────────────────────────────
    if "marl_flat" in strategies and "marl_gnn" in strategies:
        from scipy import stats
        flat_rewards = [r["reward"] for r in strategies["marl_flat"]]
        gnn_rewards = [r["reward"] for r in strategies["marl_gnn"]]
        flat_rest = [r["restoration_pct"] for r in strategies["marl_flat"]]
        gnn_rest = [r["restoration_pct"] for r in strategies["marl_gnn"]]

        # Paired t-test (same fault scenarios)
        t_r, p_r = stats.ttest_rel(flat_rewards, gnn_rewards)
        t_rest, p_rest = stats.ttest_rel(flat_rest, gnn_rest)
        # Mann-Whitney U (non-parametric)
        u_r, up_r = stats.mannwhitneyu(flat_rewards, gnn_rewards, alternative="two-sided")
        u_rest, up_rest = stats.mannwhitneyu(flat_rest, gnn_rest, alternative="two-sided")

        logger.info("\n" + "=" * 60)
        logger.info("STATISTICAL TESTS: MARL-Flat vs MARL+GNN")
        logger.info("=" * 60)
        logger.info(f"  Reward:       paired t={t_r:.3f}, p={p_r:.4f} {'*' if p_r < 0.05 else ''}")
        logger.info(f"  Reward:       Mann-Whitney U={u_r:.0f}, p={up_r:.4f} {'*' if up_r < 0.05 else ''}")
        logger.info(f"  Restoration:  paired t={t_rest:.3f}, p={p_rest:.4f} {'*' if p_rest < 0.05 else ''}")
        logger.info(f"  Restoration:  Mann-Whitney U={u_rest:.0f}, p={up_rest:.4f} {'*' if up_rest < 0.05 else ''}")
        logger.info(f"  (* = significant at p<0.05)")

    # ── Per Fault-Type Breakdown ──────────────────────────────────────────
    if "marl_gnn" in strategies:
        logger.info("\n" + "=" * 60)
        logger.info("PER FAULT-TYPE BREAKDOWN (MARL+GNN)")
        logger.info("=" * 60)
        for ft in FAULT_TYPES:
            ft_results = [r for r in strategies["marl_gnn"] if r["fault_type"] == ft]
            if ft_results:
                avg_r = np.mean([r["reward"] for r in ft_results])
                avg_rest = np.mean([r["restoration_pct"] for r in ft_results])
                n = len(ft_results)
                logger.info(f"  {ft:<10} n={n:>3} | AvgReward={avg_r:>8.1f} | RestPct={avg_rest:>5.1f}%")

    # ── Per Fault-Bus Breakdown ───────────────────────────────────────────
    if "marl_gnn" in strategies:
        logger.info("\n" + "=" * 60)
        logger.info("PER FAULT-BUS BREAKDOWN (MARL+GNN)")
        logger.info("=" * 60)
        for fb in TEST_FAULTS:
            fb_results = [r for r in strategies["marl_gnn"] if r["fault_bus"] == fb]
            if fb_results:
                avg_r = np.mean([r["reward"] for r in fb_results])
                avg_rest = np.mean([r["restoration_pct"] for r in fb_results])
                n = len(fb_results)
                logger.info(f"  {fb:<15} n={n:>3} | AvgReward={avg_r:>8.1f} | RestPct={avg_rest:>5.1f}%")

    # ── Save Results ──────────────────────────────────────────────────────
    output_dir = PROJECT_ROOT / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "benchmark_results.json"
    save_data = {
        "config": {
            "scenarios": args.scenarios,
            "max_steps": args.max_steps,
            "seed": args.seed,
            "test_faults": TEST_FAULTS,
            "fault_types": FAULT_TYPES,
        },
        "summaries": summaries,
        "detailed": {name: results for name, results in strategies.items()},
    }
    with open(output_file, "w") as f:
        json.dump(save_data, f, indent=2)
    logger.info(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--scenarios", type=int, default=100)
    p.add_argument("--max-steps", type=int, default=10)
    p.add_argument("--seed", type=int, default=123)
    a = p.parse_args()
    main(a)
