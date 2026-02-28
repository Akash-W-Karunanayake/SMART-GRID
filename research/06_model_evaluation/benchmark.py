"""
Benchmark — Compare all models on the test set.
Runs: SVM baseline, CNN-Transformer (v2 DenseNet), R-GNN (v2), Hybrid BHAF (v2).

Usage:
    python research/06_model_evaluation/benchmark.py

IT22577924 — Karunanayake K.P.A.W.
"""
import sys
import pickle
import logging
import warnings
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore")

RESEARCH_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(RESEARCH_DIR / "03_cnn_transformer"))
sys.path.insert(0, str(RESEARCH_DIR / "04_r_gnn"))
sys.path.insert(0, str(RESEARCH_DIR / "05_hybrid_model"))

from metrics import compute_all_metrics, FAULT_TYPE_NAMES

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("benchmark")

TEST_NPZ  = RESEARCH_DIR / "datasets" / "processed" / "test" / "test.npz"
TEST_PKL  = RESEARCH_DIR / "datasets" / "processed" / "test" / "test_pyg.pkl"
REPORTS_DIR = RESEARCH_DIR / "results" / "reports"
PLOTS_DIR   = RESEARCH_DIR / "results" / "plots"


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_test_npz():
    d = np.load(TEST_NPZ, allow_pickle=True)
    I = d["current_seq"]   # [N, T, N_branches, 6]
    N, T, B, F = I.shape
    X = I.reshape(N, T, B * F)
    return {
        "X":     X,
        "y_det": d["label_detection"].astype(int),
        "y_type": d["label_type"].astype(int),
        "y_ph":  d["label_phase"].astype(int),
        "y_loc": np.clip(d["label_location"].astype(int), 0, None),
    }


def load_test_pkl():
    if not TEST_PKL.exists():
        return None
    with open(TEST_PKL, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# Baseline: SVM on hand-crafted features
# ---------------------------------------------------------------------------

def run_svm_baseline(data: dict) -> dict:
    """SVM on RMS-style aggregated features."""
    logger.info("Running SVM baseline...")
    try:
        from sklearn.svm import SVC
        from sklearn.preprocessing import StandardScaler

        X = data["X"]
        X_feat = np.sqrt((X ** 2).mean(axis=1))
        sc     = StandardScaler()
        X_s    = sc.fit_transform(X_feat)

        y_type = data["y_type"]
        split  = int(0.8 * len(X_s))
        X_tr, X_te = X_s[:split], X_s[split:]
        y_tr, y_te = y_type[:split], y_type[split:]

        svm = SVC(kernel="rbf", C=10, gamma="scale", random_state=42)
        svm.fit(X_tr, y_tr)
        y_pred = svm.predict(X_te)

        m = compute_all_metrics(y_te, y_pred, "type")
        return {
            "model": "SVM (baseline)",
            "type_accuracy": m["accuracy"],
            "type_f1_macro": m["f1_macro"],
            "type_mcc":      m["mcc"],
        }
    except Exception as e:
        logger.warning(f"SVM failed: {e}")
        return {"model": "SVM (baseline)", "type_accuracy": 0, "type_f1_macro": 0, "type_mcc": 0}


# ---------------------------------------------------------------------------
# Model evaluation helpers
# ---------------------------------------------------------------------------

def _eval_all_tasks(preds, trues):
    """Compute metrics for all tasks from prediction dicts."""
    result = {}
    for task, ytrue_key in [("detection","y_det"), ("type","y_type"),
                              ("phase","y_ph"), ("location","y_loc")]:
        if task in preds and ytrue_key in trues:
            m = compute_all_metrics(
                np.array(trues[ytrue_key]), np.array(preds[task]), task)
            result[f"{task}_accuracy"] = m["accuracy"]
            result[f"{task}_f1_macro"] = m["f1_macro"]
            result[f"{task}_mcc"]      = m["mcc"]
    return result


def run_cnn_transformer(data: dict) -> dict:
    logger.info("Evaluating CNN-Transformer (v2 DenseNet)...")
    ckpt_path = RESEARCH_DIR / "models" / "cnn_transformer" / "best_model.pt"
    if not ckpt_path.exists():
        logger.warning("CNN-T checkpoint not found — skipping")
        return {"model": "CNN-Transformer-v2"}

    try:
        import importlib.util
        _spec = importlib.util.spec_from_file_location(
            "cnn_t_model", RESEARCH_DIR / "03_cnn_transformer" / "model.py")
        _mod = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_mod)
        CNNTransformerModel = _mod.CNNTransformerModel
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model = CNNTransformerModel(ckpt["in_features"], ckpt["n_buses"])
        model.load_state_dict(ckpt["model_state"])
        model.eval()

        X = torch.tensor(data["X"], dtype=torch.float32)
        preds = {k: [] for k in ["detection","type","phase","location"]}
        with torch.no_grad():
            for i in range(0, len(X), 64):
                out = model(X[i:i+64])
                for k in preds:
                    preds[k].extend(out[k].argmax(1).numpy().tolist())

        result = {"model": "CNN-Transformer-v2"}
        result.update(_eval_all_tasks(preds, {
            "y_det": data["y_det"], "y_type": data["y_type"],
            "y_ph": data["y_ph"], "y_loc": data["y_loc"]}))
        return result
    except Exception as e:
        logger.warning(f"CNN-T eval failed: {e}")
        return {"model": "CNN-Transformer-v2"}


def run_r_gnn(pyg_list: list) -> dict:
    logger.info("Evaluating R-GNN (v2)...")
    ckpt_path = RESEARCH_DIR / "models" / "r_gnn" / "best_model.pt"
    if not ckpt_path.exists() or pyg_list is None:
        logger.warning("R-GNN checkpoint or data not found — skipping")
        return {"model": "R-GNN-v2"}

    try:
        import importlib.util
        _spec = importlib.util.spec_from_file_location(
            "rgnn_model", RESEARCH_DIR / "04_r_gnn" / "model.py")
        _mod = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_mod)
        RGNNModel = _mod.RGNNModel
        ckpt   = torch.load(ckpt_path, map_location="cpu")
        model  = RGNNModel(n_buses=ckpt["n_buses"])
        model.load_state_dict(ckpt["model_state"])
        model.eval()

        ei = pyg_list[0].edge_index
        ea = pyg_list[0].edge_attr

        preds = {k: [] for k in ["detection","type","phase","location"]}
        trues = {"y_det": [], "y_type": [], "y_ph": [], "y_loc": []}
        with torch.no_grad():
            for d in pyg_list:
                x = d.x.unsqueeze(0)
                out = model(x, ei, ea)
                for k in preds:
                    preds[k].append(out[k].argmax(1).item())
                trues["y_det"].append(int(d.y_detection))
                trues["y_type"].append(int(d.y_type))
                trues["y_ph"].append(int(d.y_phase))
                trues["y_loc"].append(max(0, int(d.y_location)))

        result = {"model": "R-GNN-v2"}
        result.update(_eval_all_tasks(preds, trues))
        return result
    except Exception as e:
        logger.warning(f"R-GNN eval failed: {e}")
        return {"model": "R-GNN-v2"}


def run_hybrid(data: dict, pyg_list: list) -> dict:
    logger.info("Evaluating Hybrid BHAF Model (v2)...")
    ckpt_path = RESEARCH_DIR / "models" / "hybrid" / "best_model.pt"
    if not ckpt_path.exists() or pyg_list is None:
        logger.warning("Hybrid checkpoint or data not found — skipping")
        return {"model": "Hybrid-BHAF-v2"}

    try:
        import importlib.util
        sys.path.insert(0, str(RESEARCH_DIR / "05_hybrid_model"))
        _spec = importlib.util.spec_from_file_location(
            "hybrid_model", RESEARCH_DIR / "05_hybrid_model" / "model.py")
        _mod = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_mod)
        HybridModel = _mod.HybridModel
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model = HybridModel(in_features_cnn=ckpt["in_features_cnn"], n_buses=ckpt["n_buses"])
        model.load_state_dict(ckpt["model_state"])
        model.eval()

        ei = pyg_list[0].edge_index
        ea = pyg_list[0].edge_attr

        preds = {k: [] for k in ["detection","type","phase","location"]}
        trues = {"y_det": [], "y_type": [], "y_ph": [], "y_loc": []}
        X = torch.tensor(data["X"], dtype=torch.float32)

        with torch.no_grad():
            for i, d in enumerate(pyg_list):
                x_cur = X[i].unsqueeze(0)
                x_vol = d.x.unsqueeze(0)
                out = model(x_cur, x_vol, ei, ea)
                for k in preds:
                    preds[k].append(out[k].argmax(1).item())
                trues["y_det"].append(int(d.y_detection))
                trues["y_type"].append(int(d.y_type))
                trues["y_ph"].append(int(d.y_phase))
                trues["y_loc"].append(max(0, int(d.y_location)))

        result = {"model": "Hybrid-BHAF-v2"}
        result.update(_eval_all_tasks(preds, trues))
        return result
    except Exception as e:
        logger.warning(f"Hybrid eval failed: {e}")
        return {"model": "Hybrid-BHAF-v2"}


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def save_report(results: list) -> Path:
    import csv
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORTS_DIR / "benchmark_comparison.csv"

    all_keys = set()
    for r in results:
        all_keys.update(r.keys())
    all_keys.discard("model")
    fieldnames = ["model"] + sorted(all_keys)

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({k: f"{r.get(k, 0):.4f}" if isinstance(r.get(k), float) else r.get(k, "")
                             for k in fieldnames})
    logger.info(f"Benchmark report → {out_path}")
    return out_path


def print_summary_table(results: list) -> None:
    logger.info("\n" + "="*90)
    logger.info(f"{'Model':<25} {'Det Acc':>8} {'Type Acc':>9} {'Type F1':>8} "
                f"{'Phase Acc':>9} {'Loc Acc':>8}")
    logger.info("-"*90)
    for r in results:
        logger.info(f"{r['model']:<25} "
                    f"{r.get('detection_accuracy', 0):>8.4f} "
                    f"{r.get('type_accuracy', 0):>9.4f} "
                    f"{r.get('type_f1_macro', 0):>8.4f} "
                    f"{r.get('phase_accuracy', 0):>9.4f} "
                    f"{r.get('location_accuracy', 0):>8.4f}")
    logger.info("="*90)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    logger.info("=== Stage 06: Benchmark Evaluation ===")

    if not TEST_NPZ.exists():
        logger.error(f"Test data not found: {TEST_NPZ}\nRun Stage 02 first.")
        sys.exit(1)

    data     = load_test_npz()
    pyg_list = load_test_pkl()

    results = []

    # 1. SVM baseline
    results.append(run_svm_baseline(data))

    # 2. CNN-Transformer v2
    results.append(run_cnn_transformer(data))

    # 3. R-GNN v2
    results.append(run_r_gnn(pyg_list))

    # 4. Hybrid BHAF v2
    results.append(run_hybrid(data, pyg_list))

    # Summary table
    print_summary_table(results)

    # Save CSV
    report_path = save_report(results)

    # Plots
    try:
        from visualizer import plot_accuracy_comparison
        for task in ["type", "detection", "phase", "location"]:
            plot_accuracy_comparison(
                [{"model": r["model"], f"{task}_accuracy": r.get(f"{task}_accuracy", 0)}
                 for r in results],
                task,
                PLOTS_DIR / f"comparison_{task}.png"
            )
    except Exception as e:
        logger.warning(f"Plotting failed: {e}")

    # Loss curves
    try:
        from visualizer import plot_loss_curves
        for model_name, subdir in [
            ("CNN-Transformer-v2", "cnn_transformer"),
            ("R-GNN-v2",           "r_gnn"),
            ("Hybrid-BHAF-v2",     "hybrid"),
        ]:
            hist_path = RESEARCH_DIR / "models" / subdir / "history.json"
            plot_loss_curves(hist_path, model_name,
                             PLOTS_DIR / f"loss_curve_{subdir}.png")
    except Exception as e:
        logger.warning(f"Loss curve plotting failed: {e}")

    logger.info(f"\nDone. Report: {report_path}")


if __name__ == "__main__":
    main()
