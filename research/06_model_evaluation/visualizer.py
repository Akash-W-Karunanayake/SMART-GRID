"""
Visualizer — Confusion matrices, ROC curves, and training loss curves.
IT22577924 — Karunanayake K.P.A.W.
"""
import json
import logging
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

logger = logging.getLogger(__name__)

PLOTS_DIR = Path(__file__).resolve().parent.parent / "results" / "plots"


def plot_confusion_matrix(cm: np.ndarray, class_names: list,
                          title: str, out_path: Path) -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(max(6, len(class_names)), max(5, len(class_names))))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info(f"Confusion matrix saved → {out_path}")


def plot_loss_curves(history_path: Path, model_name: str, out_path: Path) -> None:
    """Plot training and validation loss from a JSON history file."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    if not history_path.exists():
        logger.warning(f"History not found: {history_path}")
        return

    with open(history_path) as f:
        history = json.load(f)

    epochs     = [h["epoch"]      for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss   = [h["val_loss"]   for h in history]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_loss, label="Train Loss", linewidth=2)
    ax.plot(epochs, val_loss,   label="Val Loss",   linewidth=2, linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"{model_name} — Training Curves")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info(f"Loss curve saved → {out_path}")


def plot_accuracy_comparison(results_table: list, task: str, out_path: Path) -> None:
    """
    Bar chart comparing model accuracies on a given task.

    results_table : list of dicts with keys 'model', 'accuracy'
    """
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    models = [r["model"] for r in results_table]
    accs   = [r.get(f"{task}_accuracy", r.get("accuracy", 0)) for r in results_table]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(models, accs, color=plt.cm.viridis(np.linspace(0.2, 0.8, len(models))))
    ax.set_xlabel("Accuracy")
    ax.set_title(f"Model Comparison — {task.capitalize()} Task")
    ax.set_xlim(0, 1.05)
    for bar, val in zip(bars, accs):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f"{val:.3f}", va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info(f"Comparison plot saved → {out_path}")
