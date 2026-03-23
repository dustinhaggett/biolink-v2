"""
Evaluation metrics and visualization utilities.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_recall_curve,
    roc_curve
)
import torch
from torch.utils.data import TensorDataset, DataLoader


def precision_at_k(y_true: np.ndarray, y_scores: np.ndarray, ks=(10, 50, 100, 200)) -> dict:
    """
    Compute precision@k for a set of ks.
    
    Args:
        y_true: True binary labels
        y_scores: Predicted scores (higher = more positive)
        ks: List of k values
        
    Returns:
        Dictionary mapping k to precision@k
    """
    order = np.argsort(-y_scores)
    y_sorted = y_true[order]
    
    results = {}
    for k in ks:
        if k > len(y_sorted):
            results[k] = np.nan
            continue
        topk = y_sorted[:k]
        results[k] = float(topk.mean())
    return results


def evaluate_model(
    model: torch.nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    split_name: str = "Test",
    batch_size: int = 256,
    device: str = None
):
    """
    Evaluate model and return predictions.
    
    Args:
        model: Trained model
        X: Features
        y: Labels
        split_name: Name of the split (for printing)
        batch_size: Batch size
        device: Device to use (auto-detect if None)
        
    Returns:
        Tuple of (y_true, y_prob)
    """
    if device is None:
        device = torch.device(
            "mps" if torch.backends.mps.is_available()
            else "cuda" if torch.cuda.is_available()
            else "cpu"
        )
    
    model.eval()
    ds = TensorDataset(
        torch.from_numpy(X).float(),
        torch.from_numpy(y).float()
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    
    all_logits = []
    all_labels = []
    
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            logits = model(xb)
            all_logits.append(logits.cpu().numpy())
            all_labels.append(yb.numpy())
    
    y_true = np.concatenate(all_labels)
    y_logit = np.concatenate(all_logits)
    y_prob = 1.0 / (1.0 + np.exp(-y_logit))
    
    acc = accuracy_score(y_true, (y_prob >= 0.5).astype(int))
    auc = roc_auc_score(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    pks = precision_at_k(y_true, y_prob)
    
    print(f"\n=== {split_name} performance ===")
    print(f"Accuracy: {acc:.4f}")
    print(f"AUC:      {auc:.4f}")
    print(f"AP:       {ap:.4f}")
    for k, v in pks.items():
        if np.isnan(v):
            print(f"P@{k}:   N/A")
        else:
            print(f"P@{k}:   {v:.4f}")
    
    return y_true, y_prob


def plot_roc_curve(y_true: np.ndarray, y_prob: np.ndarray, label: str, title: str = "ROC Curve"):
    """
    Plot ROC curve.
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        label: Label for the curve
        title: Plot title
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"{label} (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_pr_curve(y_true: np.ndarray, y_prob: np.ndarray, label: str, title: str = "Precision–Recall Curve"):
    """
    Plot Precision-Recall curve.
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        label: Label for the curve
        title: Plot title
    """
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    
    plt.figure(figsize=(6, 5))
    plt.plot(rec, prec, label=f"{label} (AP={ap:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def summarize_results(name: str, y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    """
    Summarize model results in a dictionary.
    
    Args:
        name: Model name
        y_true: True labels
        y_prob: Predicted probabilities
        
    Returns:
        Dictionary with metrics
    """
    acc = accuracy_score(y_true, (y_prob >= 0.5).astype(int))
    auc = roc_auc_score(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    pks = precision_at_k(y_true, y_prob)
    
    return {
        "Model": name,
        "Accuracy": acc,
        "AUC": auc,
        "AP": ap,
        "P@10": pks.get(10, np.nan),
        "P@50": pks.get(50, np.nan),
        "P@100": pks.get(100, np.nan),
        "P@200": pks.get(200, np.nan),
    }

