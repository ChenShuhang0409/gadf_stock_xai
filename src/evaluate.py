import warnings
from pathlib import Path
from typing import Dict, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score
)
from torch.utils.data import DataLoader, TensorDataset

from src.models import get_model
from src.utils import ensure_dir, get_device, save_json, setup_logger

logger = setup_logger()


def load_model(checkpoint_path: Path, device: torch.device) -> Tuple[torch.nn.Module, dict]:
    logger.info(f"Loading trusted local checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint['config']
    
    model = get_model(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    model_name = checkpoint.get('model_name', 'unknown')
    best_metric_name = checkpoint.get('best_metric_name', 'unknown')
    best_metric_value = checkpoint.get('best_metric_value', None)
    epoch = checkpoint.get('epoch', None)
    
    logger.info(f"Loaded model: {model_name}")
    logger.info(f"Model was trained for {epoch} epochs")
    if best_metric_value is not None:
        logger.info(f"Best metric: {best_metric_name}={float(best_metric_value):.4f}")
    
    return model, config


def load_data_by_split(config: dict, split: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    processed_dir = Path('data/processed')
    
    X_img = np.load(processed_dir / 'X_img.npy')
    y = np.load(processed_dir / 'y.npy')
    dates = np.load(processed_dir / 'dates.npy', allow_pickle=True)
    future_returns = np.load(processed_dir / 'future_returns.npy')
    
    split_start = pd.Timestamp(config['split'][f'{split}_start'])
    split_end = pd.Timestamp(config['split'][f'{split}_end'])
    
    dates_pd = pd.to_datetime(dates)
    split_idx = (dates_pd >= split_start) & (dates_pd <= split_end)
    
    if hasattr(split_idx, 'values'):
        split_idx = split_idx.values
    
    X_split = X_img[split_idx]
    y_split = y[split_idx]
    dates_split = dates[split_idx]
    future_returns_split = future_returns[split_idx]
    
    return X_split, y_split, dates_split, future_returns_split


def load_test_data(config: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X_test, y_test, dates_test, future_returns_test = load_data_by_split(config, 'test')
    logger.info(f"Test samples: {len(X_test)}")
    return X_test, y_test, dates_test, future_returns_test


def predict(
    model: torch.nn.Module,
    X: np.ndarray,
    batch_size: int,
    device: torch.device
) -> Tuple[np.ndarray, np.ndarray]:
    X_tensor = torch.tensor(X, dtype=torch.float32)
    dataset = TensorDataset(X_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    all_probs = []
    all_preds = []
    
    with torch.no_grad():
        for (X_batch,) in loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)
            
            all_probs.append(probs.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
    
    probs = np.concatenate(all_probs, axis=0)
    preds = np.concatenate(all_preds, axis=0)
    
    return probs, preds


def find_best_threshold(
    y_true: np.ndarray,
    probs: np.ndarray,
    thresholds: np.ndarray = None
) -> Tuple[float, Dict]:
    if thresholds is None:
        thresholds = np.arange(0.3, 0.71, 0.01)
    
    best_threshold = 0.5
    best_balanced_acc = 0.0
    best_metrics = {}
    
    for thresh in thresholds:
        y_pred = (probs[:, 1] >= thresh).astype(int)
        
        acc = accuracy_score(y_true, y_pred)
        bal_acc = balanced_accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        
        if bal_acc > best_balanced_acc:
            best_balanced_acc = bal_acc
            best_threshold = thresh
            best_metrics = {
                'accuracy': acc,
                'balanced_accuracy': bal_acc,
                'f1': f1,
                'precision': prec,
                'recall': rec
            }
    
    return best_threshold, best_metrics


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict:
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
    }
    
    try:
        if len(np.unique(y_true)) < 2:
            warnings.warn("Only one class present in y_true. ROC AUC is not defined.")
            metrics['roc_auc'] = None
        else:
            metrics['roc_auc'] = roc_auc_score(y_true, y_prob[:, 1])
    except ValueError as e:
        warnings.warn(f"ROC AUC calculation failed: {e}")
        metrics['roc_auc'] = None
    
    cm = confusion_matrix(y_true, y_pred)
    metrics['confusion_matrix'] = cm.tolist()
    
    return metrics


def compute_baseline_metrics(y_true: np.ndarray) -> Dict:
    n_total = len(y_true)
    n_label_0 = int(np.sum(y_true == 0))
    n_label_1 = int(np.sum(y_true == 1))
    
    always_0_acc = n_label_0 / n_total
    always_1_acc = n_label_1 / n_total
    
    if n_label_0 >= n_label_1:
        majority_acc = always_0_acc
        majority_class = 0
    else:
        majority_acc = always_1_acc
        majority_class = 1
    
    baseline = {
        'always_predict_0_accuracy': always_0_acc,
        'always_predict_1_accuracy': always_1_acc,
        'majority_class': majority_class,
        'majority_class_accuracy': majority_acc,
        'test_label_distribution': {
            'n_total': n_total,
            'n_label_0': n_label_0,
            'n_label_1': n_label_1,
            'pct_label_0': float(n_label_0 / n_total * 100),
            'pct_label_1': float(n_label_1 / n_total * 100)
        }
    }
    
    return baseline


def compute_diagnostics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probs: np.ndarray,
    model_accuracy: float = None,
    majority_baseline_accuracy: float = None
) -> Dict:
    pred_0_count = int(np.sum(y_pred == 0))
    pred_1_count = int(np.sum(y_pred == 1))
    prob_up_mean = float(np.mean(probs[:, 1]))
    prob_up_std = float(np.std(probs[:, 1]))
    
    n_total = len(y_true)
    n_label_0 = int(np.sum(y_true == 0))
    n_label_1 = int(np.sum(y_true == 1))
    
    cm = confusion_matrix(y_true, y_pred)
    
    unique_pred_classes = sorted(list(set(y_pred.tolist())))
    collapsed = len(unique_pred_classes) == 1
    
    try:
        if len(np.unique(y_true)) < 2:
            roc_auc = None
        else:
            roc_auc = roc_auc_score(y_true, probs[:, 1])
    except Exception:
        roc_auc = None
    
    beats_majority_baseline = False
    auc_above_threshold = False
    ready_for_lrp = False
    
    if model_accuracy is not None and majority_baseline_accuracy is not None:
        beats_majority_baseline = model_accuracy > majority_baseline_accuracy
    
    if roc_auc is not None:
        auc_above_threshold = roc_auc > 0.52
    
    if not collapsed and beats_majority_baseline and auc_above_threshold:
        ready_for_lrp = True
        recommendation = "Model predicts both classes, beats baseline, and has AUC > 0.52. LRP and clustering can be considered."
    else:
        recommendation = "Do not run LRP or clustering. Model has not beaten baseline or AUC is too weak."
    
    diagnostics = {
        'collapsed_prediction': collapsed,
        'unique_pred_classes': unique_pred_classes,
        'recommendation': recommendation,
        'beats_majority_baseline': beats_majority_baseline,
        'auc_above_threshold': auc_above_threshold,
        'ready_for_lrp': ready_for_lrp,
        'test_pred_0_count': pred_0_count,
        'test_pred_1_count': pred_1_count,
        'test_prob_up_mean': prob_up_mean,
        'test_prob_up_std': prob_up_std,
        'test_label_distribution': {
            'n_total': n_total,
            'n_label_0': n_label_0,
            'n_label_1': n_label_1,
            'pct_label_0': float(n_label_0 / n_total * 100),
            'pct_label_1': float(n_label_1 / n_total * 100)
        },
        'confusion_matrix': {
            'true_0_pred_0': int(cm[0, 0]) if cm.shape[0] > 1 and cm.shape[1] > 0 else 0,
            'true_0_pred_1': int(cm[0, 1]) if cm.shape[0] > 1 and cm.shape[1] > 1 else 0,
            'true_1_pred_0': int(cm[1, 0]) if cm.shape[0] > 1 and cm.shape[1] > 1 else 0,
            'true_1_pred_1': int(cm[1, 1]) if cm.shape[0] > 1 and cm.shape[1] > 1 else 0
        }
    }
    
    return diagnostics


def save_predictions(
    dates: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probs: np.ndarray,
    future_returns: np.ndarray,
    save_path: Path
):
    df = pd.DataFrame({
        'date': dates,
        'y_true': y_true,
        'y_pred': y_pred,
        'prob_down': probs[:, 0],
        'prob_up': probs[:, 1],
        'future_return': future_returns
    })
    
    df.to_csv(save_path, index=False)
    logger.info(f"Saved predictions to {save_path}")


def plot_confusion_matrix(cm: np.ndarray, save_path: Path):
    fig, ax = plt.subplots(figsize=(8, 6))
    
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    classes = ['Down (0)', 'Up (1)']
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes,
           yticklabels=classes,
           ylabel='True label',
           xlabel='Predicted label',
           title='Confusion Matrix')
    
    ax.set_ylim(len(classes) - 0.5, -0.5)
    
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    fig.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved confusion matrix to {save_path}")


def evaluate_model(config: dict) -> Dict:
    logger.info("Starting evaluation...")
    
    device = get_device(config['runtime']['device'])
    logger.info(f"Using device: {device}")
    
    models_dir = Path('outputs/models')
    checkpoint_path = models_dir / 'best_model.pt'
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")
    
    model, model_config = load_model(checkpoint_path, device)
    
    batch_size = config['model']['batch_size']
    
    X_val, y_val, _, _ = load_data_by_split(config, 'val')
    logger.info(f"Val samples for threshold optimization: {len(X_val)}")
    
    val_probs, _ = predict(model, X_val, batch_size, device)
    
    best_threshold, val_metrics = find_best_threshold(y_val, val_probs)
    logger.info(f"Best decision threshold from validation: {best_threshold:.2f}")
    logger.info(f"Val balanced accuracy with best threshold: {val_metrics['balanced_accuracy']:.4f}")
    
    X_test, y_test, dates_test, future_returns_test = load_test_data(config)
    
    probs, preds_default = predict(model, X_test, batch_size, device)
    
    preds_with_threshold = (probs[:, 1] >= best_threshold).astype(int)
    
    metrics = compute_metrics(y_test, preds_with_threshold, probs)
    metrics['decision_threshold'] = float(best_threshold)
    
    baseline = compute_baseline_metrics(y_test)
    
    diagnostics = compute_diagnostics(
        y_test, preds_with_threshold, probs,
        model_accuracy=metrics['accuracy'],
        majority_baseline_accuracy=baseline['majority_class_accuracy']
    )
    
    logger.info("=" * 60)
    logger.info("Test Set Results")
    logger.info("=" * 60)
    
    logger.info(f"Decision threshold: {best_threshold:.2f}")
    logger.info(f"Test label distribution:")
    label_dist = diagnostics['test_label_distribution']
    logger.info(f"  label=0: {label_dist['n_label_0']} ({label_dist['pct_label_0']:.2f}%)")
    logger.info(f"  label=1: {label_dist['n_label_1']} ({label_dist['pct_label_1']:.2f}%)")
    
    logger.info(f"Test prediction counts:")
    logger.info(f"  pred_0_count: {diagnostics['test_pred_0_count']}")
    logger.info(f"  pred_1_count: {diagnostics['test_pred_1_count']}")
    
    logger.info(f"Test prob_up statistics:")
    logger.info(f"  mean: {diagnostics['test_prob_up_mean']:.4f}")
    logger.info(f"  std: {diagnostics['test_prob_up_std']:.4f}")
    
    cm_dict = diagnostics['confusion_matrix']
    logger.info(f"Confusion Matrix:")
    logger.info(f"  True 0, Pred 0: {cm_dict['true_0_pred_0']}")
    logger.info(f"  True 0, Pred 1: {cm_dict['true_0_pred_1']}")
    logger.info(f"  True 1, Pred 0: {cm_dict['true_1_pred_0']}")
    logger.info(f"  True 1, Pred 1: {cm_dict['true_1_pred_1']}")
    
    logger.info("=" * 60)
    logger.info("Baseline Comparison")
    logger.info("=" * 60)
    
    logger.info(f"Model accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"Balanced accuracy: {metrics['balanced_accuracy']:.4f}")
    logger.info(f"Always-0 accuracy: {baseline['always_predict_0_accuracy']:.4f}")
    logger.info(f"Always-1 accuracy: {baseline['always_predict_1_accuracy']:.4f}")
    logger.info(f"Majority baseline accuracy (class {baseline['majority_class']}): {baseline['majority_class_accuracy']:.4f}")
    
    if metrics['roc_auc'] is not None:
        logger.info(f"Model ROC AUC: {metrics['roc_auc']:.4f}")
    else:
        logger.info("Model ROC AUC: N/A")
    
    if metrics['accuracy'] <= baseline['majority_class_accuracy']:
        warnings.warn("Model does not outperform majority baseline.")
    
    logger.info("=" * 60)
    
    if diagnostics['collapsed_prediction']:
        warnings.warn("Model predicts only one class. Results are not meaningful for LRP or clustering.")
    
    logger.info(f"Test Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"Test Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
    logger.info(f"Test Precision: {metrics['precision']:.4f}")
    logger.info(f"Test Recall: {metrics['recall']:.4f}")
    logger.info(f"Test F1: {metrics['f1']:.4f}")
    if metrics['roc_auc'] is not None:
        logger.info(f"Test ROC AUC: {metrics['roc_auc']:.4f}")
    else:
        logger.warning("ROC AUC: Not available (single class in test set)")
    
    reports_dir = Path('outputs/reports')
    ensure_dir(str(reports_dir))
    
    decision_threshold_info = {
        'best_decision_threshold': float(best_threshold),
        'selection_metric': 'val_balanced_accuracy',
        'val_balanced_accuracy': float(val_metrics['balanced_accuracy']),
        'test_accuracy_with_threshold': float(metrics['accuracy']),
        'test_balanced_accuracy_with_threshold': float(metrics['balanced_accuracy'])
    }
    save_json(decision_threshold_info, str(reports_dir / 'decision_threshold.json'))
    logger.info(f"Saved decision threshold info to {reports_dir / 'decision_threshold.json'}")
    
    metrics_with_diagnostics = {**metrics, **diagnostics}
    save_json(metrics_with_diagnostics, str(reports_dir / 'metrics.json'))
    logger.info(f"Saved metrics to {reports_dir / 'metrics.json'}")
    
    save_json(baseline, str(reports_dir / 'baseline_metrics.json'))
    logger.info(f"Saved baseline metrics to {reports_dir / 'baseline_metrics.json'}")
    
    save_json(diagnostics, str(reports_dir / 'prediction_diagnostics.json'))
    logger.info(f"Saved prediction diagnostics to {reports_dir / 'prediction_diagnostics.json'}")
    
    save_predictions(
        dates_test, y_test, preds_with_threshold, probs, future_returns_test,
        reports_dir / 'predictions.csv'
    )
    
    cm = np.array(metrics['confusion_matrix'])
    plot_confusion_matrix(cm, reports_dir / 'confusion_matrix.png')
    
    logger.info("Evaluation completed.")
    
    return metrics
