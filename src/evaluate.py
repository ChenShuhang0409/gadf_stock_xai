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
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score
)
from torch.utils.data import DataLoader, TensorDataset

from src.models import SimpleCNN
from src.utils import ensure_dir, get_device, save_json, setup_logger

logger = setup_logger()


def load_model(checkpoint_path: Path, device: torch.device) -> Tuple[SimpleCNN, dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']
    
    in_channels = config['model']['in_channels']
    num_classes = config['model']['num_classes']
    
    model = SimpleCNN(in_channels=in_channels, num_classes=num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    logger.info(f"Loaded model from {checkpoint_path}")
    logger.info(f"Model was trained for {checkpoint['epoch']} epochs")
    logger.info(f"Best val accuracy: {checkpoint['best_val_accuracy']:.4f}")
    
    return model, config


def load_test_data(config: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    processed_dir = Path('data/processed')
    
    X_img = np.load(processed_dir / 'X_img.npy')
    y = np.load(processed_dir / 'y.npy')
    dates = np.load(processed_dir / 'dates.npy', allow_pickle=True)
    future_returns = np.load(processed_dir / 'future_returns.npy')
    
    train_start = pd.Timestamp(config['split']['train_start'])
    train_end = pd.Timestamp(config['split']['train_end'])
    val_start = pd.Timestamp(config['split']['val_start'])
    val_end = pd.Timestamp(config['split']['val_end'])
    test_start = pd.Timestamp(config['split']['test_start'])
    test_end = pd.Timestamp(config['split']['test_end'])
    
    dates_pd = pd.to_datetime(dates)
    test_idx = (dates_pd >= test_start) & (dates_pd <= test_end)
    
    if hasattr(test_idx, 'values'):
        test_idx = test_idx.values
    
    X_test = X_img[test_idx]
    y_test = y[test_idx]
    dates_test = dates[test_idx]
    future_returns_test = future_returns[test_idx]
    
    logger.info(f"Test samples: {len(X_test)}")
    
    return X_test, y_test, dates_test, future_returns_test


def predict(
    model: SimpleCNN,
    X_test: np.ndarray,
    batch_size: int,
    device: torch.device
) -> Tuple[np.ndarray, np.ndarray]:
    X_tensor = torch.tensor(X_test, dtype=torch.float32)
    test_dataset = TensorDataset(X_tensor)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    all_probs = []
    all_preds = []
    
    with torch.no_grad():
        for (X_batch,) in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)
            
            all_probs.append(probs.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
    
    probs = np.concatenate(all_probs, axis=0)
    preds = np.concatenate(all_preds, axis=0)
    
    return probs, preds


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict:
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
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
    
    X_test, y_test, dates_test, future_returns_test = load_test_data(config)
    
    batch_size = config['model']['batch_size']
    probs, preds = predict(model, X_test, batch_size, device)
    
    metrics = compute_metrics(y_test, preds, probs)
    
    logger.info(f"Test Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"Test Precision: {metrics['precision']:.4f}")
    logger.info(f"Test Recall: {metrics['recall']:.4f}")
    logger.info(f"Test F1: {metrics['f1']:.4f}")
    if metrics['roc_auc'] is not None:
        logger.info(f"Test ROC AUC: {metrics['roc_auc']:.4f}")
    else:
        logger.warning("ROC AUC: Not available (single class in test set)")
    
    reports_dir = Path('outputs/reports')
    ensure_dir(str(reports_dir))
    
    save_json(metrics, str(reports_dir / 'metrics.json'))
    logger.info(f"Saved metrics to {reports_dir / 'metrics.json'}")
    
    save_predictions(
        dates_test, y_test, preds, probs, future_returns_test,
        reports_dir / 'predictions.csv'
    )
    
    cm = np.array(metrics['confusion_matrix'])
    plot_confusion_matrix(cm, reports_dir / 'confusion_matrix.png')
    
    logger.info("Evaluation completed.")
    
    return metrics
