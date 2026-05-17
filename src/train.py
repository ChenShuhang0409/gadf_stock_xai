import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score
)
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from src.models import get_model
from src.utils import ensure_dir, get_device, save_json, setup_logger

logger = setup_logger()


def load_processed_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    processed_dir = Path('data/processed')
    
    X_img = np.load(processed_dir / 'X_img.npy')
    y = np.load(processed_dir / 'y.npy')
    dates = np.load(processed_dir / 'dates.npy', allow_pickle=True)
    future_returns = np.load(processed_dir / 'future_returns.npy')
    
    logger.info(f"Loaded X_img shape: {X_img.shape}")
    logger.info(f"Loaded y shape: {y.shape}")
    
    return X_img, y, dates, future_returns


def get_split_indices(dates: np.ndarray, config: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_start = pd.Timestamp(config['split']['train_start'])
    train_end = pd.Timestamp(config['split']['train_end'])
    val_start = pd.Timestamp(config['split']['val_start'])
    val_end = pd.Timestamp(config['split']['val_end'])
    test_start = pd.Timestamp(config['split']['test_start'])
    test_end = pd.Timestamp(config['split']['test_end'])
    
    dates_pd = pd.to_datetime(dates)
    
    train_idx = (dates_pd >= train_start) & (dates_pd <= train_end)
    val_idx = (dates_pd >= val_start) & (dates_pd <= val_end)
    test_idx = (dates_pd >= test_start) & (dates_pd <= test_end)
    
    if hasattr(train_idx, 'values'):
        train_idx = train_idx.values
        val_idx = val_idx.values
        test_idx = test_idx.values
    
    return train_idx, val_idx, test_idx


def compute_input_stats(X: np.ndarray, split_name: str) -> Dict:
    stats = {
        'shape': list(X.shape),
        'mean': float(np.mean(X)),
        'std': float(np.std(X)),
        'min': float(np.min(X)),
        'max': float(np.max(X)),
        'nan_count': int(np.sum(np.isnan(X))),
        'inf_count': int(np.sum(np.isinf(X))),
        'per_channel': []
    }
    
    n_channels = X.shape[1]
    for c in range(n_channels):
        channel_data = X[:, c, :, :]
        channel_stats = {
            'channel': c,
            'mean': float(np.mean(channel_data)),
            'std': float(np.std(channel_data)),
            'min': float(np.min(channel_data)),
            'max': float(np.max(channel_data))
        }
        stats['per_channel'].append(channel_stats)
    
    return stats


def print_and_save_input_stats(
    X_img: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    reports_dir: Path
) -> Dict:
    logger.info("=" * 60)
    logger.info("Input Data Statistics")
    logger.info("=" * 60)
    
    input_stats = {}
    
    for name, idx in [('train', train_idx), ('val', val_idx), ('test', test_idx)]:
        X_split = X_img[idx]
        stats = compute_input_stats(X_split, name)
        input_stats[name] = stats
        
        logger.info(f"{name.upper()} X_img stats:")
        logger.info(f"  shape: {stats['shape']}")
        logger.info(f"  mean: {stats['mean']:.6f}")
        logger.info(f"  std: {stats['std']:.6f}")
        logger.info(f"  min: {stats['min']:.6f}")
        logger.info(f"  max: {stats['max']:.6f}")
        logger.info(f"  NaN count: {stats['nan_count']}")
        logger.info(f"  Inf count: {stats['inf_count']}")
        
        for ch_stats in stats['per_channel']:
            logger.info(
                f"  Channel {ch_stats['channel']}: "
                f"mean={ch_stats['mean']:.6f}, std={ch_stats['std']:.6f}, "
                f"min={ch_stats['min']:.6f}, max={ch_stats['max']:.6f}"
            )
        
        if stats['std'] < 1e-6:
            warnings.warn(f"{name.upper()} X_img std is near zero ({stats['std']:.6f}). Data may be constant.")
        
        if stats['nan_count'] > 0:
            warnings.warn(f"{name.upper()} X_img contains {stats['nan_count']} NaN values.")
        
        if stats['inf_count'] > 0:
            warnings.warn(f"{name.upper()} X_img contains {stats['inf_count']} Inf values.")
    
    save_json(input_stats, str(reports_dir / 'input_stats.json'))
    logger.info(f"Saved input stats to {reports_dir / 'input_stats.json'}")
    logger.info("=" * 60)
    
    return input_stats


def print_label_distribution(y: np.ndarray, split_name: str) -> Dict:
    n_total = len(y)
    n_label_0 = int(np.sum(y == 0))
    n_label_1 = int(np.sum(y == 1))
    pct_0 = n_label_0 / n_total * 100
    pct_1 = n_label_1 / n_total * 100
    
    logger.info(f"{split_name} label distribution:")
    logger.info(f"  label=0: {n_label_0} ({pct_0:.2f}%)")
    logger.info(f"  label=1: {n_label_1} ({pct_1:.2f}%)")
    
    return {
        'n_total': int(n_total),
        'n_label_0': int(n_label_0),
        'n_label_1': int(n_label_1),
        'pct_label_0': float(pct_0),
        'pct_label_1': float(pct_1)
    }


def compute_class_weights(y: np.ndarray) -> torch.Tensor:
    n_total = len(y)
    n_class_0 = int(np.sum(y == 0))
    n_class_1 = int(np.sum(y == 1))
    
    weight_0 = n_total / (2 * n_class_0) if n_class_0 > 0 else 1.0
    weight_1 = n_total / (2 * n_class_1) if n_class_1 > 0 else 1.0
    
    weights = torch.tensor([weight_0, weight_1], dtype=torch.float32)
    
    return weights


def create_dataloaders(
    X_img: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    batch_size: int
) -> Tuple[DataLoader, DataLoader, np.ndarray, np.ndarray]:
    X_train = torch.tensor(X_img[train_idx], dtype=torch.float32)
    y_train_np = y[train_idx]
    y_train = torch.tensor(y_train_np, dtype=torch.long)
    
    X_val = torch.tensor(X_img[val_idx], dtype=torch.float32)
    y_val_np = y[val_idx]
    y_val = torch.tensor(y_val_np, dtype=torch.long)
    
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Val samples: {len(val_dataset)}")
    
    return train_loader, val_loader, y_train_np, y_val_np


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    num_batches = 0
    
    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        _, predicted = torch.max(outputs, 1)
        all_preds.append(predicted.cpu().numpy())
        all_labels.append(y_batch.cpu().numpy())
    
    avg_loss = total_loss / num_batches
    
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    train_acc = accuracy_score(all_labels, all_preds)
    
    return avg_loss, train_acc


def evaluate_with_metrics(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float, Dict]:
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            total_loss += loss.item()
            num_batches += 1
            
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.append(predicted.cpu().numpy())
            all_labels.append(y_batch.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
    
    avg_loss = total_loss / num_batches
    
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    all_probs = np.concatenate(all_probs)
    
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    
    try:
        if len(np.unique(all_labels)) < 2:
            auc = None
        else:
            auc = roc_auc_score(all_labels, all_probs[:, 1])
    except Exception:
        auc = None
    
    pred_0_count = int(np.sum(all_preds == 0))
    pred_1_count = int(np.sum(all_preds == 1))
    prob_up_mean = float(np.mean(all_probs[:, 1]))
    prob_up_std = float(np.std(all_probs[:, 1]))
    
    metrics = {
        'loss': avg_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'pred_0_count': pred_0_count,
        'pred_1_count': pred_1_count,
        'prob_up_mean': prob_up_mean,
        'prob_up_std': prob_up_std
    }
    
    return avg_loss, accuracy, metrics


def save_checkpoint(
    model: nn.Module,
    config: dict,
    metric_name: str,
    metric_value: float,
    epoch: int,
    save_path: Path
):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'config': config,
        'model_name': config['model'].get('name', 'simple_cnn'),
        'best_metric_name': metric_name,
        'best_metric_value': metric_value,
        'epoch': epoch
    }
    torch.save(checkpoint, save_path)
    logger.info(f"Saved checkpoint to {save_path} ({metric_name}={metric_value:.4f})")


def save_train_history(history: List[Dict], save_path: Path):
    df = pd.DataFrame(history)
    df.to_csv(save_path, index=False)
    logger.info(f"Saved training history to {save_path}")


def train_model(config: dict) -> Dict:
    logger.info("Starting training...")
    
    device = get_device(config['runtime']['device'])
    logger.info(f"Using device: {device}")
    
    reports_dir = Path('outputs/reports')
    ensure_dir(str(reports_dir))
    
    X_img, y, dates, future_returns = load_processed_data()
    
    train_idx, val_idx, test_idx = get_split_indices(dates, config)
    
    print_and_save_input_stats(X_img, train_idx, val_idx, test_idx, reports_dir)
    
    logger.info("=" * 60)
    logger.info("Label Distribution Diagnostics")
    logger.info("=" * 60)
    
    train_label_dist = print_label_distribution(y[train_idx], "Train")
    val_label_dist = print_label_distribution(y[val_idx], "Val")
    test_label_dist = print_label_distribution(y[test_idx], "Test")
    
    label_distributions = {
        'train': train_label_dist,
        'val': val_label_dist,
        'test': test_label_dist
    }
    
    save_json(label_distributions, str(reports_dir / 'label_distributions.json'))
    
    logger.info("=" * 60)
    
    batch_size = config['model']['batch_size']
    train_loader, val_loader, y_train, y_val = create_dataloaders(X_img, y, train_idx, val_idx, batch_size)
    
    model = get_model(config)
    model = model.to(device)
    logger.info(f"Model: {model.__class__.__name__}")
    
    use_class_weights = config['model'].get('use_class_weights', False)
    if use_class_weights:
        class_weights = compute_class_weights(y_train)
        class_weights = class_weights.to(device)
        logger.info(f"Using class weights: {class_weights.tolist()}")
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        logger.info("Not using class weights")
        criterion = nn.CrossEntropyLoss()
    
    learning_rate = config['model']['learning_rate']
    weight_decay = config['model']['weight_decay']
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    epochs = config['model']['epochs']
    
    models_dir = Path('outputs/models')
    ensure_dir(str(models_dir))
    
    save_metric = config['model'].get('save_metric', 'val_loss')
    logger.info(f"Save metric: {save_metric}")
    
    best_metric_value = float('inf') if 'loss' in save_metric else float('-inf')
    history = []
    
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_metrics = evaluate_with_metrics(model, val_loader, criterion, device)
        
        epoch_record = {
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'val_precision': val_metrics['precision'],
            'val_recall': val_metrics['recall'],
            'val_f1': val_metrics['f1'],
            'val_auc': val_metrics['auc'] if val_metrics['auc'] is not None else None,
            'val_pred_0_count': val_metrics['pred_0_count'],
            'val_pred_1_count': val_metrics['pred_1_count'],
            'val_prob_up_mean': val_metrics['prob_up_mean'],
            'val_prob_up_std': val_metrics['prob_up_std']
        }
        history.append(epoch_record)
        
        logger.info(
            f"Epoch {epoch}/{epochs} - "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
        )
        auc_str = f"{val_metrics['auc']:.4f}" if val_metrics['auc'] is not None else "N/A"
        logger.info(
            f"  Val Precision: {val_metrics['precision']:.4f}, "
            f"Val Recall: {val_metrics['recall']:.4f}, "
            f"Val F1: {val_metrics['f1']:.4f}, "
            f"Val AUC: {auc_str}"
        )
        logger.info(
            f"  Val Pred: 0={val_metrics['pred_0_count']}, 1={val_metrics['pred_1_count']} | "
            f"Prob Up: mean={val_metrics['prob_up_mean']:.4f}, std={val_metrics['prob_up_std']:.4f}"
        )
        
        if val_metrics['pred_0_count'] == 0 or val_metrics['pred_1_count'] == 0:
            warnings.warn("Model predicts only one class on validation set.")
        
        current_metric_value = None
        if save_metric == 'val_auc':
            if val_metrics['auc'] is not None:
                current_metric_value = val_metrics['auc']
            else:
                current_metric_value = val_loss
                actual_metric = 'val_loss (fallback)'
        elif save_metric == 'val_loss':
            current_metric_value = val_loss
            actual_metric = 'val_loss'
        elif save_metric == 'val_acc':
            current_metric_value = val_acc
            actual_metric = 'val_acc'
        elif save_metric == 'val_f1':
            current_metric_value = val_metrics['f1']
            actual_metric = 'val_f1'
        else:
            current_metric_value = val_loss
            actual_metric = 'val_loss'
        
        if 'loss' in save_metric:
            is_better = current_metric_value < best_metric_value
        else:
            is_better = current_metric_value > best_metric_value
        
        if is_better:
            best_metric_value = current_metric_value
            save_checkpoint(
                model, config, save_metric, best_metric_value, epoch,
                models_dir / 'best_model.pt'
            )
    
    save_train_history(history, reports_dir / 'train_history.csv')
    
    logger.info(f"Training completed. Best {save_metric}: {best_metric_value:.4f}")
    
    return {
        'best_metric_value': best_metric_value,
        'save_metric': save_metric,
        'final_epoch': epochs,
        'history': history,
        'label_distributions': label_distributions
    }
