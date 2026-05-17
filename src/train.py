from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from src.models import SimpleCNN
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


def create_dataloaders(
    X_img: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    batch_size: int
) -> Tuple[DataLoader, DataLoader]:
    X_train = torch.tensor(X_img[train_idx], dtype=torch.float32)
    y_train = torch.tensor(y[train_idx], dtype=torch.long)
    
    X_val = torch.tensor(X_img[val_idx], dtype=torch.float32)
    y_val = torch.tensor(y[val_idx], dtype=torch.long)
    
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Val samples: {len(val_dataset)}")
    
    return train_loader, val_loader


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> float:
    model.train()
    total_loss = 0.0
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
    
    return total_loss / num_batches


def evaluate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    num_batches = 0
    
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            total_loss += loss.item()
            num_batches += 1
            
            _, predicted = torch.max(outputs, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
    
    avg_loss = total_loss / num_batches
    accuracy = correct / total
    
    return avg_loss, accuracy


def save_checkpoint(
    model: nn.Module,
    config: dict,
    best_val_accuracy: float,
    epoch: int,
    save_path: Path
):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'config': config,
        'best_val_accuracy': best_val_accuracy,
        'epoch': epoch
    }
    torch.save(checkpoint, save_path)
    logger.info(f"Saved checkpoint to {save_path}")


def save_train_history(history: List[Dict], save_path: Path):
    df = pd.DataFrame(history)
    df.to_csv(save_path, index=False)
    logger.info(f"Saved training history to {save_path}")


def train_model(config: dict) -> Dict:
    logger.info("Starting training...")
    
    device = get_device(config['runtime']['device'])
    logger.info(f"Using device: {device}")
    
    X_img, y, dates, future_returns = load_processed_data()
    
    train_idx, val_idx, test_idx = get_split_indices(dates, config)
    
    batch_size = config['model']['batch_size']
    train_loader, val_loader = create_dataloaders(X_img, y, train_idx, val_idx, batch_size)
    
    in_channels = config['model']['in_channels']
    num_classes = config['model']['num_classes']
    model = SimpleCNN(in_channels=in_channels, num_classes=num_classes)
    model = model.to(device)
    logger.info(f"Model: {model.__class__.__name__}")
    
    criterion = nn.CrossEntropyLoss()
    
    learning_rate = config['model']['learning_rate']
    weight_decay = config['model']['weight_decay']
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    epochs = config['model']['epochs']
    
    models_dir = Path('outputs/models')
    ensure_dir(str(models_dir))
    
    best_val_accuracy = 0.0
    history = []
    
    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_accuracy = evaluate(model, val_loader, criterion, device)
        
        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_accuracy': val_accuracy
        })
        
        logger.info(
            f"Epoch {epoch}/{epochs} - "
            f"Train Loss: {train_loss:.4f}, "
            f"Val Loss: {val_loss:.4f}, "
            f"Val Accuracy: {val_accuracy:.4f}"
        )
        
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            save_checkpoint(
                model, config, best_val_accuracy, epoch,
                models_dir / 'best_model.pt'
            )
    
    reports_dir = Path('outputs/reports')
    ensure_dir(str(reports_dir))
    save_train_history(history, reports_dir / 'train_history.csv')
    
    logger.info(f"Training completed. Best Val Accuracy: {best_val_accuracy:.4f}")
    
    return {
        'best_val_accuracy': best_val_accuracy,
        'final_epoch': epochs,
        'history': history
    }
