import argparse
import warnings
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, TensorDataset

from src.models import get_model
from src.utils import load_config, set_seed, get_device, setup_logger

logger = setup_logger()


def load_small_batch_data(config: dict, n_samples: int = 64):
    processed_dir = Path('data/processed')
    
    X_img = np.load(processed_dir / 'X_img.npy')
    y = np.load(processed_dir / 'y.npy')
    
    train_start = pd.Timestamp(config['split']['train_start'])
    train_end = pd.Timestamp(config['split']['train_end'])
    
    dates = np.load(processed_dir / 'dates.npy', allow_pickle=True)
    dates_pd = pd.to_datetime(dates)
    train_idx = (dates_pd >= train_start) & (dates_pd <= train_end)
    
    if hasattr(train_idx, 'values'):
        train_idx = train_idx.values
    
    X_train = X_img[train_idx]
    y_train = y[train_idx]
    
    X_small = X_train[:n_samples]
    y_small = y_train[:n_samples]
    
    logger.info(f"Using {len(X_small)} samples from train set for sanity check")
    logger.info(f"Label distribution in small batch: 0={np.sum(y_small==0)}, 1={np.sum(y_small==1)}")
    
    return X_small, y_small


def train_small_batch(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    config: dict,
    device: torch.device,
    epochs: int = 100
):
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    
    criterion = nn.CrossEntropyLoss()
    learning_rate = config['model']['learning_rate']
    weight_decay = config['model']['weight_decay']
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    logger.info(f"Training for {epochs} epochs...")
    logger.info(f"Learning rate: {learning_rate}")
    logger.info(f"Weight decay: {weight_decay}")
    
    for epoch in range(1, epochs + 1):
        model.train()
        
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
        
        if epoch % 10 == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                outputs = model(X_tensor.to(device))
                _, preds = torch.max(outputs, 1)
                preds = preds.cpu().numpy()
                acc = accuracy_score(y, preds)
            
            logger.info(f"Epoch {epoch:3d}: Train Loss = {loss.item():.4f}, Train Acc = {acc:.4f}")
    
    model.eval()
    with torch.no_grad():
        outputs = model(X_tensor.to(device))
        probs = torch.softmax(outputs, dim=1)
        _, preds = torch.max(outputs, 1)
        preds = preds.cpu().numpy()
        probs = probs.cpu().numpy()
        final_acc = accuracy_score(y, preds)
    
    logger.info(f"Final Train Accuracy: {final_acc:.4f}")
    
    pred_0 = int(np.sum(preds == 0))
    pred_1 = int(np.sum(preds == 1))
    logger.info(f"Final predictions: 0={pred_0}, 1={pred_1}")
    logger.info(f"Prob up mean: {np.mean(probs[:, 1]):.4f}, std: {np.std(probs[:, 1]):.4f}")
    
    return final_acc


def main(config_path: str):
    config = load_config(config_path)
    logger.info(f"Loaded config from: {config_path}")
    
    set_seed(config['runtime']['seed'])
    device = get_device(config['runtime']['device'])
    logger.info(f"Using device: {device}")
    
    logger.info("=" * 60)
    logger.info("Small-Batch Overfit Sanity Check")
    logger.info("=" * 60)
    
    X_small, y_small = load_small_batch_data(config, n_samples=64)
    
    model = get_model(config)
    model = model.to(device)
    logger.info(f"Model: {model.__class__.__name__}")
    
    final_acc = train_small_batch(model, X_small, y_small, config, device, epochs=100)
    
    logger.info("=" * 60)
    logger.info("Sanity Check Result")
    logger.info("=" * 60)
    
    if final_acc < 0.90:
        warnings.warn(
            "Model cannot overfit a small batch. Check model architecture, labels, or input pipeline."
        )
        logger.warning(f"Final accuracy {final_acc:.4f} is below 0.90 threshold")
    else:
        logger.info(f"PASSED: Model successfully overfits small batch with accuracy {final_acc:.4f}")
    
    logger.info("Sanity check completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Small-Batch Overfit Sanity Check")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/spy_daily.yaml",
        help="Path to configuration file"
    )
    args = parser.parse_args()
    
    main(args.config)
