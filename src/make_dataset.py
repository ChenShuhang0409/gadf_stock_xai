import os
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils import ensure_dir, save_json, setup_logger

logger = setup_logger()


def compute_labels(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    df = df.copy()
    
    horizon = config['label']['horizon']
    threshold = config['label']['threshold']
    
    future_closes = []
    for i in range(1, horizon + 1):
        future_closes.append(df['close'].shift(-i))
    
    future_avg_close = pd.concat(future_closes, axis=1).mean(axis=1)
    
    df['future_avg_close'] = future_avg_close
    df['label_return'] = future_avg_close / df['close'] - 1
    df['label'] = (df['label_return'] > threshold).astype(int)
    
    logger.info(f"Label distribution: {df['label'].value_counts().to_dict()}")
    
    return df


def create_sliding_windows(df: pd.DataFrame, config: dict):
    window_size = config['dataset']['window_size']
    feature_names = config['features']['names']
    
    X_ts_list = []
    y_list = []
    dates_list = []
    future_returns_list = []
    
    for i in range(len(df) - window_size):
        if i + window_size >= len(df):
            break
        
        if pd.isna(df.iloc[i + window_size]['label']):
            continue
        
        window_data = df.iloc[i:i + window_size][feature_names].values
        
        if np.any(np.isnan(window_data)):
            continue
        
        X_ts_list.append(window_data.T)
        
        y_list.append(df.iloc[i + window_size]['label'])
        
        dates_list.append(df.iloc[i + window_size]['date'])
        
        future_returns_list.append(df.iloc[i + window_size]['label_return'])
    
    X_ts = np.array(X_ts_list)
    y = np.array(y_list)
    dates = np.array(dates_list)
    future_returns = np.array(future_returns_list)
    
    logger.info(f"Created {len(X_ts)} samples")
    logger.info(f"X_ts shape: {X_ts.shape}")
    logger.info(f"y shape: {y.shape}")
    
    return X_ts, y, dates, future_returns


def split_by_date(dates: np.ndarray, config: dict):
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
    
    return train_idx, val_idx, test_idx


def build_dataset(df: pd.DataFrame, config: dict) -> dict:
    logger.info("Building dataset...")
    
    df = compute_labels(df, config)
    
    X_ts, y, dates, future_returns = create_sliding_windows(df, config)
    
    train_idx, val_idx, test_idx = split_by_date(dates, config)
    
    data_dict = {
        'X_ts': X_ts,
        'y': y,
        'dates': dates,
        'future_returns': future_returns,
        'train_idx': train_idx,
        'val_idx': val_idx,
        'test_idx': test_idx
    }
    
    logger.info(f"Train samples: {train_idx.sum()}")
    logger.info(f"Val samples: {val_idx.sum()}")
    logger.info(f"Test samples: {test_idx.sum()}")
    
    processed_dir = Path('data/processed')
    ensure_dir(str(processed_dir))
    
    np.save(processed_dir / 'X_ts.npy', X_ts)
    np.save(processed_dir / 'y.npy', y)
    np.save(processed_dir / 'dates.npy', dates)
    np.save(processed_dir / 'future_returns.npy', future_returns)
    
    logger.info(f"Saved processed data to {processed_dir}")
    
    stats = {
        'total_samples': len(X_ts),
        'train_samples': int(train_idx.sum()),
        'val_samples': int(val_idx.sum()),
        'test_samples': int(test_idx.sum()),
        'X_ts_shape': list(X_ts.shape),
        'feature_names': config['features']['names'],
        'window_size': config['dataset']['window_size']
    }
    save_json(stats, str(processed_dir / 'dataset_stats.json'))
    
    return data_dict
