import os
from pathlib import Path

import numpy as np
import pandas as pd

from src.indicators import get_feature_names
from src.utils import ensure_dir, save_json, setup_logger

logger = setup_logger()


def compute_labels(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    df = df.copy()
    
    horizon = config['label']['horizon']
    threshold = config['label']['threshold']
    neutral_policy = config['label'].get('neutral_policy', 'drop')
    
    has_asset = 'asset' in df.columns
    
    if has_asset:
        df_list = []
        for asset_name, asset_df in df.groupby('asset'):
            asset_df = asset_df.sort_values('date').reset_index(drop=True)
            asset_df = asset_df.copy()
            
            future_closes = []
            for i in range(1, horizon + 1):
                future_closes.append(asset_df['close'].shift(-i))
            
            future_avg_close = pd.concat(future_closes, axis=1).mean(axis=1)
            
            asset_df['future_avg_close'] = future_avg_close
            asset_df['label_return'] = future_avg_close / asset_df['close'] - 1
            df_list.append(asset_df)
        
        df = pd.concat(df_list, ignore_index=True)
    else:
        future_closes = []
        for i in range(1, horizon + 1):
            future_closes.append(df['close'].shift(-i))
        
        future_avg_close = pd.concat(future_closes, axis=1).mean(axis=1)
        
        df['future_avg_close'] = future_avg_close
        df['label_return'] = future_avg_close / df['close'] - 1
    
    def assign_label(row):
        ret = row['label_return']
        if pd.isna(ret):
            return -1
        if ret > threshold:
            return 1
        elif ret < -threshold:
            return 0
        else:
            return -1
    
    df['label'] = df.apply(assign_label, axis=1)
    
    n_up = int((df['label'] == 1).sum())
    n_down = int((df['label'] == 0).sum())
    n_neutral = int((df['label'] == -1).sum())
    n_total = len(df)
    
    logger.info(f"Label distribution before filtering:")
    logger.info(f"  label=1 (up): {n_up} ({n_up/n_total*100:.2f}%)")
    logger.info(f"  label=0 (down): {n_down} ({n_down/n_total*100:.2f}%)")
    logger.info(f"  label=-1 (neutral): {n_neutral} ({n_neutral/n_total*100:.2f}%)")
    
    return df


def create_sliding_windows(df: pd.DataFrame, config: dict):
    window_size = config['dataset']['window_size']
    feature_mode = config['features'].get('mode', 'current')
    feature_names = get_feature_names(feature_mode)
    neutral_policy = config['label'].get('neutral_policy', 'drop')
    
    logger.info(f"Using feature mode: {feature_mode}")
    logger.info(f"Feature names: {feature_names}")
    
    has_asset = 'asset' in df.columns
    
    X_ts_list = []
    y_list = []
    dates_list = []
    future_returns_list = []
    assets_list = []
    
    if has_asset:
        for asset_name, asset_df in df.groupby('asset'):
            asset_df = asset_df.sort_values('date').reset_index(drop=True)
            
            for i in range(len(asset_df) - window_size):
                if i + window_size >= len(asset_df):
                    break
                
                label = asset_df.iloc[i + window_size]['label']
                
                if pd.isna(label):
                    continue
                
                if label == -1:
                    if neutral_policy == 'drop':
                        continue
                    elif neutral_policy == 'keep_as_0':
                        label = 0
                    else:
                        continue
                
                window_data = asset_df.iloc[i:i + window_size][feature_names].values
                
                if np.any(np.isnan(window_data)):
                    continue
                
                X_ts_list.append(window_data.T)
                y_list.append(int(label))
                dates_list.append(asset_df.iloc[i + window_size]['date'])
                future_returns_list.append(asset_df.iloc[i + window_size]['label_return'])
                assets_list.append(asset_name)
    else:
        for i in range(len(df) - window_size):
            if i + window_size >= len(df):
                break
            
            label = df.iloc[i + window_size]['label']
            
            if pd.isna(label):
                continue
            
            if label == -1:
                if neutral_policy == 'drop':
                    continue
                elif neutral_policy == 'keep_as_0':
                    label = 0
                else:
                    continue
            
            window_data = df.iloc[i:i + window_size][feature_names].values
            
            if np.any(np.isnan(window_data)):
                continue
            
            X_ts_list.append(window_data.T)
            y_list.append(int(label))
            dates_list.append(df.iloc[i + window_size]['date'])
            future_returns_list.append(df.iloc[i + window_size]['label_return'])
    
    X_ts = np.array(X_ts_list)
    y = np.array(y_list, dtype=np.int64)
    dates = np.array(dates_list)
    future_returns = np.array(future_returns_list)
    assets = np.array(assets_list) if assets_list else None
    
    logger.info(f"Created {len(X_ts)} samples after neutral filtering")
    logger.info(f"X_ts shape: {X_ts.shape}")
    logger.info(f"y shape: {y.shape}")
    
    if assets is not None:
        unique_assets = np.unique(assets)
        logger.info(f"Assets in dataset: {unique_assets.tolist()}")
        for asset in unique_assets:
            count = np.sum(assets == asset)
            logger.info(f"  {asset}: {count} samples")
    
    return X_ts, y, dates, future_returns, assets


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


def get_split_name(date, config):
    train_start = pd.Timestamp(config['split']['train_start'])
    train_end = pd.Timestamp(config['split']['train_end'])
    val_start = pd.Timestamp(config['split']['val_start'])
    val_end = pd.Timestamp(config['split']['val_end'])
    test_start = pd.Timestamp(config['split']['test_start'])
    test_end = pd.Timestamp(config['split']['test_end'])
    
    date = pd.Timestamp(date)
    
    if train_start <= date <= train_end:
        return 'train'
    elif val_start <= date <= val_end:
        return 'val'
    elif test_start <= date <= test_end:
        return 'test'
    else:
        return 'none'


def build_dataset(df: pd.DataFrame, config: dict) -> dict:
    logger.info("Building dataset...")
    
    threshold = config['label']['threshold']
    horizon = config['label']['horizon']
    neutral_policy = config['label'].get('neutral_policy', 'drop')
    feature_mode = config['features'].get('mode', 'current')
    feature_names = get_feature_names(feature_mode)
    data_mode = config['data'].get('mode', 'single_ticker')
    
    df = compute_labels(df, config)
    
    total_before_filter = len(df)
    n_up_before = int((df['label'] == 1).sum())
    n_down_before = int((df['label'] == 0).sum())
    n_neutral_before = int((df['label'] == -1).sum())
    
    has_asset = 'asset' in df.columns
    
    per_asset_stats = {}
    if has_asset:
        for asset_name, asset_df in df.groupby('asset'):
            per_asset_stats[asset_name] = {
                'total_before_filter': len(asset_df),
                'n_up': int((asset_df['label'] == 1).sum()),
                'n_down': int((asset_df['label'] == 0).sum()),
                'n_neutral': int((asset_df['label'] == -1).sum())
            }
    
    X_ts, y, dates, future_returns, assets = create_sliding_windows(df, config)
    
    train_idx, val_idx, test_idx = split_by_date(dates, config)
    
    data_dict = {
        'X_ts': X_ts,
        'y': y,
        'dates': dates,
        'future_returns': future_returns,
        'assets': assets,
        'train_idx': train_idx,
        'val_idx': val_idx,
        'test_idx': test_idx
    }
    
    logger.info(f"Train samples: {int(train_idx.sum())}")
    logger.info(f"Val samples: {int(val_idx.sum())}")
    logger.info(f"Test samples: {int(test_idx.sum())}")
    
    processed_dir = Path('data/processed')
    ensure_dir(str(processed_dir))
    
    np.save(processed_dir / 'X_ts.npy', X_ts)
    np.save(processed_dir / 'y.npy', y)
    np.save(processed_dir / 'dates.npy', dates)
    np.save(processed_dir / 'future_returns.npy', future_returns)
    if assets is not None:
        np.save(processed_dir / 'assets.npy', assets)
    
    splits = []
    for i in range(len(dates)):
        splits.append(get_split_name(dates[i], config))
    
    metadata_df = pd.DataFrame({
        'date': dates,
        'y': y,
        'future_return': future_returns,
        'split': splits
    })
    if assets is not None:
        metadata_df['asset'] = assets
        metadata_df = metadata_df[['asset', 'date', 'y', 'future_return', 'split']]
    
    metadata_df.to_csv(processed_dir / 'metadata.csv', index=False)
    logger.info(f"Saved metadata to {processed_dir / 'metadata.csv'}")
    
    logger.info(f"Saved processed data to {processed_dir}")
    
    stats = {
        'total_samples': len(X_ts),
        'train_samples': int(train_idx.sum()),
        'val_samples': int(val_idx.sum()),
        'test_samples': int(test_idx.sum()),
        'X_ts_shape': list(X_ts.shape),
        'feature_mode': feature_mode,
        'feature_names': feature_names,
        'data_mode': data_mode,
        'window_size': config['dataset']['window_size']
    }
    if assets is not None:
        stats['n_assets'] = len(np.unique(assets))
        stats['assets'] = np.unique(assets).tolist()
    save_json(stats, str(processed_dir / 'dataset_stats.json'))
    
    reports_dir = Path('outputs/reports')
    ensure_dir(str(reports_dir))
    
    total_after_filter = len(y)
    pct_dropped = (total_before_filter - total_after_filter) / total_before_filter * 100 if total_before_filter > 0 else 0
    
    label_filtering_stats = {
        'threshold': float(threshold),
        'horizon': int(horizon),
        'neutral_policy': neutral_policy,
        'feature_mode': feature_mode,
        'data_mode': data_mode,
        'total_before_filter': int(total_before_filter),
        'n_up_before_filter': n_up_before,
        'n_down_before_filter': n_down_before,
        'n_neutral_before_filter': n_neutral_before,
        'total_after_filter': int(total_after_filter),
        'pct_dropped': float(pct_dropped)
    }
    
    if has_asset and assets is not None:
        per_asset_after = {}
        for asset_name in np.unique(assets):
            asset_mask = assets == asset_name
            per_asset_after[asset_name] = {
                'total_after_filter': int(np.sum(asset_mask)),
                'n_train': int(np.sum(asset_mask & train_idx)),
                'n_val': int(np.sum(asset_mask & val_idx)),
                'n_test': int(np.sum(asset_mask & test_idx)),
                'n_label_0': int(np.sum((y == 0) & asset_mask)),
                'n_label_1': int(np.sum((y == 1) & asset_mask))
            }
        
        label_filtering_stats['per_asset'] = {}
        for asset_name in per_asset_stats:
            label_filtering_stats['per_asset'][asset_name] = {
                'before_filter': per_asset_stats.get(asset_name, {}),
                'after_filter': per_asset_after.get(asset_name, {})
            }
    
    save_json(label_filtering_stats, str(reports_dir / 'label_filtering_stats.json'))
    
    logger.info(f"Label filtering stats:")
    logger.info(f"  threshold: {threshold}")
    logger.info(f"  horizon: {horizon}")
    logger.info(f"  neutral_policy: {neutral_policy}")
    logger.info(f"  feature_mode: {feature_mode}")
    logger.info(f"  data_mode: {data_mode}")
    logger.info(f"  total_before_filter: {total_before_filter}")
    logger.info(f"  n_up: {n_up_before}, n_down: {n_down_before}, n_neutral: {n_neutral_before}")
    logger.info(f"  total_after_filter: {total_after_filter}")
    logger.info(f"  pct_dropped: {pct_dropped:.2f}%")
    
    return data_dict
