import numpy as np
import pandas as pd

from src.utils import setup_logger

logger = setup_logger()


def compute_log_return_single(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    return df


def compute_volume_change_single(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['volume_change'] = df['volume'] / df['volume'].shift(1) - 1
    return df


def compute_rsi_single(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    df = df.copy()
    
    delta = df['close'].diff()
    
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    return df


def compute_bias_single(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    df = df.copy()
    ma = df['close'].rolling(window=period, min_periods=period).mean()
    df['BIAS'] = (df['close'] - ma) / ma
    return df


def get_feature_names(mode: str) -> list:
    if mode == 'current':
        return ['log_return', 'volume_change', 'RSI', 'BIAS']
    elif mode == 'paper':
        return ['close', 'volume', 'RSI', 'BIAS']
    else:
        raise ValueError(f"Unknown feature mode: {mode}. Supported modes: 'current', 'paper'")


def add_indicators(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    logger.info("Adding technical indicators...")
    
    df = df.copy()
    
    rsi_period = config['features']['rsi_period']
    bias_period = config['features']['bias_period']
    feature_mode = config['features'].get('mode', 'current')
    
    has_asset = 'asset' in df.columns
    has_ticker = 'ticker' in df.columns
    
    group_col = None
    if has_ticker:
        group_col = 'ticker'
    elif has_asset:
        group_col = 'asset'
    
    if group_col:
        logger.info(f"Computing indicators per {group_col}...")
        
        df_list = []
        for group_name, group_df in df.groupby(group_col):
            group_df = group_df.sort_values('date').reset_index(drop=True)
            
            group_df = compute_log_return_single(group_df)
            group_df = compute_volume_change_single(group_df)
            group_df = compute_rsi_single(group_df, period=rsi_period)
            group_df = compute_bias_single(group_df, period=bias_period)
            
            df_list.append(group_df)
        
        df = pd.concat(df_list, ignore_index=True)
    else:
        df = compute_log_return_single(df)
        df = compute_volume_change_single(df)
        df = compute_rsi_single(df, period=rsi_period)
        df = compute_bias_single(df, period=bias_period)
    
    df = df.replace([np.inf, -np.inf], np.nan)
    
    initial_len = len(df)
    df = df.dropna()
    dropped = initial_len - len(df)
    
    if dropped > 0:
        logger.info(f"Dropped {dropped} rows with NaN values")
    
    feature_names = get_feature_names(feature_mode)
    required_cols = ['date', 'open', 'high', 'low', 'close', 'volume',
                     'log_return', 'volume_change', 'RSI', 'BIAS']
    if has_ticker:
        required_cols.append('ticker')
    if has_asset:
        required_cols.append('asset')
    df = df[required_cols]
    
    logger.info(f"Feature mode: {feature_mode}")
    logger.info(f"Selected features: {feature_names}")
    logger.info(f"Final dataset shape: {df.shape}")
    
    return df
