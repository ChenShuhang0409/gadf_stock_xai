import numpy as np
import pandas as pd

from src.utils import setup_logger

logger = setup_logger()


def compute_log_return(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    return df


def compute_volume_change(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['volume_change'] = df['volume'] / df['volume'].shift(1) - 1
    return df


def compute_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    df = df.copy()
    
    delta = df['close'].diff()
    
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    return df


def compute_bias(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    df = df.copy()
    ma = df['close'].rolling(window=period, min_periods=period).mean()
    df['BIAS'] = (df['close'] - ma) / ma
    return df


def add_indicators(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    logger.info("Adding technical indicators...")
    
    df = df.copy()
    
    rsi_period = config['features']['rsi_period']
    bias_period = config['features']['bias_period']
    
    df = compute_log_return(df)
    df = compute_volume_change(df)
    df = compute_rsi(df, period=rsi_period)
    df = compute_bias(df, period=bias_period)
    
    df = df.replace([np.inf, -np.inf], np.nan)
    
    initial_len = len(df)
    df = df.dropna()
    dropped = initial_len - len(df)
    
    if dropped > 0:
        logger.info(f"Dropped {dropped} rows with NaN values")
    
    required_cols = ['date', 'open', 'high', 'low', 'close', 'volume',
                     'log_return', 'volume_change', 'RSI', 'BIAS']
    df = df[required_cols]
    
    logger.info(f"Added indicators: {df.columns.tolist()}")
    logger.info(f"Final dataset shape: {df.shape}")
    
    return df
