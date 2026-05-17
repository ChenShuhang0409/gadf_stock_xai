import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.utils import setup_logger

logger = setup_logger()


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    column_mapping = {
        'Date': 'date',
        'date': 'date',
        'datetime': 'date',
        'time': 'date',
        'Open': 'open',
        'open': 'open',
        'High': 'high',
        'high': 'high',
        'Low': 'low',
        'low': 'low',
        'Close': 'close',
        'close': 'close',
        'Adj Close': 'close',
        'adj_close': 'close',
        'Volume': 'volume',
        'volume': 'volume',
        'Symbol': 'ticker',
        'symbol': 'ticker',
        'Ticker': 'ticker',
        'ticker': 'ticker'
    }
    
    df.columns = [column_mapping.get(col, col) for col in df.columns]
    
    return df


def find_spy_csv_in_directory(directory: str, ticker: str) -> Optional[str]:
    logger.info(f"Searching for {ticker} CSV files in directory: {directory}")
    
    csv_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.csv'):
                csv_files.append(os.path.join(root, file))
    
    if not csv_files:
        logger.error(f"No CSV files found in {directory}")
        return None
    
    logger.info(f"Found {len(csv_files)} CSV files:")
    for csv_file in csv_files:
        logger.info(f"  - {csv_file}")
    
    ticker_lower = ticker.lower()
    ticker_upper = ticker.upper()
    
    for csv_file in csv_files:
        filename = os.path.basename(csv_file)
        if ticker_lower in filename.lower() or ticker_upper in filename:
            logger.info(f"Found {ticker} file: {csv_file}")
            return csv_file
    
    logger.error(f"No CSV file with '{ticker}' in filename found")
    logger.error("Please manually specify the correct CSV file path in config")
    return None


def load_and_filter_csv(csv_path: str, ticker: str) -> pd.DataFrame:
    logger.info(f"Loading CSV file: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    logger.info(f"CSV columns: {df.columns.tolist()}")
    logger.info(f"CSV shape: {df.shape}")
    
    df = standardize_columns(df)
    
    if 'ticker' in df.columns:
        logger.info(f"Found 'ticker' column, filtering for {ticker}")
        df = df[df['ticker'].str.upper() == ticker.upper()]
        logger.info(f"After filtering by ticker: {len(df)} rows")
    
    required_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}. Available columns: {df.columns.tolist()}")
    
    df = df[required_cols].copy()
    
    return df


def validate_data(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    logger.info("Validating data...")
    
    df['date'] = pd.to_datetime(df['date'])
    
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df = df.sort_values('date').reset_index(drop=True)
    
    start_date = pd.Timestamp(config['data']['start_date'])
    end_date = pd.Timestamp(config['data']['end_date'])
    
    df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
    
    initial_len = len(df)
    df = df.dropna()
    dropped = initial_len - len(df)
    
    if dropped > 0:
        logger.info(f"Dropped {dropped} rows with NaN values")
    
    if len(df) < 500:
        raise ValueError(f"Insufficient data: only {len(df)} rows after filtering (minimum 500 required)")
    
    logger.info(f"Data validation passed: {len(df)} rows")
    
    return df


def load_data_from_csv(config: dict) -> pd.DataFrame:
    ticker = config['data']['ticker']
    csv_path = config['data']['csv_path']
    local_csv_path = config['data']['local_csv_path']
    
    actual_csv_path = None
    
    if os.path.exists(csv_path):
        logger.info(f"Found Kaggle path: {csv_path}")
        
        if os.path.isfile(csv_path):
            actual_csv_path = csv_path
        elif os.path.isdir(csv_path):
            actual_csv_path = find_spy_csv_in_directory(csv_path, ticker)
    elif os.path.exists(local_csv_path):
        logger.info(f"Found local path: {local_csv_path}")
        actual_csv_path = local_csv_path
    else:
        logger.error(f"Kaggle path not found: {csv_path}")
        logger.error(f"Local path not found: {local_csv_path}")
        raise FileNotFoundError(
            f"Data file not found. Please ensure either:\n"
            f"  1. Kaggle dataset is added at: {csv_path}\n"
            f"  2. Local CSV file exists at: {local_csv_path}"
        )
    
    if actual_csv_path is None:
        raise FileNotFoundError(f"Could not find {ticker} CSV file in the specified paths")
    
    logger.info(f"Loading data from: {actual_csv_path}")
    
    df = load_and_filter_csv(actual_csv_path, ticker)
    
    df = validate_data(df, config)
    
    logger.info(f"Successfully loaded {len(df)} rows")
    logger.info(f"Date range: {df['date'].min()} to {df['date'].max()}")
    logger.info(f"Columns: {df.columns.tolist()}")
    
    return df
