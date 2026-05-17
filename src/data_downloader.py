import os
from pathlib import Path
from typing import Optional, Dict, List

import numpy as np
import pandas as pd

from src.utils import setup_logger

logger = setup_logger()


def standardize_columns(df: pd.DataFrame, ticker_col: str = None) -> pd.DataFrame:
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
        'ticker': 'ticker',
        'Name': 'ticker',
        'name': 'ticker'
    }
    
    df.columns = [column_mapping.get(col, col) for col in df.columns]
    
    if ticker_col is not None and ticker_col.lower() != 'ticker':
        if ticker_col in df.columns:
            df = df.rename(columns={ticker_col: 'ticker'})
    
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


def load_and_filter_csv(csv_path: str, asset_name: str = None) -> pd.DataFrame:
    logger.info(f"Loading CSV file: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    logger.info(f"CSV columns: {df.columns.tolist()}")
    logger.info(f"CSV shape: {df.shape}")
    
    df = standardize_columns(df)
    
    if 'ticker' in df.columns and asset_name is not None:
        logger.info(f"Found 'ticker' column, filtering for {asset_name}")
        df = df[df['ticker'].str.upper() == asset_name.upper()]
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
    
    sort_cols = []
    if 'asset' in df.columns:
        sort_cols.append('asset')
    if 'ticker' in df.columns:
        sort_cols.append('ticker')
    sort_cols.append('date')
    
    df = df.sort_values(sort_cols).reset_index(drop=True)
    
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


def load_single_ticker(config: dict) -> pd.DataFrame:
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
    
    df['ticker'] = ticker
    
    df = validate_data(df, config)
    
    logger.info(f"Successfully loaded {len(df)} rows for {ticker}")
    logger.info(f"Date range: {df['date'].min()} to {df['date'].max()}")
    logger.info(f"Columns: {df.columns.tolist()}")
    
    return df


def load_multi_asset(config: dict) -> pd.DataFrame:
    assets = config['data'].get('assets', {})
    local_assets = config['data'].get('local_assets', {})
    
    if not assets and not local_assets:
        raise ValueError("No assets configured. Please specify 'assets' or 'local_assets' in config.")
    
    all_dfs = []
    
    for asset_name, csv_path in assets.items():
        local_path = local_assets.get(asset_name)
        
        actual_path = None
        
        if os.path.exists(csv_path):
            actual_path = csv_path
            logger.info(f"Found Kaggle path for {asset_name}: {csv_path}")
        elif local_path and os.path.exists(local_path):
            actual_path = local_path
            logger.info(f"Found local path for {asset_name}: {local_path}")
        else:
            logger.error(f"File not found for asset: {asset_name}")
            logger.error(f"  Kaggle path: {csv_path}")
            if local_path:
                logger.error(f"  Local path: {local_path}")
            raise FileNotFoundError(f"Data file not found for asset: {asset_name}")
        
        logger.info(f"Loading asset: {asset_name}")
        logger.info(f"  File path: {actual_path}")
        
        df = load_and_filter_csv(actual_path, asset_name)
        df['ticker'] = asset_name
        
        df['date'] = pd.to_datetime(df['date'])
        
        start_date = pd.Timestamp(config['data']['start_date'])
        end_date = pd.Timestamp(config['data']['end_date'])
        df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
        
        initial_len = len(df)
        df = df.dropna()
        dropped = initial_len - len(df)
        
        logger.info(f"  Date range: {df['date'].min()} to {df['date'].max()}")
        logger.info(f"  Row count: {len(df)} (dropped {dropped} NaN rows)")
        
        all_dfs.append(df)
    
    if not all_dfs:
        raise ValueError("No data loaded from any asset")
    
    combined_df = pd.concat(all_dfs, ignore_index=True)
    
    combined_df = combined_df.sort_values(['ticker', 'date']).reset_index(drop=True)
    
    logger.info("=" * 60)
    logger.info("Multi-Asset Data Summary")
    logger.info("=" * 60)
    
    for asset_name in assets.keys():
        asset_df = combined_df[combined_df['ticker'] == asset_name]
        if len(asset_df) > 0:
            logger.info(f"  {asset_name}: {len(asset_df)} rows, "
                       f"{asset_df['date'].min().date()} to {asset_df['date'].max().date()}")
    
    logger.info(f"Total combined rows: {len(combined_df)}")
    logger.info("=" * 60)
    
    return combined_df


def load_multi_ticker(config: dict) -> pd.DataFrame:
    csv_path = config['data']['csv_path']
    ticker_col = config['data'].get('ticker_col', 'Name')
    
    logger.info(f"Loading multi-ticker data from: {csv_path}")
    logger.info(f"Ticker column: {ticker_col}")
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Data file not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    logger.info(f"CSV columns: {df.columns.tolist()}")
    logger.info(f"CSV shape: {df.shape}")
    
    df = standardize_columns(df, ticker_col)
    
    required_cols = ['date', 'open', 'high', 'low', 'close', 'volume', 'ticker']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}. Available columns: {df.columns.tolist()}")
    
    df['date'] = pd.to_datetime(df['date'])
    
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    start_date = pd.Timestamp(config['data']['start_date'])
    end_date = pd.Timestamp(config['data']['end_date'])
    df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
    
    initial_len = len(df)
    df = df.dropna()
    dropped = initial_len - len(df)
    
    if dropped > 0:
        logger.info(f"Dropped {dropped} rows with NaN values")
    
    df = df.sort_values(['ticker', 'date']).reset_index(drop=True)
    
    unique_tickers = df['ticker'].unique()
    n_tickers = len(unique_tickers)
    
    ticker_counts = df.groupby('ticker').size()
    
    logger.info("=" * 60)
    logger.info("Multi-Ticker Data Summary")
    logger.info("=" * 60)
    logger.info(f"Number of tickers: {n_tickers}")
    logger.info(f"Total rows: {len(df)}")
    logger.info(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    logger.info(f"Samples per ticker:")
    logger.info(f"  min: {ticker_counts.min()}")
    logger.info(f"  median: {ticker_counts.median():.0f}")
    logger.info(f"  max: {ticker_counts.max()}")
    logger.info("=" * 60)
    
    return df


def load_data_from_csv(config: dict) -> pd.DataFrame:
    data_mode = config['data'].get('mode', 'single_ticker')
    
    logger.info(f"Data loading mode: {data_mode}")
    
    if data_mode == 'single_ticker':
        return load_single_ticker(config)
    elif data_mode == 'multi_asset':
        return load_multi_asset(config)
    elif data_mode == 'multi_ticker':
        return load_multi_ticker(config)
    else:
        raise ValueError(f"Unknown data mode: {data_mode}. Supported modes: 'single_ticker', 'multi_asset', 'multi_ticker'")
