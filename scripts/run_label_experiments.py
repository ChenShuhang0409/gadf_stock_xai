import argparse
import copy
import json
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import torch

from src.utils import load_config, set_seed, get_device, ensure_dir, save_json, setup_logger
from src.data_downloader import load_data_from_csv
from src.indicators import add_indicators, get_feature_names
from src.make_dataset import build_dataset
from src.gadf_encoder import encode_gadf
from src.models import get_model
from src.train import (
    create_dataloaders,
    train_epoch,
    evaluate_with_metrics,
    save_checkpoint,
    compute_class_weights
)
from src.evaluate import (
    compute_metrics,
    compute_baseline_metrics,
    compute_diagnostics,
    predict,
    find_best_threshold
)

logger = setup_logger()


def run_single_experiment(
    df_features: pd.DataFrame,
    base_config: dict,
    data_mode: str,
    feature_mode: str,
    horizon: int,
    threshold: float,
    neutral_policy: str,
    experiment_dir: Path,
    epochs: int
) -> dict:
    config = copy.deepcopy(base_config)
    config['features']['mode'] = feature_mode
    config['label']['horizon'] = horizon
    config['label']['threshold'] = threshold
    config['label']['neutral_policy'] = neutral_policy
    config['model']['epochs'] = epochs
    
    exp_name = f"data_{data_mode}_feature_{feature_mode}_horizon_{horizon}_threshold_{str(threshold).replace('.', 'p')}"
    exp_dir = experiment_dir / exp_name
    ensure_dir(str(exp_dir))
    
    logger.info("=" * 60)
    logger.info(f"Experiment: data_mode={data_mode}, feature_mode={feature_mode}, horizon={horizon}, threshold={threshold}")
    logger.info("=" * 60)
    
    data_dict = build_dataset(df_features, config)
    
    X_ts = data_dict['X_ts']
    y = data_dict['y']
    dates = data_dict['dates']
    future_returns = data_dict['future_returns']
    assets = data_dict['assets']
    train_idx = data_dict['train_idx']
    val_idx = data_dict['val_idx']
    test_idx = data_dict['test_idx']
    
    if len(X_ts) == 0:
        logger.warning(f"No samples generated for this configuration. Skipping.")
        return None
    
    n_train = int(train_idx.sum())
    n_val = int(val_idx.sum())
    n_test = int(test_idx.sum())
    
    if n_train == 0 or n_val == 0:
        logger.warning(f"Insufficient train/val samples. Skipping.")
        return None
    
    logger.info(f"Train: {n_train}, Val: {n_val}, Test: {n_test}")
    
    n_assets = 0
    assets_list = []
    if assets is not None:
        n_assets = len(np.unique(assets))
        assets_list = np.unique(assets).tolist()
        logger.info(f"Assets: {assets_list}")
    
    X_img = encode_gadf(X_ts, config)
    
    device = get_device(config['runtime']['device'])
    batch_size = config['model']['batch_size']
    
    train_loader, val_loader, y_train, y_val = create_dataloaders(
        X_img, y, train_idx, val_idx, batch_size
    )
    
    model = get_model(config)
    model = model.to(device)
    
    use_class_weights = config['model'].get('use_class_weights', False)
    if use_class_weights:
        class_weights = compute_class_weights(y_train)
        class_weights = class_weights.to(device)
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = torch.nn.CrossEntropyLoss()
    
    learning_rate = config['model']['learning_rate']
    weight_decay = config['model']['weight_decay']
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    save_metric = config['model'].get('save_metric', 'val_auc')
    best_metric_value = float('inf') if 'loss' in save_metric else float('-inf')
    best_epoch = 0
    best_val_auc = None
    
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_metrics = evaluate_with_metrics(model, val_loader, criterion, device)
        
        current_metric_value = None
        if save_metric == 'val_auc':
            if val_metrics['auc'] is not None:
                current_metric_value = val_metrics['auc']
            else:
                current_metric_value = val_loss
        elif save_metric == 'val_loss':
            current_metric_value = val_loss
        elif save_metric == 'val_acc':
            current_metric_value = val_acc
        else:
            current_metric_value = val_loss
        
        if 'loss' in save_metric:
            is_better = current_metric_value < best_metric_value
        else:
            is_better = current_metric_value > best_metric_value
        
        if is_better:
            best_metric_value = current_metric_value
            best_epoch = epoch
            if val_metrics['auc'] is not None:
                best_val_auc = val_metrics['auc']
    
    if n_test > 0:
        X_test = X_img[test_idx]
        y_test = y[test_idx]
        dates_test = dates[test_idx]
        future_returns_test = future_returns[test_idx]
        assets_test = assets[test_idx] if assets is not None else None
        
        probs, preds_default = predict(model, X_test, batch_size, device)
        
        X_val_data = X_img[val_idx]
        y_val_data = y[val_idx]
        val_probs, _ = predict(model, X_val_data, batch_size, device)
        
        best_threshold, val_thresh_metrics = find_best_threshold(y_val_data, val_probs)
        
        preds_with_threshold = (probs[:, 1] >= best_threshold).astype(int)
        
        metrics = compute_metrics(y_test, preds_with_threshold, probs)
        metrics['decision_threshold'] = float(best_threshold)
        
        baseline = compute_baseline_metrics(y_test)
        
        diagnostics = compute_diagnostics(
            y_test, preds_with_threshold, probs,
            model_accuracy=metrics['accuracy'],
            majority_baseline_accuracy=baseline['majority_class_accuracy']
        )
    else:
        metrics = {'accuracy': None, 'balanced_accuracy': None, 'precision': None, 'recall': None, 'f1': None, 'roc_auc': None, 'decision_threshold': None}
        baseline = {'majority_class_accuracy': None}
        diagnostics = {'collapsed_prediction': None, 'ready_for_lrp': None, 'beats_majority_baseline': None}
        best_threshold = 0.5
        assets_test = None
    
    label_filtering_path = Path('outputs/reports/label_filtering_stats.json')
    if label_filtering_path.exists():
        with open(label_filtering_path, 'r') as f:
            label_filtering_stats = json.load(f)
        pct_dropped = label_filtering_stats.get('pct_dropped', None)
    else:
        pct_dropped = None
    
    save_json(metrics, str(exp_dir / 'metrics.json'))
    save_json(baseline, str(exp_dir / 'baseline_metrics.json'))
    save_json(diagnostics, str(exp_dir / 'prediction_diagnostics.json'))
    if label_filtering_path.exists():
        import shutil
        shutil.copy(str(label_filtering_path), str(exp_dir / 'label_filtering_stats.json'))
    
    accuracy_minus_baseline = None
    if metrics.get('accuracy') is not None and baseline.get('majority_class_accuracy') is not None:
        accuracy_minus_baseline = metrics['accuracy'] - baseline['majority_class_accuracy']
    
    result = {
        'data_mode': data_mode,
        'n_assets': n_assets,
        'assets': str(assets_list) if assets_list else None,
        'feature_mode': feature_mode,
        'horizon': horizon,
        'threshold': threshold,
        'neutral_policy': neutral_policy,
        'n_train': n_train,
        'n_val': n_val,
        'n_test': n_test,
        'pct_dropped': pct_dropped,
        'decision_threshold': float(best_threshold) if best_threshold is not None else None,
        'test_accuracy': metrics.get('accuracy'),
        'majority_baseline_accuracy': baseline.get('majority_class_accuracy'),
        'accuracy_minus_baseline': accuracy_minus_baseline,
        'balanced_accuracy': metrics.get('balanced_accuracy'),
        'beats_majority_baseline': diagnostics.get('beats_majority_baseline'),
        'roc_auc': metrics.get('roc_auc'),
        'f1': metrics.get('f1'),
        'precision': metrics.get('precision'),
        'recall': metrics.get('recall'),
        'collapsed_prediction': diagnostics.get('collapsed_prediction'),
        'ready_for_lrp': diagnostics.get('ready_for_lrp'),
        'best_val_auc': best_val_auc,
        'best_epoch': best_epoch
    }
    
    logger.info(f"Experiment result: accuracy={metrics.get('accuracy')}, auc={metrics.get('roc_auc')}, ready_for_lrp={diagnostics.get('ready_for_lrp')}")
    
    return result


def main(config_path: str, quick: bool = False):
    config = load_config(config_path)
    logger.info(f"Loaded config from: {config_path}")
    
    set_seed(config['runtime']['seed'])
    
    data_mode = config['data'].get('mode', 'single_ticker')
    
    if quick:
        feature_modes = ['paper']
        horizons = [5, 10]
        thresholds = [0.0, 0.002]
        epochs = 8
        logger.info("Running in QUICK mode (reduced combinations, 8 epochs)")
    else:
        feature_modes = ['current', 'paper']
        horizons = [3, 5, 10]
        thresholds = [0.0, 0.002]
        epochs = 15
        logger.info("Running in FULL mode (2x3x2=12 combinations, 15 epochs)")
    
    neutral_policy = "drop"
    
    logger.info(f"Data mode: {data_mode}")
    logger.info(f"Feature modes: {feature_modes}")
    logger.info(f"Horizons: {horizons}")
    logger.info(f"Thresholds: {thresholds}")
    logger.info(f"Neutral policy: {neutral_policy}")
    logger.info(f"Epochs per experiment: {epochs}")
    
    df_raw = load_data_from_csv(config)
    df_features = add_indicators(df_raw, config)
    
    experiment_dir = Path('outputs/experiments')
    ensure_dir(str(experiment_dir))
    
    results = []
    
    total_combinations = len(feature_modes) * len(horizons) * len(thresholds)
    current = 0
    
    for feature_mode in feature_modes:
        for horizon in horizons:
            for threshold in thresholds:
                current += 1
                logger.info(f"\n{'='*60}")
                logger.info(f"Progress: {current}/{total_combinations}")
                logger.info(f"{'='*60}")
                
                try:
                    result = run_single_experiment(
                        df_features,
                        config,
                        data_mode,
                        feature_mode,
                        horizon,
                        threshold,
                        neutral_policy,
                        experiment_dir,
                        epochs
                    )
                    
                    if result is not None:
                        results.append(result)
                except Exception as e:
                    logger.error(f"Experiment failed: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
    
    if len(results) > 0:
        df_results = pd.DataFrame(results)
        
        df_results = df_results.sort_values(
            by=['ready_for_lrp', 'roc_auc', 'accuracy_minus_baseline', 'balanced_accuracy'],
            ascending=[False, False, False, False],
            na_position='last'
        )
        
        summary_path = experiment_dir / 'experiment_summary.csv'
        df_results.to_csv(summary_path, index=False)
        logger.info(f"\nSaved experiment summary to {summary_path}")
        
        logger.info("\n" + "=" * 60)
        logger.info("EXPERIMENT SUMMARY")
        logger.info("=" * 60)
        
        ready_for_lrp_count = df_results['ready_for_lrp'].sum() if 'ready_for_lrp' in df_results.columns else 0
        logger.info(f"Data mode: {data_mode}")
        logger.info(f"Total experiments: {len(results)}")
        logger.info(f"Experiments ready for LRP: {ready_for_lrp_count}")
        
        if ready_for_lrp_count > 0:
            logger.info("\nBest configurations (ready for LRP):")
            best = df_results[df_results['ready_for_lrp'] == True].head(3)
            for _, row in best.iterrows():
                logger.info(f"  feature={row['feature_mode']}, horizon={row['horizon']}, threshold={row['threshold']}: "
                           f"acc={row['test_accuracy']:.4f}, auc={row['roc_auc']:.4f}")
        else:
            if data_mode == 'multi_asset':
                logger.warning("\nNo multi_asset configuration is ready for LRP.")
            else:
                logger.warning("\nNo configuration beats majority baseline with AUC > 0.52")
            
            logger.info("\nTop configurations by AUC:")
            best_auc = df_results.nlargest(5, 'roc_auc')
            for _, row in best_auc.iterrows():
                acc_baseline = f"{row['accuracy_minus_baseline']:.4f}" if row['accuracy_minus_baseline'] is not None else 'N/A'
                logger.info(f"  feature={row['feature_mode']}, horizon={row['horizon']}, threshold={row['threshold']}: "
                           f"acc={row['test_accuracy']:.4f}, auc={row['roc_auc']:.4f}, "
                           f"acc-baseline={acc_baseline}")
    else:
        logger.warning("No experiments completed successfully.")
    
    logger.info("\nAll experiments completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Label Parameter Experiments")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/spy_daily.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run in quick mode (fewer combinations, fewer epochs)"
    )
    args = parser.parse_args()
    
    main(args.config, quick=args.quick)
