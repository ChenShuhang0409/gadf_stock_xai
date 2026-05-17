import argparse
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.utils import load_config, set_seed, get_device, print_step, setup_logger
from src.data_downloader import load_data_from_csv
from src.indicators import add_indicators
from src.make_dataset import build_dataset
from src.gadf_encoder import encode_gadf


def main(config_path: str):
    config = load_config(config_path)
    logger = setup_logger()
    
    logger.info(f"Loaded config from: {config_path}")
    
    set_seed(config['runtime']['seed'])
    device = get_device(config['runtime']['device'])
    logger.info(f"Using device: {device}")
    
    print_step(1, "Load market data from CSV")
    df_raw = load_data_from_csv(config)
    logger.info(f"Data shape: {df_raw.shape}")
    
    print_step(2, "Build features and labels")
    df_features = add_indicators(df_raw, config)
    logger.info(f"Features shape: {df_features.shape}")
    
    print_step(3, "Create sliding windows")
    data_dict = build_dataset(df_features, config)
    logger.info(f"Total samples: {len(data_dict['X_ts'])}")
    
    print_step(4, "Encode GADF images")
    X_img = encode_gadf(data_dict['X_ts'], config)
    logger.info(f"GADF images shape: {X_img.shape}")
    
    print_step(5, "Train SimpleCNN")
    
    print_step(6, "Evaluate on test set")
    
    print_step(7, "Generate LRP heatmaps")
    
    print_step(8, "Run spectral clustering")
    
    print_step(9, "Save final outputs")
    
    logger.info("All steps completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GADF-CNN-LRP Stock Prediction Pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/spy_daily.yaml",
        help="Path to configuration file"
    )
    args = parser.parse_args()
    
    main(args.config)
