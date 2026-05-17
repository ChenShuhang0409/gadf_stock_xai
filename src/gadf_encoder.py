from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pyts.image import GramianAngularField
from tqdm import tqdm

from src.utils import ensure_dir, setup_logger

logger = setup_logger()


def encode_gadf(X_ts: np.ndarray, config: dict) -> np.ndarray:
    logger.info("Encoding time series to GADF images...")
    
    image_size = config['dataset']['image_size']
    n_samples, n_features, window_size = X_ts.shape
    
    logger.info(f"Processing {n_samples} samples with {n_features} features...")
    logger.info(f"Image size: {image_size}x{image_size}")
    
    gadf = GramianAngularField(image_size=image_size, method='difference')
    
    X_img_list = []
    
    for feature_idx in tqdm(range(n_features), desc="Encoding features"):
        logger.info(f"Encoding feature {feature_idx + 1}/{n_features}...")
        
        X_feature = X_ts[:, feature_idx, :]
        
        X_img_feature = gadf.fit_transform(X_feature)
        
        X_img_feature = X_img_feature[:, np.newaxis, :, :]
        
        X_img_list.append(X_img_feature)
    
    X_img = np.concatenate(X_img_list, axis=1)
    
    logger.info(f"X_img shape: {X_img.shape}")
    
    save_gadf_examples(X_img, config)
    
    processed_dir = Path('data/processed')
    np.save(processed_dir / 'X_img.npy', X_img)
    logger.info(f"Saved X_img to {processed_dir / 'X_img.npy'}")
    
    return X_img


def save_gadf_examples(X_img: np.ndarray, config: dict, n_examples: int = 5):
    logger.info(f"Saving {n_examples} GADF example images...")
    
    output_dir = Path('outputs/heatmaps/gadf_examples')
    ensure_dir(str(output_dir))
    
    n_samples = X_img.shape[0]
    n_features = X_img.shape[1]
    
    indices = np.linspace(0, n_samples - 1, n_examples, dtype=int)
    
    feature_names = config['features']['names']
    
    for i, idx in enumerate(indices):
        fig, axes = plt.subplots(1, n_features, figsize=(4 * n_features, 4))
        
        if n_features == 1:
            axes = [axes]
        
        for feature_idx in range(n_features):
            ax = axes[feature_idx]
            img = X_img[idx, feature_idx, :, :]
            
            im = ax.imshow(img, cmap='rainbow', aspect='auto')
            ax.set_title(f'{feature_names[feature_idx]}', fontsize=12)
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        plt.suptitle(f'Sample {idx}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        save_path = output_dir / f'gadf_sample_{i + 1}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved {save_path}")
    
    logger.info(f"Saved {n_examples} GADF examples to {output_dir}")
