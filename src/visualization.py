from pathlib import Path
from typing import List, Dict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils import ensure_dir, setup_logger

logger = setup_logger()


def plot_confusion_matrix(cm: np.ndarray, save_path: Path, title: str = "Confusion Matrix"):
    fig, ax = plt.subplots(figsize=(8, 6))
    
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    classes = ['Down (0)', 'Up (1)']
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes,
           yticklabels=classes,
           ylabel='True label',
           xlabel='Predicted label',
           title=title)
    
    ax.set_ylim(len(classes) - 0.5, -0.5)
    
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    fig.tight_layout()
    ensure_dir(str(save_path.parent))
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved confusion matrix to {save_path}")


def plot_training_curves(history_path: Path, save_path: Path):
    df = pd.read_csv(history_path)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1 = axes[0]
    ax1.plot(df['epoch'], df['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax1.plot(df['epoch'], df['val_loss'], 'r-', label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[1]
    ax2.plot(df['epoch'], df['val_accuracy'], 'g-', label='Val Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Validation Accuracy', fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    best_epoch = df.loc[df['val_accuracy'].idxmax(), 'epoch']
    best_acc = df['val_accuracy'].max()
    ax2.axhline(y=best_acc, color='r', linestyle='--', alpha=0.5)
    ax2.axvline(x=best_epoch, color='r', linestyle='--', alpha=0.5)
    ax2.annotate(f'Best: {best_acc:.4f}\nEpoch: {int(best_epoch)}',
                 xy=(best_epoch, best_acc),
                 xytext=(best_epoch + 1, best_acc - 0.05),
                 fontsize=10,
                 arrowprops=dict(arrowstyle='->', color='red'))
    
    plt.tight_layout()
    ensure_dir(str(save_path.parent))
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved training curves to {save_path}")


def visualize_results(config: dict):
    logger.info("Generating visualizations...")
    
    reports_dir = Path('outputs/reports')
    ensure_dir(str(reports_dir))
    
    history_path = reports_dir / 'train_history.csv'
    if history_path.exists():
        plot_training_curves(history_path, reports_dir / 'training_curves.png')
    else:
        logger.warning(f"Training history not found: {history_path}")
    
    logger.info("Visualization completed.")
