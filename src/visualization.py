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
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    ax1 = axes[0, 0]
    ax1.plot(df['epoch'], df['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax1.plot(df['epoch'], df['val_loss'], 'r-', label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[0, 1]
    if 'train_acc' in df.columns:
        ax2.plot(df['epoch'], df['train_acc'], 'b-', label='Train Acc', linewidth=2)
    ax2.plot(df['epoch'], df['val_acc'], 'g-', label='Val Acc', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Training and Validation Accuracy', fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    best_acc_idx = df['val_acc'].idxmax()
    best_epoch = df.loc[best_acc_idx, 'epoch']
    best_acc = df['val_acc'].max()
    ax2.axhline(y=best_acc, color='r', linestyle='--', alpha=0.5)
    ax2.axvline(x=best_epoch, color='r', linestyle='--', alpha=0.5)
    ax2.annotate(f'Best: {best_acc:.4f}\nEpoch: {int(best_epoch)}',
                 xy=(best_epoch, best_acc),
                 xytext=(best_epoch + 1, best_acc - 0.05),
                 fontsize=10,
                 arrowprops=dict(arrowstyle='->', color='red'))
    
    ax3 = axes[1, 0]
    if 'val_f1' in df.columns:
        ax3.plot(df['epoch'], df['val_f1'], 'm-', label='Val F1', linewidth=2)
    if 'val_precision' in df.columns:
        ax3.plot(df['epoch'], df['val_precision'], 'c-', label='Val Precision', linewidth=2)
    if 'val_recall' in df.columns:
        ax3.plot(df['epoch'], df['val_recall'], 'y-', label='Val Recall', linewidth=2)
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('Score', fontsize=12)
    ax3.set_title('Validation Metrics', fontsize=14)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    ax4 = axes[1, 1]
    if 'val_auc' in df.columns:
        val_auc = df['val_auc'].dropna()
        if len(val_auc) > 0:
            ax4.plot(df['epoch'], df['val_auc'], 'orange', label='Val AUC', linewidth=2)
            ax4.set_xlabel('Epoch', fontsize=12)
            ax4.set_ylabel('AUC', fontsize=12)
            ax4.set_title('Validation AUC', fontsize=14)
            ax4.legend(fontsize=10)
            ax4.grid(True, alpha=0.3)
            
            best_auc_idx = df['val_auc'].idxmax()
            best_auc_epoch = df.loc[best_auc_idx, 'epoch']
            best_auc = df['val_auc'].max()
            ax4.axhline(y=best_auc, color='r', linestyle='--', alpha=0.5)
            ax4.axvline(x=best_auc_epoch, color='r', linestyle='--', alpha=0.5)
            ax4.annotate(f'Best: {best_auc:.4f}\nEpoch: {int(best_auc_epoch)}',
                         xy=(best_auc_epoch, best_auc),
                         xytext=(best_auc_epoch + 1, best_auc - 0.05),
                         fontsize=10,
                         arrowprops=dict(arrowstyle='->', color='red'))
        else:
            ax4.text(0.5, 0.5, 'AUC not available\n(single class predictions)',
                     ha='center', va='center', fontsize=12, transform=ax4.transAxes)
            ax4.set_title('Validation AUC', fontsize=14)
    else:
        ax4.text(0.5, 0.5, 'AUC not available',
                 ha='center', va='center', fontsize=12, transform=ax4.transAxes)
        ax4.set_title('Validation AUC', fontsize=14)
    
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
