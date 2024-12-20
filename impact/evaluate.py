"""
Evaluation script for IMPACT model.
"""

import argparse
import logging
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import roc_curve, auc
from nilearn import plotting

from impact.data.loader import IMPACTDataLoader
from impact.models.impact import IMPACTModel
from impact.utils.metrics import compute_metrics

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate IMPACT model')
    
    parser.add_argument('--model_path', type=str, required=True,
                      help='Path to saved model checkpoint')
    parser.add_argument('--data_dir', type=str, required=True,
                      help='Directory containing test data')
    parser.add_argument('--output_dir', type=str, default='evaluation',
                      help='Directory to save evaluation results')
    parser.add_argument('--batch_size', type=int, default=8,
                      help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4,
                      help='Number of data loading workers')
    parser.add_argument('--device', type=str, default='cuda',
                      help='Device to use (cuda or cpu)')
    
    return parser.parse_args()

def plot_roc_curve(labels, predictions, output_path):
    """Plot ROC curve."""
    fpr, tpr, _ = roc_curve(labels, predictions)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(output_path)
    plt.close()

def plot_attention_weights(attention_weights, roi_labels, output_path):
    """Plot attention weights for interpretability."""
    # Average attention weights across heads and layers
    avg_attention = np.mean([w.cpu().numpy() for w in attention_weights], axis=0)
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(avg_attention, xticklabels=roi_labels, yticklabels=roi_labels,
                cmap='viridis', center=0)
    plt.title('Average Attention Weights')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_brain_importance(importance_scores, atlas, output_path):
    """Plot importance scores on brain surface."""
    # Create brain map
    display = plotting.plot_glass_brain(None)
    display.add_overlay(importance_scores, colorbar=True)
    display.savefig(output_path)
    display.close()

def evaluate_model(model, data_loader, device, output_dir):
    """Comprehensive model evaluation."""
    model.eval()
    all_preds = []
    all_labels = []
    all_attention_weights = {'roi': [], 'ica': [], 'conn': []}
    
    with torch.no_grad():
        for batch in data_loader:
            # Get data
            inputs = {k: v.to(device) for k, v in batch[0].items()}
            labels = batch[1].to(device)
            
            # Forward pass
            logits, attention = model(inputs)
            
            # Store predictions and attention weights
            preds = torch.softmax(logits, dim=1)[:, 1]  # Probability of positive class
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            for modality in attention:
                all_attention_weights[modality].extend(attention[modality])
    
    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Compute metrics
    metrics = compute_metrics(all_labels, (all_preds > 0.5).astype(int))
    logger.info(f'Test metrics: {metrics}')
    
    # Save metrics
    with open(output_dir / 'metrics.txt', 'w') as f:
        for name, value in metrics.items():
            f.write(f'{name}: {value}\n')
    
    # Plot ROC curve
    plot_roc_curve(all_labels, all_preds, output_dir / 'roc_curve.png')
    
    # Plot attention weights
    for modality in all_attention_weights:
        plot_attention_weights(
            all_attention_weights[modality],
            data_loader.dataset.roi_labels,
            output_dir / f'{modality}_attention.png'
        )
    
    return metrics, all_attention_weights

def main():
    args = parse_args()
    
    # Set up directories
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')
    
    # Load data
    data_loader = IMPACTDataLoader(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    _, _, test_loader = data_loader.get_dataloaders()
    
    # Load model
    checkpoint = torch.load(args.model_path)
    model = IMPACTModel(
        roi_dim=test_loader.dataset.roi_data.size(-1),
        ica_dim=test_loader.dataset.ica_data.size(-1)
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Evaluate model
    metrics, attention_weights = evaluate_model(
        model,
        test_loader,
        device,
        output_dir
    )
    
    logger.info('Evaluation complete. Results saved to {}'.format(output_dir))

if __name__ == '__main__':
    main()
