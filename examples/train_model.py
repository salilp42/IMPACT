import argparse
from pathlib import Path
import numpy as np
from torch.utils.data import DataLoader, random_split

from impact.models.impact import IMPACTModel
from impact.training.trainer import TrainingConfig, IMPACTDataset, IMPACTTrainer
from impact.data.loaders import TaoWuLoader, NeuroconLoader
from impact.utils.visualization import (
    plot_attention_weights,
    plot_roi_importance,
    plot_network_analysis,
    plot_gradcam_analysis,
    plot_statistical_comparison
)

def load_data(data_dir: Path):
    """Load processed data."""
    # Load ROI time series
    pd_roi = np.load(data_dir / 'pd_roi_timeseries.npy')
    hc_roi = np.load(data_dir / 'hc_roi_timeseries.npy')
    
    # Load ICA time series
    pd_ica = np.load(data_dir / 'pd_ica_timeseries.npy')
    hc_ica = np.load(data_dir / 'hc_ica_timeseries.npy')
    
    # Load correlation matrices
    pd_corr = np.load(data_dir / 'pd_correlations.npy')
    hc_corr = np.load(data_dir / 'hc_correlations.npy')
    
    # Combine data
    roi_data = np.concatenate([pd_roi, hc_roi])
    ica_data = np.concatenate([pd_ica, hc_ica])
    corr_data = np.concatenate([pd_corr, hc_corr])
    
    # Create labels (1 for PD, 0 for HC)
    labels = np.concatenate([
        np.ones(len(pd_roi)),
        np.zeros(len(hc_roi))
    ])
    
    return roi_data, ica_data, corr_data, labels

def main(args):
    # Process dataset if needed
    if args.process_data:
        if args.dataset == 'taowu':
            loader = TaoWuLoader(args.data_dir, args.processed_dir)
        else:
            loader = NeuroconLoader(args.data_dir, args.processed_dir)
        loader.process_dataset()
    
    # Load processed data
    roi_data, ica_data, corr_data, labels = load_data(args.processed_dir)
    
    # Create dataset
    dataset = IMPACTDataset(roi_data, ica_data, corr_data, labels)
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )
    
    # Initialize model
    model = IMPACTModel(
        roi_dim=roi_data.shape[2],
        ica_dim=ica_data.shape[2],
        roi_hidden_dim=args.hidden_dim,
        ica_hidden_dim=args.hidden_dim,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout
    )
    
    # Set up training
    config = TrainingConfig(
        roi_hidden_dim=args.hidden_dim,
        ica_hidden_dim=args.hidden_dim,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        results_dir=args.output_dir
    )
    
    trainer = IMPACTTrainer(model, config)
    
    # Train model
    history = trainer.train(train_loader, val_loader)
    trainer.save_history(history)
    
    # Generate visualizations
    if args.visualize:
        # Load best model
        model.load_state_dict(torch.load(args.output_dir / 'best_model.pt'))
        model.eval()
        
        # Get sample batch
        batch = next(iter(val_loader))
        with torch.no_grad():
            _, attention = model(
                batch['roi'].to(config.device),
                batch['ica'].to(config.device),
                batch['corr'].to(config.device)
            )
        
        # Plot attention weights
        plot_attention_weights(
            attention[-1],  # Use last layer's attention
            args.output_dir / 'attention_weights.png'
        )
        
        # Plot ROI importance
        importance = attention[-1].mean(dim=(0,1)).cpu().numpy()
        plot_roi_importance(
            importance,
            args.roi_labels if hasattr(args, 'roi_labels') else None,
            args.output_dir / 'roi_importance.png'
        )
        
        # Plot network analysis
        plot_network_analysis(
            batch['corr'][0].cpu().numpy(),
            args.roi_labels if hasattr(args, 'roi_labels') else None,
            args.output_dir / 'network_analysis.png'
        )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train IMPACT model')
    
    # Data arguments
    parser.add_argument('--dataset', choices=['taowu', 'neurocon'], required=True,
                       help='Dataset to use')
    parser.add_argument('--data_dir', type=Path, required=True,
                       help='Path to raw data directory')
    parser.add_argument('--processed_dir', type=Path, required=True,
                       help='Path to processed data directory')
    parser.add_argument('--process_data', action='store_true',
                       help='Whether to process the raw data')
    
    # Model arguments
    parser.add_argument('--hidden_dim', type=int, default=256,
                       help='Hidden dimension size')
    parser.add_argument('--n_heads', type=int, default=8,
                       help='Number of attention heads')
    parser.add_argument('--n_layers', type=int, default=2,
                       help='Number of transformer layers')
    parser.add_argument('--dropout', type=float, default=0.12,
                       help='Dropout rate')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=250,
                       help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=6e-4,
                       help='Learning rate')
    
    # Output arguments
    parser.add_argument('--output_dir', type=Path, required=True,
                       help='Path to output directory')
    parser.add_argument('--visualize', action='store_true',
                       help='Whether to generate visualizations')
    
    args = parser.parse_args()
    main(args) 