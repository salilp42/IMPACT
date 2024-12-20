"""
Training script for IMPACT model.
"""

import argparse
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from impact.data.loader import IMPACTDataLoader
from impact.models.impact import IMPACTModel
from impact.utils.metrics import compute_metrics

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description='Train IMPACT model')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, required=True,
                      help='Directory containing preprocessed data')
    parser.add_argument('--output_dir', type=str, default='outputs',
                      help='Directory to save model outputs')
    
    # Model arguments
    parser.add_argument('--embed_dim', type=int, default=256,
                      help='Embedding dimension')
    parser.add_argument('--n_heads', type=int, default=4,
                      help='Number of attention heads')
    parser.add_argument('--n_layers', type=int, default=3,
                      help='Number of transformer layers')
    parser.add_argument('--dropout', type=float, default=0.1,
                      help='Dropout rate')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=8,
                      help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4,
                      help='Number of data loading workers')
    parser.add_argument('--lr', type=float, default=1e-4,
                      help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                      help='Weight decay')
    parser.add_argument('--epochs', type=int, default=100,
                      help='Number of epochs')
    parser.add_argument('--patience', type=int, default=10,
                      help='Patience for early stopping')
    parser.add_argument('--device', type=str, default='cuda',
                      help='Device to use (cuda or cpu)')
    
    return parser.parse_args()

def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(train_loader, desc='Training')
    for batch in pbar:
        # Get data
        inputs = {k: v.to(device) for k, v in batch[0].items()}
        labels = batch[1].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        logits, _ = model(inputs)
        loss = criterion(logits, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Track metrics
        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        pbar.set_postfix({'loss': loss.item()})
    
    metrics = compute_metrics(all_labels, all_preds)
    metrics['loss'] = total_loss / len(train_loader)
    
    return metrics

def validate(model, val_loader, criterion, device):
    """Validate model."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in val_loader:
            # Get data
            inputs = {k: v.to(device) for k, v in batch[0].items()}
            labels = batch[1].to(device)
            
            # Forward pass
            logits, _ = model(inputs)
            loss = criterion(logits, labels)
            
            # Track metrics
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    metrics = compute_metrics(all_labels, all_preds)
    metrics['loss'] = total_loss / len(val_loader)
    
    return metrics

def main():
    args = parse_args()
    
    # Set up directories
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / run_name
    run_dir.mkdir()
    
    # Set up logging
    writer = SummaryWriter(run_dir / 'tensorboard')
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')
    
    # Load data
    data_loader = IMPACTDataLoader(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    train_loader, val_loader, test_loader = data_loader.get_dataloaders()
    
    # Create model
    model = IMPACTModel(
        roi_dim=train_loader.dataset.roi_data.size(-1),
        ica_dim=train_loader.dataset.ica_data.size(-1),
        embed_dim=args.embed_dim,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout
    ).to(device)
    
    # Set up training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        verbose=True
    )
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(args.epochs):
        logger.info(f'Epoch {epoch+1}/{args.epochs}')
        
        # Train
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
        logger.info(f'Train metrics: {train_metrics}')
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, device)
        logger.info(f'Validation metrics: {val_metrics}')
        
        # Log metrics
        for name, value in train_metrics.items():
            writer.add_scalar(f'train/{name}', value, epoch)
        for name, value in val_metrics.items():
            writer.add_scalar(f'val/{name}', value, epoch)
        
        # Update learning rate
        scheduler.step(val_metrics['loss'])
        
        # Save best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_metrics['loss'],
                'train_loss': train_metrics['loss']
            }, run_dir / 'best_model.pt')
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= args.patience:
            logger.info('Early stopping triggered')
            break
    
    # Test best model
    logger.info('Loading best model for testing')
    checkpoint = torch.load(run_dir / 'best_model.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    test_metrics = validate(model, test_loader, criterion, device)
    logger.info(f'Test metrics: {test_metrics}')
    
    # Save test metrics
    with open(run_dir / 'test_metrics.txt', 'w') as f:
        for name, value in test_metrics.items():
            f.write(f'{name}: {value}\n')
    
    writer.close()

if __name__ == '__main__':
    main()
