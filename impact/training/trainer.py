import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
from tqdm import tqdm
import logging
import json
from dataclasses import dataclass

@dataclass
class TrainingConfig:
    """Configuration for training the IMPACT model."""
    
    # Model parameters
    roi_hidden_dim: int = 256
    ica_hidden_dim: int = 256
    n_heads: int = 8
    n_layers: int = 2
    dropout: float = 0.12
    
    # Training parameters
    n_epochs: int = 250
    batch_size: int = 16
    learning_rate: float = 6e-4
    weight_decay: float = 2e-4
    gradient_clip: float = 4.0
    label_smoothing: float = 0.1
    warmup_steps: int = 300
    early_stopping_patience: int = 35
    
    # Data parameters
    data_dir: Path = Path('data')
    results_dir: Path = Path('results')
    
    def __post_init__(self):
        """Set up directories and device."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.data_dir = Path(self.data_dir)
        self.results_dir = Path(self.results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

class IMPACTDataset(Dataset):
    """Dataset class for IMPACT model."""
    
    def __init__(self, roi_data: np.ndarray, ica_data: np.ndarray, 
                 corr_data: np.ndarray, labels: np.ndarray):
        """
        Initialize dataset.
        
        Args:
            roi_data: ROI time series [n_subjects, n_timepoints, n_rois]
            ica_data: ICA time series [n_subjects, n_timepoints, n_components]
            corr_data: Correlation matrices [n_subjects, n_rois, n_rois]
            labels: Subject labels [n_subjects]
        """
        self.roi_data = torch.FloatTensor(roi_data)
        self.ica_data = torch.FloatTensor(ica_data)
        self.corr_data = torch.FloatTensor(corr_data)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return {
            'roi': self.roi_data[idx],
            'ica': self.ica_data[idx],
            'corr': self.corr_data[idx],
            'label': self.labels[idx]
        }

class IMPACTTrainer:
    """Trainer class for IMPACT model."""
    
    def __init__(self, model: nn.Module, config: TrainingConfig):
        """
        Initialize trainer.
        
        Args:
            model: IMPACT model instance
            config: Training configuration
        """
        self.model = model.to(config.device)
        self.config = config
        
        # Set up optimizer and scheduler
        self.optimizer = AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        self.criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
        
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        
        for batch in tqdm(train_loader, desc="Training"):
            self.optimizer.zero_grad()
            
            # Move data to device
            roi = batch['roi'].to(self.config.device)
            ica = batch['ica'].to(self.config.device)
            corr = batch['corr'].to(self.config.device)
            labels = batch['label'].to(self.config.device)
            
            # Forward pass
            logits, _ = self.model(roi, ica, corr)
            loss = self.criterion(logits, labels)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.gradient_clip
            )
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    @torch.no_grad()
    def evaluate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Evaluate model."""
        self.model.eval()
        total_loss = 0
        predictions = []
        true_labels = []
        
        for batch in val_loader:
            # Move data to device
            roi = batch['roi'].to(self.config.device)
            ica = batch['ica'].to(self.config.device)
            corr = batch['corr'].to(self.config.device)
            labels = batch['label'].to(self.config.device)
            
            # Forward pass
            logits, _ = self.model(roi, ica, corr)
            loss = self.criterion(logits, labels)
            
            total_loss += loss.item()
            predictions.extend(logits.argmax(dim=1).cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
        
        accuracy = np.mean(np.array(predictions) == np.array(true_labels))
        return total_loss / len(val_loader), accuracy
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> Dict:
        """
        Train the model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
        
        Returns:
            dict: Training history
        """
        best_val_loss = float('inf')
        patience_counter = 0
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': []
        }
        
        for epoch in range(self.config.n_epochs):
            # Train
            train_loss = self.train_epoch(train_loader)
            
            # Evaluate
            val_loss, val_accuracy = self.evaluate(val_loader)
            
            # Log progress
            self.logger.info(
                f"Epoch {epoch+1}/{self.config.n_epochs} - "
                f"Train Loss: {train_loss:.4f} - "
                f"Val Loss: {val_loss:.4f} - "
                f"Val Accuracy: {val_accuracy:.4f}"
            )
            
            # Update history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_accuracy'].append(val_accuracy)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(
                    self.model.state_dict(),
                    self.config.results_dir / 'best_model.pt'
                )
            else:
                patience_counter += 1
                if patience_counter >= self.config.early_stopping_patience:
                    self.logger.info("Early stopping triggered")
                    break
        
        return history
    
    def save_history(self, history: Dict, filename: str = 'training_history.json'):
        """Save training history."""
        path = self.config.results_dir / filename
        with open(path, 'w') as f:
            json.dump(
                {k: [float(v) for v in vals] for k, vals in history.items()},
                f,
                indent=2
            ) 