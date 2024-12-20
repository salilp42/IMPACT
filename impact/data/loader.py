"""
IMPACT Data Loader

This module handles loading and preprocessing of data for the IMPACT pipeline.
It includes functionality for:
- Loading preprocessed fMRI data
- Data normalization and standardization
- Creating PyTorch datasets and dataloaders
- Train/validation/test splitting
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional, Union
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IMPACTDataset(Dataset):
    """PyTorch Dataset for IMPACT model."""
    
    def __init__(
        self,
        roi_data: np.ndarray,
        ica_data: np.ndarray,
        conn_data: np.ndarray,
        labels: np.ndarray,
        normalize: bool = True
    ):
        """
        Initialize dataset.
        
        Args:
            roi_data: ROI time series (n_subjects, n_timepoints, n_rois)
            ica_data: ICA time series (n_subjects, n_timepoints, n_components)
            conn_data: Connectivity matrices (n_subjects, n_windows, n_rois, n_rois)
            labels: Subject labels (n_subjects,)
            normalize: Whether to normalize the data
        """
        self.roi_data = torch.FloatTensor(roi_data)
        self.ica_data = torch.FloatTensor(ica_data)
        self.conn_data = torch.FloatTensor(conn_data)
        self.labels = torch.LongTensor(labels)
        
        if normalize:
            # Normalize each subject's data independently
            for i in range(len(self)):
                # ROI data
                mean = self.roi_data[i].mean(dim=0, keepdim=True)
                std = self.roi_data[i].std(dim=0, keepdim=True)
                self.roi_data[i] = (self.roi_data[i] - mean) / (std + 1e-8)
                
                # ICA data
                mean = self.ica_data[i].mean(dim=0, keepdim=True)
                std = self.ica_data[i].std(dim=0, keepdim=True)
                self.ica_data[i] = (self.ica_data[i] - mean) / (std + 1e-8)
                
                # Connectivity data (already normalized by correlation)
                self.conn_data[i] = torch.clamp(self.conn_data[i], -1, 1)
    
    def __len__(self) -> int:
        return len(self.labels)
    
    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        return {
            'roi': self.roi_data[idx],
            'ica': self.ica_data[idx],
            'conn': self.conn_data[idx]
        }, self.labels[idx]

class IMPACTDataLoader:
    """Data loader for IMPACT model."""
    
    def __init__(
        self,
        data_dir: Union[str, Path],
        batch_size: int = 8,
        num_workers: int = 4,
        test_size: float = 0.2,
        val_size: float = 0.2,
        random_state: int = 42
    ):
        """
        Initialize data loader.
        
        Args:
            data_dir: Directory containing preprocessed data
            batch_size: Batch size for DataLoader
            num_workers: Number of workers for DataLoader
            test_size: Fraction of data to use for testing
            val_size: Fraction of training data to use for validation
            random_state: Random seed for reproducibility
        """
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        
        # Load metadata
        with open(self.data_dir / 'metadata.json', 'r') as f:
            self.metadata = json.load(f)
    
    def load_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load all preprocessed data.
        
        Returns:
            roi_data: ROI time series
            ica_data: ICA time series
            conn_data: Connectivity matrices
            labels: Subject labels
        """
        # Load data from numpy files
        roi_data = np.load(self.data_dir / 'roi_timeseries.npy')
        ica_data = np.load(self.data_dir / 'ica_timeseries.npy')
        conn_data = np.load(self.data_dir / 'connectivity_matrices.npy')
        labels = np.load(self.data_dir / 'labels.npy')
        
        return roi_data, ica_data, conn_data, labels
    
    def create_data_splits(
        self,
        stratify: Optional[np.ndarray] = None
    ) -> Tuple[IMPACTDataset, IMPACTDataset, IMPACTDataset]:
        """
        Create train/validation/test splits.
        
        Args:
            stratify: Labels to use for stratification
            
        Returns:
            train_dataset, val_dataset, test_dataset
        """
        # Load data
        roi_data, ica_data, conn_data, labels = self.load_data()
        
        # Create train/test split
        indices = np.arange(len(labels))
        train_idx, test_idx = train_test_split(
            indices,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=stratify
        )
        
        # Create train/val split
        if stratify is not None:
            stratify = stratify[train_idx]
        train_idx, val_idx = train_test_split(
            train_idx,
            test_size=self.val_size,
            random_state=self.random_state,
            stratify=stratify
        )
        
        # Create datasets
        train_dataset = IMPACTDataset(
            roi_data[train_idx],
            ica_data[train_idx],
            conn_data[train_idx],
            labels[train_idx]
        )
        
        val_dataset = IMPACTDataset(
            roi_data[val_idx],
            ica_data[val_idx],
            conn_data[val_idx],
            labels[val_idx]
        )
        
        test_dataset = IMPACTDataset(
            roi_data[test_idx],
            ica_data[test_idx],
            conn_data[test_idx],
            labels[test_idx]
        )
        
        return train_dataset, val_dataset, test_dataset
    
    def get_dataloaders(
        self,
        stratify: Optional[np.ndarray] = None
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Get PyTorch DataLoaders for train/validation/test sets.
        
        Args:
            stratify: Labels to use for stratification
            
        Returns:
            train_loader, val_loader, test_loader
        """
        train_dataset, val_dataset, test_dataset = self.create_data_splits(stratify)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        return train_loader, val_loader, test_loader
