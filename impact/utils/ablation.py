#!/usr/bin/env python3
"""
Ablation Analysis for IMPACT Model

This script performs ablation studies on the IMPACT model to evaluate the 
contribution of different input modalities (ROI, ICA, and DFNC/Connectivity)
to the model's performance on Parkinson's Disease classification.

Configurations tested:
1. ROI only
2. ICA only
3. Connectivity only
4. ROI + ICA
5. ROI + Connectivity
6. ICA + Connectivity
7. All modalities (Full model)
"""

import os
import json
import math
import time
import logging
import warnings
import argparse
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

from sklearn.metrics import (roc_auc_score, accuracy_score, precision_score,
                           recall_score, f1_score, roc_curve, confusion_matrix,
                           balanced_accuracy_score, matthews_corrcoef)
from sklearn.model_selection import StratifiedKFold
from scipy.stats import mannwhitneyu, ttest_rel, wilcoxon
import matplotlib.gridspec as gridspec

# Configure logging and styling
warnings.filterwarnings('ignore')

def setup_logging(log_level_str, log_file=None):
    numeric_level = getattr(logging, log_level_str.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {log_level_str}')

    log_format = '%(asctime)s - %(levelname)s - %(name)s - %(message)s'
    
    # Configure root logger
    handlers = [logging.StreamHandler()] # Log to console by default
    if log_file:
        handlers.append(logging.FileHandler(log_file, mode='a')) # Append to log file if specified
        
    logging.basicConfig(level=numeric_level, format=log_format, handlers=handlers)
    
    # Set level for specific libraries if needed (e.g., suppress verbose logs)
    # logging.getLogger('matplotlib').setLevel(logging.WARNING)
    # logging.getLogger('nilearn').setLevel(logging.WARNING)
    
    logging.info(f"Logging configured with level: {log_level_str}")
    if log_file:
        logging.info(f"Logging additionally to file: {log_file}")

# Set plot styles
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['figure.titlesize'] = 18
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

# For reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")

class Config:
    """Configuration class for ablation study parameters."""
    
    def __init__(
        self,
        dataset_name: str = "neurocon",  # "neurocon" or "taowu"
        data_dir: Optional[Path] = None,
        results_dir: Optional[Path] = None,
        n_epochs: int = 150,
        early_stopping_patience: int = 15,
        batch_size: int = 16,
        learning_rate: float = 6e-4,
        roi_hidden_dim: int = 256,
        ica_hidden_dim: int = 256,
        n_heads: int = 8,
        n_layers: int = 2,
        dropout: float = 0.12,
        weight_decay: float = 2e-4,
        gradient_clip: float = 4.0,
        predefined_folds: int = 5,
        ablation_config: str = "full",  # Options: "roi", "ica", "conn", "roi+ica", "roi+conn", "ica+conn", "full"
    ):
        self.dataset_name = dataset_name
        self.ablation_config = ablation_config
        
        # Set data directory based on dataset
        if data_dir is None:
            if dataset_name == "taowu":
                self.data_dir = Path("taowu/processed_tao_wu/dl_ready")
            else:  # neurocon
                self.data_dir = Path("processed/dl_ready")
        else:
            self.data_dir = data_dir
            
        # Set results directory
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        if results_dir is None:
            self.results_dir = Path(f"ablation_results_{timestamp}")
        else:
            self.results_dir = results_dir
            
        # Training parameters
        self.n_epochs = n_epochs
        self.early_stopping_patience = early_stopping_patience
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.gradient_clip = gradient_clip
        self.predefined_folds = predefined_folds
        
        # Model parameters
        self.roi_hidden_dim = roi_hidden_dim
        self.ica_hidden_dim = ica_hidden_dim
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.dropout = dropout
        
        # Will be set later when loading data
        self.n_rois = None
        self.n_ica = None
        self.roi_labels = None
        
        # Setup directories and device
        self.setup_directories()
        self.device = device
        
    def setup_directories(self):
        """Create necessary directories for saving results."""
        suffix = f"{self.dataset_name}_{self.ablation_config}"
        self.run_dir = self.results_dir / f"{suffix}_{self.results_dir.name}"
        
        dirs = ['figures', 'metrics', 'models', 'logs']
        for d in dirs:
            (self.run_dir / d).mkdir(parents=True, exist_ok=True)
            
        # Save config
        self.save_config()
        
    def save_config(self):
        """Save configuration to JSON file."""
        config_dict = {k: str(v) if isinstance(v, Path) else v 
                      for k, v in self.__dict__.items() 
                      if not k.startswith('__') and not callable(v) and k != 'device'}
        
        with open(self.run_dir / "config.json", 'w') as f:
            json.dump(config_dict, f, indent=4)
            
    def __str__(self):
        """String representation of config."""
        return f"Dataset: {self.dataset_name}, Ablation: {self.ablation_config}"

##############################################################################
# Data Loading and Processing
##############################################################################

def load_data(config: Config) -> Dict:
    """
    Load ROI/ICA data for HC/PD based on the specified dataset.
    Return a dict with:
      "roi_data"      [N, T, #ROI]
      "ica_data"      [N, T, #ICA]
      "corr_data"     [N, W, ROI, ROI]
      "labels"        [N] -> 0 or 1
      "subject_ids"   [N] -> string IDs
    """
    if config.dataset_name == "taowu":
        # Use the correct path for Tao Wu dataset
        data_dir = Path("taowu/processed_tao_wu/dl_ready")
    else:
        # Use the configured data directory for other datasets
        data_dir = Path(config.data_dir)
    
    logging.info(f"Loading data from: {data_dir}")
    
    # Load HC
    hc_roi = np.load(data_dir / "hc_roi_timeseries.npy")      # [N_HC, T, ROI]
    hc_ica = np.load(data_dir / "hc_ica_timeseries.npy")      # [N_HC, T, ICA]
    
    # Load PD
    pd_roi = np.load(data_dir / "pd_roi_timeseries.npy")      # [N_PD, T, ROI]
    pd_ica = np.load(data_dir / "pd_ica_timeseries.npy")      # [N_PD, T, ICA]
    
    # Check for windowed ROI data (for correlation calculation)
    hc_wroi = None
    pd_wroi = None
    try:
        hc_wroi = np.load(data_dir / "hc_windowed_roi.npy")   # [N_HC, W, Tw, ROI]
        pd_wroi = np.load(data_dir / "pd_windowed_roi.npy")   # [N_PD, W, Tw, ROI]
        logging.info(f"Loaded windowed ROI data for {config.dataset_name}")
    except:
        logging.warning(f"No windowed ROI data found for {config.dataset_name}, will compute on the fly")
    
    # Load subject IDs if available
    try:
        metadata_file = "metadata.json" if config.dataset_name == "taowu" else "subject_ids.json"
        with open(data_dir / metadata_file, 'r') as f:
            metadata = json.load(f)
            
            if config.dataset_name == "taowu":
                hc_ids = metadata.get('hc_subjects', [])
                pd_ids = metadata.get('pd_subjects', [])
                config.roi_labels = metadata.get('roi_labels', [])
            else:  # neurocon
                hc_ids = metadata.get('hc_subjects', [])
                pd_ids = metadata.get('pd_subjects', [])
    except:
        logging.warning(f"No metadata file found for {config.dataset_name}, using generic IDs")
        hc_ids = [f"HC_{i}" for i in range(len(hc_roi))]
        pd_ids = [f"PD_{i}" for i in range(len(pd_roi))]
    
    # Handle ROI dimension mismatch between HC and PD groups
    # This can happen in the Tao Wu dataset where HC and PD might have different number of ROIs
    hc_roi_dim = hc_roi.shape[2]
    pd_roi_dim = pd_roi.shape[2]
    
    if hc_roi_dim != pd_roi_dim:
        logging.warning(f"ROI dimension mismatch: HC has {hc_roi_dim} ROIs, PD has {pd_roi_dim} ROIs")
        
        # Use the minimum number of ROIs from both groups
        min_roi_dim = min(hc_roi_dim, pd_roi_dim)
        logging.info(f"Using the first {min_roi_dim} ROIs for both groups")
        
        hc_roi = hc_roi[:, :, :min_roi_dim]
        pd_roi = pd_roi[:, :, :min_roi_dim]
        
        # Also adjust the windowed ROI data if available
        if hc_wroi is not None and pd_wroi is not None:
            hc_wroi = hc_wroi[:, :, :, :min_roi_dim]
            pd_wroi = pd_wroi[:, :, :, :min_roi_dim]
    
    # Calculate correlation matrices
    if hc_wroi is not None and pd_wroi is not None:
        # Use windowed ROI data
        hc_corr = calculate_correlation_matrices(hc_wroi)
        pd_corr = calculate_correlation_matrices(pd_wroi)
    else:
        # Calculate from full time series
        hc_corr = calculate_correlation_matrices(hc_roi)
        pd_corr = calculate_correlation_matrices(pd_roi)
    
    # Store dimensions (after potential adjustment)
    config.n_rois = hc_roi.shape[2]  # Now this will be the same for both groups
    config.n_ica = hc_ica.shape[2]
    
    # Concatenate
    roi_data = np.concatenate([hc_roi, pd_roi], axis=0)
    ica_data = np.concatenate([hc_ica, pd_ica], axis=0)
    corr_data = np.concatenate([hc_corr, pd_corr], axis=0)
    
    labels = np.concatenate([
        np.zeros(len(hc_roi), dtype=np.int64),
        np.ones(len(pd_roi), dtype=np.int64)
    ])
    subject_ids = np.array(hc_ids + pd_ids)
    
    # Check for NaN/Inf in ICA data
    if np.any(np.isnan(ica_data)) or np.any(np.isinf(ica_data)):
        logging.warning("NaN/Inf detected in ICA data! Attempting to replace with 0.")
        ica_data = np.nan_to_num(ica_data, nan=0.0, posinf=0.0, neginf=0.0)
        
    # Log detailed statistics about ICA data (scale, variance, etc.)
    logging.warning(f"ICA data statistics - Shape: {ica_data.shape}")
    logging.warning(f"  Min: {np.min(ica_data):.6f}, Max: {np.max(ica_data):.6f}")
    logging.warning(f"  Mean: {np.mean(ica_data):.6f}, Std: {np.std(ica_data):.6f}")
    
    # Special preprocessing for ICA data to improve numerical stability
    if "ica" in config.ablation_config:
        logging.warning("Applying special ICA preprocessing for numerical stability")
        
        # Z-score normalization across each ICA component
        ica_mean = np.mean(ica_data, axis=(0, 1), keepdims=True)
        ica_std = np.std(ica_data, axis=(0, 1), keepdims=True) + 1e-8  # Add small epsilon to avoid div by zero
        ica_data = (ica_data - ica_mean) / ica_std
        
        # After normalization, log statistics again
        logging.warning(f"ICA after normalization - Min: {np.min(ica_data):.6f}, Max: {np.max(ica_data):.6f}")
        logging.warning(f"  Mean: {np.mean(ica_data):.6f}, Std: {np.std(ica_data):.6f}")
        
        # Check if we still have any extreme values
        if np.max(np.abs(ica_data)) > 10:
            logging.warning("ICA data still has extreme values after normalization. Clipping...")
            ica_data = np.clip(ica_data, -5, 5)  # Clip to reasonable range
    
    logging.info(f"Loaded {config.dataset_name.capitalize()} data:")
    logging.info(f"  ROI: {roi_data.shape}, ICA: {ica_data.shape}, Corr: {corr_data.shape}")
    logging.info(f"  Subjects: HC={len(hc_roi)}, PD={len(pd_roi)}, Total={len(labels)}")
    
    return {
        "roi_data": roi_data,
        "ica_data": ica_data,
        "corr_data": corr_data,
        "labels": labels,
        "subject_ids": subject_ids
    }

def calculate_correlation_matrices(roi_data: np.ndarray) -> np.ndarray:
    """
    Calculate correlation matrices for each subject based on ROI data.
    
    Args:
        roi_data: Either regular ROI time series [N, T, R] or
                 windowed ROI data [N, W, Tw, R]
        
    Returns:
        correlation_matrices: Correlation matrices [N, W, R, R] where W is the number of windows
    """
    # Check input shape to determine if it's windowed or not
    if len(roi_data.shape) == 4:
        # Already windowed: [N, W, Tw, R]
        n_subjects, n_windows, window_size, n_rois = roi_data.shape
        correlation_matrices = np.zeros((n_subjects, n_windows, n_rois, n_rois))
        
        # Calculate correlation matrix for each subject and window
        for s in range(n_subjects):
            for w in range(n_windows):
                # Extract window data [Tw, R]
                window_data = roi_data[s, w]
                
                # Calculate correlation matrix
                correlation_matrices[s, w] = np.corrcoef(window_data.T)
                
                # Handle NaN values
                correlation_matrices[s, w] = np.nan_to_num(correlation_matrices[s, w])
    else:
        # Regular time series: [N, T, R]
        n_subjects, n_timepoints, n_rois = roi_data.shape
        window_size = 30  # Fixed window size
        stride = 15       # Fixed stride
        
        # Calculate number of windows
        n_windows = (n_timepoints - window_size) // stride + 1
        correlation_matrices = np.zeros((n_subjects, n_windows, n_rois, n_rois))
        
        # Calculate correlation matrix for each subject and window
        for s in range(n_subjects):
            for w in range(n_windows):
                # Extract window data [Tw, R]
                window_start = w * stride
                window_end = window_start + window_size
                window_data = roi_data[s, window_start:window_end]
                
                # Calculate correlation matrix
                correlation_matrices[s, w] = np.corrcoef(window_data.T)
                
                # Handle NaN values
                correlation_matrices[s, w] = np.nan_to_num(correlation_matrices[s, w])
    
    return correlation_matrices

class MultiModalDataset(Dataset):
    """Dataset wrapper for multimodal neuroimaging data."""
    
    def __init__(
        self, 
        data_dict: Dict, 
        indices: np.ndarray,
        config: Config,
        train_mode: bool = True
    ):
        """
        Initialize dataset.
        
        Args:
            data_dict: Dictionary containing data
            indices: Indices to use
            config: Configuration object
            train_mode: Whether this is for training
        """
        self.indices = indices
        self.config = config
        self.ablation_config = config.ablation_config
        self.train_mode = train_mode
        
        # Extract data for selected indices
        self.roi_data = data_dict["roi_data"][indices]
        self.ica_data = data_dict["ica_data"][indices]
        self.corr_data = data_dict["corr_data"][indices]
        self.labels = data_dict["labels"][indices]
        
        # Convert to tensors
        self.roi_data = torch.FloatTensor(self.roi_data)
        self.ica_data = torch.FloatTensor(self.ica_data)
        self.corr_data = torch.FloatTensor(self.corr_data)
        self.labels = torch.LongTensor(self.labels)
        
    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get item by index.
        
        Returns a dictionary containing only the data modalities
        required by the current ablation configuration.
        """
        sample = {
            "label": self.labels[idx]
        }
        
        # Include data based on ablation configuration
        if "roi" in self.ablation_config or self.ablation_config == "full":
            sample["roi_data"] = self.roi_data[idx]
            
        if "ica" in self.ablation_config or self.ablation_config == "full":
            sample["ica_data"] = self.ica_data[idx]
            
        if "conn" in self.ablation_config or self.ablation_config == "full":
            sample["corr_data"] = self.corr_data[idx]
        
        return sample

##############################################################################
# Model Components
##############################################################################

class DynamicTemporalBlock(nn.Module):
    """Multi-scale temporal convolution block with dynamic gating."""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_sizes: list = [3, 5, 7]):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels, out_channels, k, padding=k//2) 
            for k in kernel_sizes
        ])
        self.gate = nn.Sequential(
            nn.Linear(out_channels * len(kernel_sizes), out_channels),
            nn.Sigmoid()
        )
        self.norm = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor [B, T, C]
            
        Returns:
            Output tensor [B, T, C']
        """
        # x: [B, T, C]
        x_conv = x.transpose(1, 2)  # [B, C, T]
        conv_outputs = []
        
        for conv in self.convs:
            conv_out = conv(x_conv)  # [B, C', T]
            conv_outputs.append(conv_out.transpose(1, 2))  # [B, T, C']
        
        multi_scale = torch.cat(conv_outputs, dim=-1)  # [B, T, C'*3]
        
        # Apply gate with proper initialization and ensure no NaN values
        gate_weights = torch.clamp(self.gate(multi_scale), min=1e-6, max=1-1e-6)  # [B, T, C']
        
        output = torch.zeros_like(conv_outputs[0])  # [B, T, C']
        for conv_out in conv_outputs:
            output = output + gate_weights * conv_out
            
        return self.dropout(self.norm(output))

class CorrelationEncoder(nn.Module):
    """Encodes correlation matrices using 2D convolutions."""
    
    def __init__(self, n_rois: int, hidden_dim: int):
        super().__init__()
        self.n_rois = n_rois

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(32)

        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )

        self.proj = nn.Sequential(
            nn.Linear(32, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights to prevent vanishing/exploding gradients."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Correlation matrices [B, W, R, R]
            
        Returns:
            Encoded features [B, W, H]
        """
        batch_size, n_windows = x.size(0), x.size(1)
        
        # Process each window separately
        window_features = []
        for w in range(n_windows):
            # Extract window and add channel dimension
            window = x[:, w].unsqueeze(1)  # [B, 1, R, R]
            
            # Apply convolutions with batch norm
            out = F.relu(self.bn1(self.conv1(window)))
            out = F.relu(self.bn2(self.conv2(out)))
            out = F.relu(self.bn3(self.conv3(out)))
            
            # Global average pooling
            out = self.gap(out)  # [B, 32]
            
            # Project to hidden dimension
            out = self.proj(out)  # [B, H]
            
            window_features.append(out)
        
        # Stack window features
        return torch.stack(window_features, dim=1)  # [B, W, H]

class CustomMultiheadAttention(nn.Module):
    """Custom multi-head attention with layer normalization."""
    
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.q_norm = nn.LayerNorm(embed_dim)
        self.k_norm = nn.LayerNorm(embed_dim)
        self.v_norm = nn.LayerNorm(embed_dim)
        self.out_norm = nn.LayerNorm(embed_dim)
        
        self.last_attn_weights = None
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, 
                need_weights: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            query: Query tensor [B, T, E]
            key: Key tensor [B, S, E]
            value: Value tensor [B, S, E]
            need_weights: Whether to return attention weights
            
        Returns:
            Output tensor [B, T, E] and attention weights [B, H, T, S]
        """
        batch_size, tgt_len, embed_dim = query.size()
        src_len = key.size(1)
        
        # Apply layer norm
        q = self.q_norm(query)
        k = self.k_norm(key)
        v = self.v_norm(value)
        
        # Project to queries, keys, values
        q = self.q_proj(q) * self.scaling  # [B, T, E]
        k = self.k_proj(k)  # [B, S, E]
        v = self.v_proj(v)  # [B, S, E]
        
        # Reshape for multi-head attention
        q = q.view(batch_size, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, T, D]
        k = k.view(batch_size, src_len, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, S, D]
        v = v.view(batch_size, src_len, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, S, D]
        
        # Calculate attention scores
        attn_weights = torch.matmul(q, k.transpose(2, 3))  # [B, H, T, S]
        
        # Apply softmax and dropout
        attn_probs = F.softmax(attn_weights, dim=-1)
        attn_probs = F.dropout(attn_probs, p=self.dropout, training=self.training)
        
        # Apply attention to values
        attn = torch.matmul(attn_probs, v)  # [B, H, T, D]
        
        # Reshape back
        attn = attn.transpose(1, 2).contiguous().view(batch_size, tgt_len, embed_dim)
        
        # Final projection and normalization
        output = self.out_proj(attn)
        output = self.out_norm(output)
        
        self.last_attn_weights = attn_weights if need_weights else None
        
        return output, attn_weights

class TransformerEncoderLayer(nn.Module):
    """Transformer encoder layer with custom multi-head attention."""
    
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.self_attn = CustomMultiheadAttention(d_model, nhead, dropout=dropout)
        
        self.ff_block = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src: torch.Tensor, return_attention: bool = False) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            src: Input tensor [B, T, E]
            return_attention: Whether to return attention weights
            
        Returns:
            Output tensor [B, T, E] and optionally attention weights
        """
        # Self-attention
        src2, attn_weights = self.self_attn(src, src, src)
        src = src + self.dropout(src2)
        src = self.norm1(src)
        
        # Feed-forward
        src2 = self.ff_block(src)
        src = src + self.dropout(src2)
        src = self.norm2(src)
        
        if return_attention:
            return src, attn_weights
        return src

##############################################################################
# Ablation Model
##############################################################################

class AblationModel(nn.Module):
    """
    Ablation model that supports different input modality combinations.
    
    Modalities:
    - ROI: Region of Interest time series
    - ICA: Independent Component Analysis time series
    - Connectivity: Dynamic functional connectivity matrices
    """
    
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.ablation_config = config.ablation_config
        
        # Initialize components based on ablation configuration
        # ROI branch
        if "roi" in self.ablation_config or self.ablation_config == "full":
            self.roi_encoder = DynamicTemporalBlock(config.n_rois, config.roi_hidden_dim)
        
        # ICA branch
        if "ica" in self.ablation_config or self.ablation_config == "full":
            self.ica_encoder = DynamicTemporalBlock(config.n_ica, config.ica_hidden_dim)
            
            # Additional dedicated ICA handling for ica-only configuration
            if self.ablation_config == "ica":
                logging.warning("Creating specialized ICA-only processing branch")
                # Add a simpler, more robust encoder path for ICA features
                self.ica_only_encoder = nn.Sequential(
                    nn.Linear(config.n_ica, config.ica_hidden_dim),
                    nn.LayerNorm(config.ica_hidden_dim),
                    nn.GELU(),
                    nn.Dropout(0.1)
                )
        
        # Connectivity branch
        if "conn" in self.ablation_config or self.ablation_config == "full":
            self.conn_encoder = CorrelationEncoder(config.n_rois, config.roi_hidden_dim)
        
        # Calculate combined hidden dimension for transformer
        self.combined_dim = 0
        if "roi" in self.ablation_config or self.ablation_config == "full":
            self.combined_dim += config.roi_hidden_dim
            
        if "ica" in self.ablation_config or self.ablation_config == "full":
            self.combined_dim += config.ica_hidden_dim
            
        if "conn" in self.ablation_config or self.ablation_config == "full":
            self.combined_dim += config.roi_hidden_dim
        
        # Ensure combined dimension is valid
        if self.combined_dim == 0:
            logging.error(f"Invalid ablation configuration: {config.ablation_config}")
            # Default to a minimal dimension to prevent errors
            self.combined_dim = 64
        
        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            TransformerEncoderLayer(
                self.combined_dim, 
                config.n_heads, 
                dim_feedforward=self.combined_dim * 4,
                dropout=config.dropout
            )
            for _ in range(config.n_layers)
        ])
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.combined_dim, self.combined_dim // 2),
            nn.LayerNorm(self.combined_dim // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(self.combined_dim // 2, 2)
        )
        
        # Special direct ICA classifier for fallback in case the main path fails
        if "ica" in self.ablation_config and "roi" not in self.ablation_config and "conn" not in self.ablation_config:
            logging.warning("Creating emergency ICA-only classifier fallback")
            self.emergency_ica_classifier = nn.Sequential(
                nn.Linear(config.n_ica, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 2)
            )
        
        self.attention_weights = []
        
        # Apply proper weight initialization to improve stability
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights to prevent vanishing/exploding gradients."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def _process_ica_features(self, ica_features):
        """
        Special processing for ICA features to ensure numerical stability.
        """
        if ica_features is None:
            return None
            
        logging.warning(f"ICA features tensor stats before processing: shape={ica_features.shape}")
        logging.warning(f"  min={ica_features.min().item():.6f}, max={ica_features.max().item():.6f}")
        logging.warning(f"  mean={ica_features.mean().item():.6f}, std={ica_features.std().item():.6f}")
        
        # Apply layer normalization to stabilize values
        batch_size, seq_len, feat_dim = ica_features.shape
        
        # 1. Apply 1D batch normalization across the feature dimension
        ica_features = ica_features.transpose(1, 2)  # [B, D, T]
        bn = nn.BatchNorm1d(feat_dim, affine=False).to(ica_features.device)
        ica_features = bn(ica_features)
        ica_features = ica_features.transpose(1, 2)  # [B, T, D]
        
        # 2. Apply additional scaling/standardization if needed
        if not torch.isfinite(ica_features).all():
            logging.warning("Non-finite values in ICA features after batch norm! Cleaning.")
            ica_features = torch.nan_to_num(ica_features, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 3. Clip if still has extreme values
        if ica_features.abs().max() > 10:
            logging.warning("ICA features still have extreme values. Clipping to [-5, 5].")
            ica_features = torch.clamp(ica_features, -5.0, 5.0)
        
        logging.warning(f"ICA features after processing: min={ica_features.min().item():.6f}, max={ica_features.max().item():.6f}")
        
        return ica_features
    
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            batch: Dictionary containing input tensors based on ablation config
            
        Returns:
            Dictionary with model outputs
        """
        logging.info(f"--- AblationModel Forward: Config='{self.ablation_config}' ---")
        
        # SPECIAL CASE: Direct ICA-only pathway
        # For pure ICA configuration, try the simpler, more direct path first
        if self.ablation_config == "ica" and "ica_data" in batch and hasattr(self, 'emergency_ica_classifier'):
            try:
                logging.warning("Attempting emergency direct ICA processing path...")
                # Get raw ICA data
                raw_ica = batch["ica_data"]  # [B, T, ICA]
                # Global average pooling across time dimension
                pooled_ica = raw_ica.mean(dim=1)  # [B, ICA]
                # Process through special classifier
                emergency_logits = self.emergency_ica_classifier(pooled_ica)
                
                # If we got here without error, return these results along with a flag
                # We'll still try the normal path too and compare results
                emergency_succeeded = True
                emergency_results = {
                    "logits": emergency_logits,
                    "features": pooled_ica,
                    "is_emergency": True
                }
                logging.warning("Emergency ICA path succeeded!")
            except Exception as e:
                logging.error(f"Emergency ICA path failed: {e}")
                emergency_succeeded = False
                emergency_results = None
        else:
            emergency_succeeded = False
            emergency_results = None
        
        # --- Continue with normal processing path ---
        # Process each modality separately first
        roi_features = None
        ica_features = None
        conn_features = None
        
        # Process ROI data if included
        if "roi" in self.ablation_config or self.ablation_config == "full":
            if "roi_data" not in batch:
                logging.error("ROI data expected but not found in batch!")
            else:
                try:
                    logging.info("Processing ROI...")
                    roi_features = self.roi_encoder(batch["roi_data"])  # [B, T_roi, D_roi]
                    if not torch.isfinite(roi_features).all():
                        logging.warning("NaN/Inf in roi_features after encoder! Cleaning.")
                        roi_features = torch.nan_to_num(roi_features, nan=0.0, posinf=0.0, neginf=0.0)
                except Exception as e:
                    logging.error(f"Error processing ROI data: {e}")
                    roi_features = None
        
        # Process ICA data if included
        if "ica" in self.ablation_config or self.ablation_config == "full":
            if "ica_data" not in batch:
                 logging.error("ICA data expected but not found in batch!")
            else:
                try:
                    logging.info("Processing ICA...")
                    # Track statistics of raw input ICA
                    ica_input = batch["ica_data"]
                    logging.warning(f"Raw ICA input: Shape={ica_input.shape}, "
                                  f"Min={ica_input.min().item():.6f}, Max={ica_input.max().item():.6f}, "
                                  f"Mean={ica_input.mean().item():.6f}, Std={ica_input.std().item():.6f}")
                    
                    # For ICA-only ablation, use the specialized simpler encoder first
                    if self.ablation_config == "ica" and hasattr(self, 'ica_only_encoder'):
                        logging.warning("Using specialized ICA-only encoder")
                        # Process raw ICA data with specialized encoder
                        # Process each time step independently for stability
                        batch_size, seq_len, n_ica = ica_input.shape
                        processed_ica = []
                        
                        for t in range(seq_len):
                            time_slice = ica_input[:, t, :]  # [B, ICA]
                            processed_t = self.ica_only_encoder(time_slice)  # [B, D_hidden]
                            processed_ica.append(processed_t)
                            
                        ica_features = torch.stack(processed_ica, dim=1)  # [B, T, D_hidden]
                        logging.warning(f"ICA features from specialized encoder: Shape={ica_features.shape}")
                    else:
                        # Process with the regular encoder for other configurations
                        ica_features = self.ica_encoder(batch["ica_data"])  # [B, T_ica, D_ica]
                    
                    logging.info(f"ICA features after encoder: Shape={ica_features.shape}, Finite={torch.isfinite(ica_features).all()}")
                    
                    # Apply special ICA processing
                    ica_features = self._process_ica_features(ica_features)
                    
                    if not torch.isfinite(ica_features).all():
                        logging.warning("NaN/Inf in ica_features after special processing! Cleaning.")
                        ica_features = torch.nan_to_num(ica_features, nan=0.0, posinf=0.0, neginf=0.0)
                except Exception as e:
                    logging.error(f"Error processing ICA data: {e}")
                    logging.error(traceback.format_exc())
                    ica_features = None
        
        # Process connectivity data if included
        if "conn" in self.ablation_config or self.ablation_config == "full":
            if "corr_data" not in batch:
                logging.error("Connectivity data expected but not found in batch!")
            else:
                try:
                    logging.info("Processing Connectivity...")
                    conn_features = self.conn_encoder(batch["corr_data"])  # [B, T_conn, D_conn]
                    if not torch.isfinite(conn_features).all():
                        logging.warning("NaN/Inf in conn_features after encoder! Cleaning.")
                        conn_features = torch.nan_to_num(conn_features, nan=0.0, posinf=0.0, neginf=0.0)
                except Exception as e:
                    logging.error(f"Error processing connectivity data: {e}")
                    conn_features = None
        
        # Collect successfully processed features
        features_to_combine = []
        processed_feature_names = []
        if roi_features is not None:
            features_to_combine.append(roi_features)
            processed_feature_names.append("ROI")
        if ica_features is not None:
            features_to_combine.append(ica_features)
            processed_feature_names.append("ICA")
        if conn_features is not None:
            features_to_combine.append(conn_features)
            processed_feature_names.append("Conn")
            
        logging.info(f"Processed features: {processed_feature_names}")

        if not features_to_combine:
            logging.error("All feature extraction failed or no features selected by config.")
            # Return zeros as a fallback
            batch_size = batch["label"].size(0)
            return {
                "logits": torch.zeros(batch_size, 2, device=batch["label"].device),
                "features": torch.zeros(batch_size, self.combined_dim, device=batch["label"].device)
            }
        
        # --- Feature Alignment (if multiple features) ---
        if len(features_to_combine) > 1:
            logging.info("Aligning multiple features...")
            target_len = -1
            # Prefer length of ROI, then ICA, then Conn for alignment target
            if roi_features is not None: target_len = roi_features.size(1)
            elif ica_features is not None: target_len = ica_features.size(1)
            elif conn_features is not None: target_len = conn_features.size(1)

            if target_len <= 0:
                logging.warning("Could not determine valid target length for alignment. Trying max length.")
                target_len = max(f.size(1) for f in features_to_combine if f.size(1) > 0) if any(f.size(1) > 0 for f in features_to_combine) else -1
            
            if target_len <= 0:
                 logging.error("All features have zero length. Cannot align or combine.")
                 # Return zeros as a fallback
                 batch_size = batch["label"].size(0)
                 return {
                    "logits": torch.zeros(batch_size, 2, device=batch["label"].device),
                    "features": torch.zeros(batch_size, self.combined_dim, device=batch["label"].device)
                 }

            aligned_features = []
            for i, features in enumerate(features_to_combine):
                name = processed_feature_names[i]
                if features.size(1) != target_len:
                    logging.info(f"Aligning {name} from {features.size(1)} to {target_len}...")
                    try:
                        aligned = F.interpolate(features.transpose(1,2), size=target_len, mode='linear', align_corners=False).transpose(1,2)
                        if not torch.isfinite(aligned).all():
                           logging.warning(f"NaN/Inf detected in aligned {name} features! Cleaning.")
                           aligned = torch.nan_to_num(aligned, nan=0.0, posinf=0.0, neginf=0.0)
                        aligned_features.append(aligned)
                    except Exception as e:
                        logging.error(f"Error aligning {name}: {e}. Skipping feature.")
                else:
                    # Still check finiteness even if not interpolating
                    if not torch.isfinite(features).all():
                        logging.warning(f"NaN/Inf detected in {name} features (no alignment needed)! Cleaning.")
                        features = torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
                    aligned_features.append(features)
            features_to_combine = aligned_features
            
            if not features_to_combine:
                 logging.error("No features left after alignment process.")
                 batch_size = batch["label"].size(0)
                 return {
                    "logits": torch.zeros(batch_size, 2, device=batch["label"].device),
                    "features": torch.zeros(batch_size, self.combined_dim, device=batch["label"].device)
                 }
                 
        # --- Concatenate features --- 
        # If only one feature type, features_to_combine has 1 element and cat is trivial
        try:
            combined_features = torch.cat(features_to_combine, dim=-1)  # [B, T, D_combined]
            logging.info(f"Combined features: Shape={combined_features.shape}, Finite={torch.isfinite(combined_features).all()}")
            if not torch.isfinite(combined_features).all():
                 logging.warning("NaN/Inf in combined_features after cat! Cleaning.")
                 combined_features = torch.nan_to_num(combined_features, nan=0.0, posinf=0.0, neginf=0.0)
        except Exception as e:
             logging.error(f"Error during feature concatenation: {e}")
             logging.error(traceback.format_exc())
             for i, f in enumerate(features_to_combine):
                 logging.error(f"  Feature {i} ({processed_feature_names[i]}) shape: {f.shape}, is_finite: {torch.isfinite(f).all()}")
             # Return zeros as a fallback
             batch_size = batch["label"].size(0)
             return {
                "logits": torch.zeros(batch_size, 2, device=batch["label"].device),
                "features": torch.zeros(batch_size, self.combined_dim, device=batch["label"].device)
             }

        # Apply transformer layers
        transformer_output = combined_features
        self.attention_weights = []
        logging.info("Applying Transformer Layers...")
        for i, layer in enumerate(self.transformer_layers):
            try:
                # Add NaN check before layer
                if not torch.isfinite(transformer_output).all():
                    logging.warning(f"NaN/Inf before Transformer layer {i}! Cleaning.")
                    transformer_output = torch.nan_to_num(transformer_output, nan=0.0, posinf=0.0, neginf=0.0)
                
                transformer_output, attn = layer(transformer_output, return_attention=True)
                self.attention_weights.append(attn)
                
                # Add NaN check after layer
                if not torch.isfinite(transformer_output).all():
                    logging.warning(f"NaN/Inf after Transformer layer {i}! Cleaning.")
                    transformer_output = torch.nan_to_num(transformer_output, nan=0.0, posinf=0.0, neginf=0.0)
                    
            except Exception as e:
                logging.error(f"Error in transformer layer {i}: {e}")
                logging.error(traceback.format_exc())
                # Use the input as output for this layer (potentially cleaned)
                self.attention_weights.append(None)
                # Break the loop if a transformer layer fails catastrophically?
                # Maybe better to continue with potentially bad data? For now, continue.
        
        logging.info(f"Transformer output: Shape={transformer_output.shape}, Finite={torch.isfinite(transformer_output).all()}")
        
        # Apply mean pooling *after* the transformer, across the sequence dimension
        try:
            if transformer_output.size(1) == 0: # Check for empty sequence dimension
                 logging.error("Transformer output has zero sequence length. Cannot pool.")
                 pooled_output = torch.zeros((transformer_output.size(0), transformer_output.size(2)), device=transformer_output.device, dtype=transformer_output.dtype)
            else:
                pooled_output = transformer_output.mean(dim=1) # [B, D_combined]
            logging.info(f"Pooled output: Shape={pooled_output.shape}, Finite={torch.isfinite(pooled_output).all()}")
            if not torch.isfinite(pooled_output).all():
                logging.warning("NaN/Inf in pooled_output! Cleaning.")
                pooled_output = torch.nan_to_num(pooled_output, nan=0.0, posinf=0.0, neginf=0.0)
        except Exception as e:
             logging.error(f"Error during pooling: {e}")
             logging.error(traceback.format_exc())
             batch_size = batch["label"].size(0)
             pooled_output = torch.zeros(batch_size, self.combined_dim, device=batch["label"].device)

        # Apply classifier
        try:
            # Add NaN check before classifier
            if not torch.isfinite(pooled_output).all():
                 logging.warning("NaN/Inf before Classifier! Cleaning.")
                 pooled_output = torch.nan_to_num(pooled_output, nan=0.0, posinf=0.0, neginf=0.0)
                 
            logits = self.classifier(pooled_output) # Use pooled output
            logging.info(f"Logits: Shape={logits.shape}, Finite={torch.isfinite(logits).all()}")
            
            # Add NaN check after classifier
            if not torch.isfinite(logits).all():
                logging.warning("NaN/Inf in logits after classifier! Cleaning.")
                logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)
        except Exception as e:
            logging.error(f"Error in classifier: {e}")
            logging.error(traceback.format_exc())
            # Return zeros as a fallback
            batch_size = batch["label"].size(0)
            logits = torch.zeros(batch_size, 2, device=batch["label"].device)
        
        logging.info("--- AblationModel Forward: END ---")
        return {
            "logits": logits,
            "features": pooled_output # Return the pooled features before classifier
        }
    
    def get_attention_weights(self) -> List[torch.Tensor]:
        """Return stored attention weights from the last forward pass."""
        return self.attention_weights
    
    def compute_loss(self, batch: Dict[str, torch.Tensor], reduction: str = "mean") -> torch.Tensor:
        """
        Compute loss for the given batch.
        
        Args:
            batch: Batch dictionary containing inputs and labels
            reduction: Loss reduction method
            
        Returns:
            Loss tensor
        """
        try:
            outputs = self.forward(batch)
            logits = outputs["logits"]
            
            # Check for NaN values in logits
            if torch.isnan(logits).any() or torch.isinf(logits).any():
                logging.warning("NaN or Inf values in logits during loss computation")
                # Return a zero loss to avoid breaking the training loop
                return torch.tensor(0.0, device=batch["label"].device, requires_grad=True)
            
            criterion = nn.CrossEntropyLoss(reduction=reduction)
            loss = criterion(logits, batch["label"])
            
            # Add L2 regularization to prevent overfitting (addressing reviewer comments)
            l2_reg = 0.0
            for param in self.parameters():
                l2_reg += torch.norm(param)
            
            # Add small regularization term
            reg_factor = 1e-5
            loss += reg_factor * l2_reg
            
            return loss
            
        except Exception as e:
            logging.error(f"Error in loss computation: {e}")
            # Return a zero loss to avoid breaking the training loop
            return torch.tensor(0.0, device=batch["label"].device, requires_grad=True)

##############################################################################
# Training and Evaluation Functions
##############################################################################

def train_and_evaluate(
    config: Config,
    fold: int,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    data_dict: Dict
) -> Dict:
    """
    Train and evaluate model for one fold.
    
    Args:
        config: Configuration object
        fold: Fold number
        train_indices: Training indices
        val_indices: Validation indices
        data_dict: Data dictionary
        
    Returns:
        Dictionary with evaluation results
    """
    logging.info(f"Training fold {fold+1}/{config.predefined_folds}")
    
    # Create datasets
    train_dataset = MultiModalDataset(data_dict, train_indices, config, train_mode=True)
    val_dataset = MultiModalDataset(data_dict, val_indices, config, train_mode=False)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True,
        num_workers=0,
        drop_last=True  # Prevent issues with batch norm on small batches
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.batch_size, 
        shuffle=False,
        num_workers=0
    )
    
    # Create model
    model = AblationModel(config).to(config.device)
    
    # Apply proper weight initialization
    for module in model.modules():
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    # Create optimizer with gradient clipping
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        eps=1e-8
    )
    
    # Create learning rate scheduler - use One Cycle policy for better convergence
    n_steps = len(train_loader) * config.n_epochs
    scheduler = OneCycleLR(
        optimizer,
        max_lr=config.learning_rate,
        total_steps=n_steps,
        pct_start=0.3,
        div_factor=25.0,
        final_div_factor=10000.0
    )
    
    # Training loop
    best_val_auc = 0.0
    best_val_loss = float("inf")
    best_epoch = 0
    best_model_state = None
    patience_counter = 0
    training_history = {"train_loss": [], "val_loss": [], "val_auc": [], "lr": []}
    
    logging.info(f"Starting training for {config.n_epochs} epochs")
    
    for epoch in range(config.n_epochs):
        # Train for one epoch
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            config.device,
            config.gradient_clip,
            scheduler
        )
        
        # Evaluate on validation set
        val_metrics, val_logits, val_labels = evaluate(
            model,
            val_loader,
            config.device
        )
        val_loss = val_metrics["loss"]
        val_auc = val_metrics["auc"]
        
        # Record training history
        training_history["train_loss"].append(train_loss)
        training_history["val_loss"].append(val_loss)
        training_history["val_auc"].append(val_auc)
        training_history["lr"].append(optimizer.param_groups[0]["lr"])
        
        # Log metrics
        logging.info(f"Epoch {epoch+1}/{config.n_epochs} - "
                  f"Train Loss: {train_loss:.4f}, "
                  f"Val Loss: {val_loss:.4f}, "
                  f"Val AUC: {val_auc:.4f}, "
                  f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Check for improvement
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_val_loss = val_loss
            best_epoch = epoch
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            
        # Early stopping
        if patience_counter >= config.early_stopping_patience:
            logging.info(f"Early stopping at epoch {epoch+1}")
            break
    
    # Load best model
    logging.info(f"Best performance at epoch {best_epoch+1}: Val AUC = {best_val_auc:.4f}")
    model.load_state_dict(best_model_state)
    
    # Final evaluation
    val_metrics, val_logits, val_labels = evaluate(
        model,
        val_loader,
        config.device
    )
    
    # Compute confidence intervals
    softmax_probs = F.softmax(torch.FloatTensor(val_logits), dim=1).numpy()
    pos_probs = softmax_probs[:, 1]
    confidence_intervals = compute_confidence_intervals(val_labels, pos_probs)
    
    # Format metrics with confidence intervals
    formatted_metrics = format_metrics_with_ci(val_metrics, confidence_intervals)
    
    # Save model
    model_save_path = config.run_dir / "models" / f"model_fold_{fold}.pt"
    torch.save(
        {"model_state": best_model_state, 
         "config": {k: v for k, v in config.__dict__.items() if not k.startswith('__') and not callable(v) and k != 'device'}},
        model_save_path
    )
    
    # Plot training history
    plot_training_history(training_history, fold, config)
    
    return {
        "fold": fold,
        "best_epoch": best_epoch,
        "metrics": val_metrics,
        "confidence_intervals": confidence_intervals,
        "formatted_metrics": formatted_metrics,
        "val_logits": val_logits,
        "val_labels": val_labels,
        "val_indices": val_indices,
        "training_history": training_history
    }

def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    gradient_clip: float,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
) -> float:
    """
    Train model for one epoch.
    
    Args:
        model: Model to train
        train_loader: Training data loader
        optimizer: Optimizer
        device: Device to use
        gradient_clip: Gradient clipping value
        scheduler: Learning rate scheduler
        
    Returns:
        Average training loss
    """
    model.train()
    total_loss = 0.0
    
    for batch in train_loader:
        # Move batch to device
        batch = {k: v.to(device) for k, v in batch.items()}
        
        # Forward and backward pass
        optimizer.zero_grad()
        
        try:
            loss = model.compute_loss(batch)
            
            # Skip bad batches
            if torch.isnan(loss).any() or torch.isinf(loss).any():
                logging.warning("NaN or Inf loss detected during training, skipping batch")
                continue
                
            loss.backward()
            
            # Gradient clipping
            if gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                
            optimizer.step()
            
            # Step scheduler if provided
            if scheduler is not None:
                scheduler.step()
            
            total_loss += loss.item()
            
        except Exception as e:
            logging.error(f"Error in training batch: {e}")
            continue
    
    return total_loss / len(train_loader)

def evaluate(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device
) -> Dict[str, float]:
    """
    Evaluate model on validation set.
    
    Args:
        model: Model to evaluate
        val_loader: Validation data loader
        device: Device to use
        
    Returns:
        Dictionary with evaluation metrics (loss, accuracy, AUC)
    """
    model.eval()
    all_logits = []
    all_labels = []
    total_loss = 0.0
    
    with torch.no_grad():
        for batch in val_loader:
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Forward pass
            outputs = model(batch)
            logits = outputs["logits"]
            
            # Check for NaN values
            if torch.isnan(logits).any():
                logging.warning("NaN values detected in logits, skipping batch")
                continue
            
            # Compute loss
            loss = model.compute_loss(batch)
            
            # Check for NaN loss
            if torch.isnan(loss).any():
                logging.warning("NaN loss detected, skipping batch")
                continue
                
            total_loss += loss.item()
            
            # Store predictions and labels
            all_logits.append(logits.detach().cpu().numpy())
            all_labels.append(batch["label"].detach().cpu().numpy())
    
    # Check if we have any predictions
    if len(all_logits) == 0:
        logging.error("No valid predictions, all batches had NaN values")
        return {
            "loss": float('inf'),
            "accuracy": 0.0,
            "balanced_accuracy": 0.0,
            "auc": 0.5,
            "f1": 0.0,
            "recall": 0.0,
            "precision": 0.0,
            "mcc": 0.0
        }, np.zeros((0, 2)), np.zeros(0)
    
    # Concatenate predictions and labels
    all_logits = np.concatenate(all_logits, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    # Check for NaN values again
    if np.isnan(all_logits).any():
        logging.warning("NaN values in concatenated logits, replacing with zeros")
        all_logits = np.nan_to_num(all_logits, nan=0.0)
    
    # Compute metrics
    softmax_probs = F.softmax(torch.FloatTensor(all_logits), dim=1).numpy()
    pos_probs = softmax_probs[:, 1]  # Probability of positive class
    predictions = np.argmax(softmax_probs, axis=1)
    
    # Ensure we have at least one sample of each class for AUC calculation
    have_both_classes = len(np.unique(all_labels)) > 1
    
    metrics = {
        "loss": total_loss / max(len(val_loader), 1),
        "accuracy": accuracy_score(all_labels, predictions),
        "balanced_accuracy": balanced_accuracy_score(all_labels, predictions),
        "auc": roc_auc_score(all_labels, pos_probs) if have_both_classes else 0.5,
        "f1": f1_score(all_labels, predictions, zero_division=0),
        "recall": recall_score(all_labels, predictions, zero_division=0),
        "precision": precision_score(all_labels, predictions, zero_division=0),
        "mcc": matthews_corrcoef(all_labels, predictions)
    }
    
    return metrics, all_logits, all_labels

def compute_confidence_intervals(
    labels: np.ndarray,
    probs: np.ndarray,
    n_bootstrap: int = 10000
) -> Dict[str, Tuple[float, float]]:
    """
    Compute 95% confidence intervals for classification metrics using bootstrapping.
    
    Args:
        labels: True labels
        probs: Predicted probabilities for positive class
        n_bootstrap: Number of bootstrap samples
        
    Returns:
        Dictionary with metric names and their 95% CIs
    """
    rng = np.random.RandomState(SEED)
    auc_vals, acc_vals, balanced_acc_vals, f1_vals = [], [], [], []
    
    preds = (probs >= 0.5).astype(int)
    
    for _ in range(n_bootstrap):
        # Sample with replacement
        idx = rng.randint(0, len(labels), len(labels))
        sample_labels = labels[idx]
        sample_probs = probs[idx]
        sample_preds = preds[idx]
        
        # Skip if we don't have both classes
        if len(np.unique(sample_labels)) < 2:
            continue
        
        # Compute metrics for this bootstrap sample
        try:
            auc_vals.append(roc_auc_score(sample_labels, sample_probs))
            acc_vals.append(accuracy_score(sample_labels, sample_preds))
            balanced_acc_vals.append(balanced_accuracy_score(sample_labels, sample_preds))
            f1_vals.append(f1_score(sample_labels, sample_preds))
        except:
            continue
    
    # Compute 95% confidence intervals
    result = {}
    if len(auc_vals) > 0:
        result["auc"] = (np.percentile(auc_vals, 2.5), np.percentile(auc_vals, 97.5))
        result["accuracy"] = (np.percentile(acc_vals, 2.5), np.percentile(acc_vals, 97.5))
        result["balanced_accuracy"] = (np.percentile(balanced_acc_vals, 2.5), np.percentile(balanced_acc_vals, 97.5))
        result["f1"] = (np.percentile(f1_vals, 2.5), np.percentile(f1_vals, 97.5))
    else:
        # Default values if bootstrapping failed
        result["auc"] = (0.5, 0.5)
        result["accuracy"] = (0.5, 0.5)
        result["balanced_accuracy"] = (0.5, 0.5)
        result["f1"] = (0.0, 0.0)
    
    return result

def format_metrics_with_ci(metrics: Dict[str, float], cis: Dict[str, Tuple[float, float]]) -> Dict[str, str]:
    """
    Format metrics with their confidence intervals.
    
    Args:
        metrics: Dictionary with metric values
        cis: Dictionary with confidence intervals
        
    Returns:
        Dictionary with formatted metrics
    """
    formatted = {}
    for metric, value in metrics.items():
        if metric in cis:
            lower, upper = cis[metric]
            formatted[metric] = f"{value:.3f} [{lower:.3f}-{upper:.3f}]"
        else:
            formatted[metric] = f"{value:.3f}"
    
    return formatted

def train_and_evaluate(
    config: Config,
    fold: int,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    data_dict: Dict
) -> Dict:
    """
    Train and evaluate model for one fold.
    
    Args:
        config: Configuration object
        fold: Fold number
        train_indices: Training indices
        val_indices: Validation indices
        data_dict: Data dictionary
        
    Returns:
        Dictionary with evaluation results
    """
    logging.info(f"Training fold {fold+1}/{config.predefined_folds}")
    
    # Create datasets
    train_dataset = MultiModalDataset(data_dict, train_indices, config, train_mode=True)
    val_dataset = MultiModalDataset(data_dict, val_indices, config, train_mode=False)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True,
        num_workers=0,
        drop_last=True  # Prevent issues with batch norm on small batches
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.batch_size, 
        shuffle=False,
        num_workers=0
    )
    
    # Create model
    model = AblationModel(config).to(config.device)
    
    # Apply proper weight initialization
    for module in model.modules():
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    # Create optimizer with gradient clipping
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        eps=1e-8
    )
    
    # Create learning rate scheduler - use One Cycle policy for better convergence
    n_steps = len(train_loader) * config.n_epochs
    scheduler = OneCycleLR(
        optimizer,
        max_lr=config.learning_rate,
        total_steps=n_steps,
        pct_start=0.3,
        div_factor=25.0,
        final_div_factor=10000.0
    )
    
    # Training loop
    best_val_auc = 0.0
    best_val_loss = float("inf")
    best_epoch = 0
    best_model_state = None
    patience_counter = 0
    training_history = {"train_loss": [], "val_loss": [], "val_auc": [], "lr": []}
    
    logging.info(f"Starting training for {config.n_epochs} epochs")
    
    for epoch in range(config.n_epochs):
        # Train for one epoch
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            config.device,
            config.gradient_clip,
            scheduler
        )
        
        # Evaluate on validation set
        val_metrics, val_logits, val_labels = evaluate(
            model,
            val_loader,
            config.device
        )
        val_loss = val_metrics["loss"]
        val_auc = val_metrics["auc"]
        
        # Record training history
        training_history["train_loss"].append(train_loss)
        training_history["val_loss"].append(val_loss)
        training_history["val_auc"].append(val_auc)
        training_history["lr"].append(optimizer.param_groups[0]["lr"])
        
        # Log metrics
        logging.info(f"Epoch {epoch+1}/{config.n_epochs} - "
                  f"Train Loss: {train_loss:.4f}, "
                  f"Val Loss: {val_loss:.4f}, "
                  f"Val AUC: {val_auc:.4f}, "
                  f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Check for improvement
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_val_loss = val_loss
            best_epoch = epoch
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            
        # Early stopping
        if patience_counter >= config.early_stopping_patience:
            logging.info(f"Early stopping at epoch {epoch+1}")
            break
    
    # Load best model
    logging.info(f"Best performance at epoch {best_epoch+1}: Val AUC = {best_val_auc:.4f}")
    model.load_state_dict(best_model_state)
    
    # Final evaluation
    val_metrics, val_logits, val_labels = evaluate(
        model,
        val_loader,
        config.device
    )
    
    # Compute confidence intervals
    softmax_probs = F.softmax(torch.FloatTensor(val_logits), dim=1).numpy()
    pos_probs = softmax_probs[:, 1]
    confidence_intervals = compute_confidence_intervals(val_labels, pos_probs)
    
    # Format metrics with confidence intervals
    formatted_metrics = format_metrics_with_ci(val_metrics, confidence_intervals)
    
    # Save model
    model_save_path = config.run_dir / "models" / f"model_fold_{fold}.pt"
    torch.save(
        {"model_state": best_model_state, 
         "config": {k: v for k, v in config.__dict__.items() if not k.startswith('__') and not callable(v) and k != 'device'}},
        model_save_path
    )
    
    # Plot training history
    plot_training_history(training_history, fold, config)
    
    return {
        "fold": fold,
        "best_epoch": best_epoch,
        "metrics": val_metrics,
        "confidence_intervals": confidence_intervals,
        "formatted_metrics": formatted_metrics,
        "val_logits": val_logits,
        "val_labels": val_labels,
        "val_indices": val_indices,
        "training_history": training_history
    }

def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    gradient_clip: float,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
) -> float:
    """
    Train model for one epoch.
    
    Args:
        model: Model to train
        train_loader: Training data loader
        optimizer: Optimizer
        device: Device to use
        gradient_clip: Gradient clipping value
        scheduler: Learning rate scheduler
        
    Returns:
        Average training loss
    """
    model.train()
    total_loss = 0.0
    
    for batch in train_loader:
        # Move batch to device
        batch = {k: v.to(device) for k, v in batch.items()}
        
        # Forward and backward pass
        optimizer.zero_grad()
        
        try:
            loss = model.compute_loss(batch)
            
            # Skip bad batches
            if torch.isnan(loss).any() or torch.isinf(loss).any():
                logging.warning("NaN or Inf loss detected during training, skipping batch")
                continue
                
            loss.backward()
            
            # Gradient clipping
            if gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                
            optimizer.step()
            
            # Step scheduler if provided
            if scheduler is not None:
                scheduler.step()
            
            total_loss += loss.item()
            
        except Exception as e:
            logging.error(f"Error in training batch: {e}")
            continue
    
    return total_loss / len(train_loader)

def plot_training_history(training_history: Dict[str, List[float]], fold: int, config: Config):
    """
    Plot training history for a fold.
    
    Args:
        training_history: Dictionary with training history
        fold: Fold number
        config: Configuration object
    """
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot loss
    epochs = range(1, len(training_history["train_loss"]) + 1)
    ax1.plot(epochs, training_history["train_loss"], 'b-', label='Training Loss')
    ax1.plot(epochs, training_history["val_loss"], 'r-', label='Validation Loss')
    ax1.set_title(f'Loss - Fold {fold+1}', fontsize=16)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.legend(fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Plot AUC and learning rate
    ax2.plot(epochs, training_history["val_auc"], 'g-', label='Validation AUC')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('AUC', fontsize=12)
    ax2.set_ylim(0, 1.1)
    ax2.legend(loc='upper left', fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # Plot learning rate on secondary y-axis
    ax3 = ax2.twinx()
    ax3.plot(epochs, training_history["lr"], 'm--', label='Learning Rate')
    ax3.set_ylabel('Learning Rate', fontsize=12)
    ax3.legend(loc='upper right', fontsize=12)
    
    # Add overall title
    plt.suptitle(f'Training History - {config.dataset_name.capitalize()} - {config.ablation_config} - Fold {fold+1}',
                fontsize=18)
    
    # Save figure
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(config.run_dir / "figures" / f"training_history_fold_{fold+1}.png", dpi=300)
    plt.close()

##############################################################################
# Cross-Validation and Visualization
##############################################################################

def run_cross_validation(config: Config, data_dict: Dict) -> List[Dict]:
    """
    Run k-fold cross-validation.
    
    Args:
        config: Configuration object
        data_dict: Data dictionary
        
    Returns:
        List of dictionaries with results for each fold
    """
    labels = data_dict["labels"]
    n_samples = len(labels)
    
    # Create stratified k-fold split
    skf = StratifiedKFold(n_splits=config.predefined_folds, shuffle=True, random_state=SEED)
    
    # Store results for each fold
    fold_results = []
    
    # Train and evaluate for each fold
    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(n_samples), labels)):
        fold_result = train_and_evaluate(
            config,
            fold,
            train_idx,
            val_idx,
            data_dict
        )
        fold_results.append(fold_result)
        
        # Log metrics for this fold
        logging.info(f"Fold {fold+1}/{config.predefined_folds} results:")
        for metric, value in fold_result["formatted_metrics"].items():
            logging.info(f"  {metric}: {value}")
    
    return fold_results

def aggregate_results(fold_results: List[Dict]) -> Dict:
    """
    Aggregate results from all folds.
    
    Args:
        fold_results: List of dictionaries with results for each fold
        
    Returns:
        Dictionary with aggregated results
    """
    # Extract metrics from each fold
    all_metrics = {
        "auc": [],
        "accuracy": [],
        "balanced_accuracy": [],
        "f1": [],
        "precision": [],
        "recall": [],
        "mcc": []
    }
    
    for result in fold_results:
        metrics = result["metrics"]
        for metric in all_metrics.keys():
            all_metrics[metric].append(metrics[metric])
    
    # Compute mean and standard deviation
    aggregated = {}
    for metric, values in all_metrics.items():
        mean_value = np.mean(values)
        std_value = np.std(values)
        ci_lower = np.percentile(values, 2.5)
        ci_upper = np.percentile(values, 97.5)
        
        aggregated[metric] = {
            "mean": mean_value,
            "std": std_value,
            "ci": (ci_lower, ci_upper),
            "formatted": f"{mean_value:.3f} ± {std_value:.3f} [{ci_lower:.3f}-{ci_upper:.3f}]"
        }
    
    return aggregated

def create_predictions_dataframe(fold_results: List[Dict], data_dict: Dict) -> pd.DataFrame:
    """
    Create a DataFrame with all predictions and true labels from all folds.
    """
    all_preds = []
    all_labels = []
    all_indices = []
    all_fold_ids = []

    for fold_idx, fold_result in enumerate(fold_results):
        if "val_predictions" not in fold_result or len(fold_result["val_predictions"]) == 0:
            logging.warning(f"No predictions found for fold {fold_idx+1}")
            continue
            
        for val_idx, pred_logits in zip(fold_result["val_indices"], fold_result["val_predictions"]):
            if pred_logits is None or np.isnan(pred_logits).any():
                logging.warning(f"Skipping prediction with NaN values in fold {fold_idx+1}")
                continue
                
            pred_probs = softmax(pred_logits)
            pred_class = np.argmax(pred_probs)
            
            all_preds.append(pred_class)
            all_labels.append(data_dict["labels"][val_idx])
            all_indices.append(val_idx)
            all_fold_ids.append(fold_idx + 1)
    
    if len(all_indices) == 0:
        logging.warning("No valid predictions found across all folds")
        # Return an empty DataFrame with the correct columns
        return pd.DataFrame(columns=[
            "fold", "subject_id", "true_label", "predicted_label"
        ])
    
    # Check if subject_ids exist and have sufficient length
    subject_ids = []
    if "subject_ids" in data_dict and len(data_dict["subject_ids"]) > 0:
        for idx in all_indices:
            if idx < len(data_dict["subject_ids"]):
                subject_ids.append(data_dict["subject_ids"][idx])
            else:
                subject_ids.append(f"Subject_{idx}")
    else:
        subject_ids = [f"Subject_{idx}" for idx in all_indices]
    
    return pd.DataFrame({
        "fold": all_fold_ids,
        "subject_id": subject_ids,
        "true_label": all_labels,
        "predicted_label": all_preds
    })

def plot_roc_curve(fold_results: List[Dict], config: Config):
    """
    Plot ROC curve for each fold.
    
    Args:
        fold_results: List of dictionaries with results for each fold
        config: Configuration object
    """
    plt.figure(figsize=(10, 8))
    
    # Setup plot
    plt.plot([0, 1], [0, 1], 'k--', label='Chance level')
    
    # Calculate mean ROC curve
    mean_fpr = np.linspace(0, 1, 100)
    tprs = []
    aucs = []
    
    # Plot ROC curve for each fold
    for i, result in enumerate(fold_results):
        labels = result["val_labels"]
        logits = result["val_logits"]
        
        # Skip if no logits available (all batches had NaN)
        if len(logits) == 0:
            logging.warning(f"Skipping ROC curve for fold {i+1} - no valid logits")
            continue
            
        probabilities = F.softmax(torch.FloatTensor(logits), dim=1).numpy()[:, 1]
        
        # Check if we have samples from both classes
        unique_labels = np.unique(labels)
        if len(unique_labels) < 2:
            logging.warning(f"Skipping ROC curve for fold {i+1} - only found classes: {unique_labels}")
            continue
        
        try:
            fpr, tpr, _ = roc_curve(labels, probabilities, pos_label=1)
            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
            
            auc_val = result["metrics"]["auc"]
            aucs.append(auc_val)
            
            plt.plot(fpr, tpr, lw=1, alpha=0.3, 
                    label=f'Fold {i+1} (AUC = {auc_val:.3f})')
        except Exception as e:
            logging.warning(f"Error calculating ROC curve for fold {i+1}: {e}")
            continue
    
    # If no valid ROC curves, show a basic plot
    if len(tprs) == 0:
        plt.title("ROC Curve (No valid folds)")
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.legend()
        
        # Save figure
        output_dir = Path(config.run_dir) / "figures"
        output_dir.mkdir(exist_ok=True)
        plt.savefig(output_dir / "roc_curve.png", dpi=300)
        plt.close()
        return
    
    # Plot mean ROC curve
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)
    
    plt.plot(mean_fpr, mean_tpr, color='b', lw=2, alpha=0.8,
            label=f'Mean ROC (AUC = {mean_auc:.3f} ± {std_auc:.3f})')
    
    # Add confidence interval
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=0.2,
                    label='± 1 std. dev.')
    
    # Configure plot
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for {config.dataset_name.capitalize()} - {config.ablation_config}')
    plt.legend(loc="lower right")
    
    # Save figure
    output_dir = Path(config.run_dir) / "figures"
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / "roc_curve.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_confusion_matrix(fold_results: List[Dict], config: Config):
    """
    Plot confusion matrix.
    
    Args:
        fold_results: List of dictionaries with results for each fold
        config: Configuration object
    """
    # Collect all predictions
    all_labels = []
    all_preds = []
    
    for result in fold_results:
        labels = result["val_labels"]
        logits = result["val_logits"]
        probabilities = F.softmax(torch.FloatTensor(logits), dim=1).numpy()
        predictions = np.argmax(probabilities, axis=1)
        
        all_labels.extend(labels)
        all_preds.extend(predictions)
    
    # Calculate confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Calculate percentages
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    # Plot
    plt.figure(figsize=(10, 8))
    
    # Setup plot
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(f'Confusion Matrix - {config.dataset_name.capitalize()} - {config.ablation_config}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks([0.5, 1.5], ['HC', 'PD'])
    plt.yticks([0.5, 1.5], ['HC', 'PD'])
    
    # Add percentages
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j+0.5, i+0.7, f"{cm_percent[i, j]:.1f}%", 
                    ha="center", va="center", color="k" if cm_percent[i, j] < 50 else "w",
                    fontsize=12)
    
    # Save plot
    plt.tight_layout()
    plt.savefig(config.run_dir / "figures" / "confusion_matrix.png", dpi=300)
    plt.close()

def plot_cross_validation_results(fold_results: List[Dict], config: Config):
    """
    Plot cross-validation results.
    
    Args:
        fold_results: List of dictionaries with results for each fold
        config: Configuration object
    """
    # Extract metrics from each fold
    fold_metrics = ["auc", "accuracy", "balanced_accuracy", "f1"]
    metric_values = {metric: [] for metric in fold_metrics}
    folds = []
    
    for result in fold_results:
        fold = result["fold"]
        folds.append(fold + 1)  # 1-indexed
        
        metrics = result["metrics"]
        for metric in fold_metrics:
            metric_values[metric].append(metrics[metric])
    
    # Create figure
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig)
    
    # Plot each metric
    for i, metric in enumerate(fold_metrics):
        row, col = i // 2, i % 2
        ax = fig.add_subplot(gs[row, col])
        
        values = metric_values[metric]
        mean_value = np.mean(values)
        std_value = np.std(values)
        
        # Bar plot
        ax.bar(folds, values, color='royalblue', alpha=0.7)
        ax.axhline(y=mean_value, color='r', linestyle='-', label=f'Mean: {mean_value:.3f}')
        ax.axhline(y=mean_value + std_value, color='r', linestyle='--', alpha=0.5, 
                  label=f'Std: {std_value:.3f}')
        ax.axhline(y=mean_value - std_value, color='r', linestyle='--', alpha=0.5)
        
        # Add value labels
        for j, value in enumerate(values):
            ax.text(j+1, value + 0.01, f"{value:.3f}", ha='center', va='bottom', fontsize=10)
        
        # Configure plot
        ax.set_title(f"{metric.replace('_', ' ').title()}", fontsize=18)
        ax.set_xlabel("Fold", fontsize=14)
        ax.set_ylabel("Score", fontsize=14)
        ax.set_ylim(0, 1.1)
        ax.set_xticks(folds)
        ax.legend(fontsize=12)
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)
    
    # Add title
    plt.suptitle(f'Cross-Validation Results - {config.dataset_name.capitalize()} - {config.ablation_config}',
                fontsize=20)
    
    # Save figure
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(config.run_dir / "figures" / "cv_results.png", dpi=300)
    plt.close()

def save_results_to_csv(fold_results: List[Dict], aggregated_results: Dict, config: Config):
    """
    Save results to CSV files.
    
    Args:
        fold_results: List of dictionaries with results for each fold
        aggregated_results: Dictionary with aggregated results
        config: Configuration object
    """
    # Save fold results
    fold_df = pd.DataFrame([
        {
            "fold": result["fold"] + 1,
            "best_epoch": result["best_epoch"] + 1,
            **{f"{metric}": value for metric, value in result["metrics"].items()}
        }
        for result in fold_results
    ])
    fold_df.to_csv(config.run_dir / "metrics" / "fold_results.csv", index=False)
    
    # Save aggregated results
    agg_df = pd.DataFrame([
        {
            "metric": metric,
            "mean": values["mean"],
            "std": values["std"],
            "ci_lower": values["ci"][0],
            "ci_upper": values["ci"][1]
        }
        for metric, values in aggregated_results.items()
    ])
    agg_df.to_csv(config.run_dir / "metrics" / "aggregated_results.csv", index=False)
    
    # Save formatted results for paper
    with open(config.run_dir / "metrics" / "formatted_results.txt", "w") as f:
        f.write(f"Results for {config.dataset_name.capitalize()} - {config.ablation_config}\n")
        f.write("=" * 60 + "\n\n")
        
        for metric, values in aggregated_results.items():
            f.write(f"{metric.replace('_', ' ').title()}: {values['formatted']}\n")

##############################################################################
# Main Execution
##############################################################################

def run_ablation_study(config: Config) -> Dict:
    """
    Run ablation study for a specific configuration.
    
    Args:
        config: Configuration object
        
    Returns:
        Dictionary with results
    """
    logging.info(f"Running ablation study for {config.dataset_name} with config {config.ablation_config}")
    
    # Load data
    data_dict = load_data(config)
    
    # Run cross-validation
    fold_results = run_cross_validation(config, data_dict)
    
    # Aggregate results
    aggregated_results = aggregate_results(fold_results)
    
    # Create predictions dataframe
    predictions_df = create_predictions_dataframe(fold_results, data_dict)
    predictions_df.to_csv(config.run_dir / "metrics" / "predictions.csv", index=False)
    
    # Plot results
    plot_roc_curve(fold_results, config)
    plot_confusion_matrix(fold_results, config)
    plot_cross_validation_results(fold_results, config)
    
    # Save results to CSV
    save_results_to_csv(fold_results, aggregated_results, config)
    
    # Log aggregated results
    logging.info(f"Aggregated results for {config.dataset_name} - {config.ablation_config}:")
    for metric, values in aggregated_results.items():
        logging.info(f"  {metric}: {values['formatted']}")
    
    return {
        "fold_results": fold_results,
        "aggregated_results": aggregated_results,
        "predictions_df": predictions_df
    }

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Ablation Analysis for IMPACT Model")
    
    parser.add_argument("--dataset", type=str, default="neurocon", choices=["neurocon", "taowu"],
                      help="Dataset to use (neurocon or taowu)")
    
    parser.add_argument("--ablation", type=str, default="full", 
                      choices=["roi", "ica", "conn", "roi+ica", "roi+conn", "ica+conn", "full"],
                      help="Ablation configuration")
    
    parser.add_argument("--epochs", type=int, default=150,
                      help="Number of epochs to train")
    
    parser.add_argument("--batch_size", type=int, default=16,
                      help="Batch size")
    
    parser.add_argument("--folds", type=int, default=5,
                      help="Number of cross-validation folds")
    
    parser.add_argument("--all", action="store_true",
                      help="Run all ablation configurations")
    
    parser.add_argument('--log_level', type=str, default='INFO', 
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], 
                        help='Set the logging level')
    parser.add_argument('--log_to_file', action='store_true', help='Enable logging to a file in the results directory')

    args = parser.parse_args()
    
    # Set up output directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(f"ablation_results_{timestamp}")
    
    # Run single configuration or all configurations
    if args.all:
        dataset_results = {}
        
        for dataset in ["neurocon", "taowu"]:
            ablation_results = {}
            
            for ablation in ["roi", "ica", "conn", "roi+ica", "roi+conn", "ica+conn", "full"]:
                # Create config
                config = Config(
                    dataset_name=dataset,
                    ablation_config=ablation,
                    results_dir=results_dir,
                    n_epochs=args.epochs,
                    batch_size=args.batch_size,
                    predefined_folds=args.folds
                )
                
                # Run ablation study
                results = run_ablation_study(config)
                ablation_results[ablation] = results
            
            dataset_results[dataset] = ablation_results
            
            # Create comparative visualization for this dataset
            compare_ablation_configs(dataset, ablation_results, results_dir)
        
        # Compare results across datasets
        compare_datasets(dataset_results, results_dir)
        
    else:
        # Create config
        config = Config(
            dataset_name=args.dataset,
            ablation_config=args.ablation,
            results_dir=results_dir,
            n_epochs=args.epochs,
            batch_size=args.batch_size,
            predefined_folds=args.folds
        )
        
        # Setup logging
        log_file = None
        if args.log_to_file:
            log_file = config.run_dir / "logs" / "run.log"
        setup_logging(args.log_level, log_file)
        
        # Run ablation study
        run_ablation_study(config)

def compare_ablation_configs(dataset: str, ablation_results: Dict, results_dir: Path):
    """
    Create comparative visualization of different ablation configurations.
    
    Args:
        dataset: Dataset name
        ablation_results: Dictionary with results for each ablation configuration
        results_dir: Directory to save results
    """
    output_dir = results_dir / f"{dataset}_comparison"
    output_dir.mkdir(exist_ok=True)
    
    # Extract mean and CI for each metric and configuration
    metrics = ["auc", "accuracy", "balanced_accuracy", "f1"]
    config_names = list(ablation_results.keys())
    
    # Data for plotting
    plot_data = {metric: {
        "means": [],
        "errors": [],
        "ci_lower": [],
        "ci_upper": []
    } for metric in metrics}
    
    # Fill data
    for config_name in config_names:
        results = ablation_results[config_name]
        agg_results = results["aggregated_results"]
        
        for metric in metrics:
            plot_data[metric]["means"].append(agg_results[metric]["mean"])
            plot_data[metric]["errors"].append(agg_results[metric]["std"])
            plot_data[metric]["ci_lower"].append(agg_results[metric]["ci"][0])
            plot_data[metric]["ci_upper"].append(agg_results[metric]["ci"][1])
    
    # Display names for configurations
    config_display_names = [
        "ROI only", 
        "ICA only", 
        "Connectivity only",
        "ROI + ICA",
        "ROI + Connectivity",
        "ICA + Connectivity",
        "Full Model"
    ]
    
    config_display_map = {
        "roi": "ROI only",
        "ica": "ICA only",
        "conn": "Connectivity only",
        "roi+ica": "ROI + ICA",
        "roi+conn": "ROI + Connectivity",
        "ica+conn": "ICA + Connectivity",
        "full": "Full Model"
    }
    
    display_names = [config_display_map[cn] for cn in config_names]
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    axes = axes.flatten()
    
    # Plot each metric
    for i, metric in enumerate(metrics):
        row, col = i // 2, i % 2
        ax = fig.add_subplot(axes[i])  # Correctly access the subplot
        means = plot_data[metric]["means"]
        errors = plot_data[metric]["errors"]
        ci_lower = plot_data[metric]["ci_lower"]
        ci_upper = plot_data[metric]["ci_upper"]
        
        # Calculate error bars (asymmetric for CI)
        lower_err = [m - l for m, l in zip(means, ci_lower)]
        upper_err = [u - m for m, u in zip(means, ci_upper)]
        
        # Bar plot with error bars
        bars = ax.bar(display_names, means, yerr=[lower_err, upper_err], 
                     capsize=10, color='royalblue', alpha=0.7)
        
        # Add value labels
        for j, value in enumerate(means):
            ax.text(j+1, value + 0.01, f"{value:.3f}", ha='center', va='bottom', fontsize=12)
        
        # Add horizontal line for best performance
        best_idx = np.argmax(means)
        ax.axhline(y=means[best_idx], color='r', linestyle='--', alpha=0.5,
                  label=f'Best: {display_names[best_idx]} ({means[best_idx]:.3f})')
        
        # Configure plot
        ax.set_title(f"{metric.replace('_', ' ').title()}", fontsize=18)
        ax.set_xlabel("Configuration", fontsize=14)
        ax.set_ylabel("Score", fontsize=14)
        ax.set_ylim(0, 1.1)
        ax.set_xticks(range(len(display_names)))
        ax.set_xticklabels(display_names, rotation=45)
        ax.legend(fontsize=12)
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)
    
    # Add title
    plt.suptitle(f'Ablation Study Results - {dataset.capitalize()} Dataset',
                fontsize=24)
    
    # Save figure
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_dir / "ablation_comparison.png", dpi=300)
    plt.close()
    
    # Create summary CSV
    summary_data = []
    for config_name, display_name in zip(config_names, display_names):
        results = ablation_results[config_name]
        agg_results = results["aggregated_results"]
        
        row = {"configuration": display_name}
        for metric in metrics:
            row[f"{metric}_mean"] = agg_results[metric]["mean"]
            row[f"{metric}_std"] = agg_results[metric]["std"]
            row[f"{metric}_ci_lower"] = agg_results[metric]["ci"][0]
            row[f"{metric}_ci_upper"] = agg_results[metric]["ci"][1]
            row[f"{metric}_formatted"] = agg_results[metric]["formatted"]
        
        summary_data.append(row)
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(output_dir / "ablation_summary.csv", index=False)
    
    # Create formatted summary for the paper
    with open(output_dir / "ablation_summary.txt", "w") as f:
        f.write(f"Ablation Study Results - {dataset.capitalize()} Dataset\n")
        f.write("=" * 80 + "\n\n")
        
        for config_name, display_name in zip(config_names, display_names):
            results = ablation_results[config_name]
            agg_results = results["aggregated_results"]
            
            f.write(f"{display_name}:\n")
            for metric in metrics:
                f.write(f"  {metric.replace('_', ' ').title()}: {agg_results[metric]['formatted']}\n")
            f.write("\n")

def compare_datasets(dataset_results: Dict, results_dir: Path):
    """
    Create comparative visualization across datasets.
    
    Args:
        dataset_results: Dictionary with results for each dataset
        results_dir: Directory to save results
    """
    output_dir = results_dir / "dataset_comparison"
    output_dir.mkdir(exist_ok=True)
    
    # Create formatted summary for the paper
    with open(output_dir / "dataset_comparison.txt", "w") as f:
        f.write("Ablation Study Results - Dataset Comparison\n")
        f.write("=" * 80 + "\n\n")
        
        for dataset, ablation_results in dataset_results.items():
            f.write(f"{dataset.capitalize()} Dataset:\n")
            f.write("-" * 50 + "\n")
            
            for ablation, results in ablation_results.items():
                agg_results = results["aggregated_results"]
                
                f.write(f"  {ablation}:\n")
                for metric in ["auc", "accuracy"]:
                    f.write(f"    {metric}: {agg_results[metric]['formatted']}\n")
                f.write("\n")

if __name__ == "__main__":
    main()
