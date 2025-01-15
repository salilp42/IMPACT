"""
IMPACT Model Architecture

This module defines the IMPACT model architecture, including:
- Multi-head self-attention layers
- Cross-modal fusion mechanism
- Positional encoding
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Tuple, Optional

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register buffer (not a parameter, but should be saved and moved with model)
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input."""
        return x + self.pe[:, :x.size(1)]

class MultiHeadAttention(nn.Module):
    """Multi-head self-attention layer."""
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        bias: bool = True
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.scaling = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        self.dropout_module = nn.Dropout(dropout)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Multi-head attention forward pass.
        
        Args:
            query: Query tensor (batch_size, tgt_len, embed_dim)
            key: Key tensor (batch_size, src_len, embed_dim)
            value: Value tensor (batch_size, src_len, embed_dim)
            key_padding_mask: Mask for padded elements (batch_size, src_len)
            attn_mask: Attention mask (tgt_len, src_len)
            
        Returns:
            output: Attention output
            attention: Attention weights
        """
        batch_size, tgt_len, embed_dim = query.size()
        src_len = key.size(1)
        
        scaling = self.scaling
        
        # Project queries, keys, and values
        q = self.q_proj(query) * scaling
        k = self.k_proj(key)
        v = self.v_proj(value)
        
        # Reshape for multi-head attention
        q = q.contiguous().view(batch_size, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.contiguous().view(batch_size, src_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.contiguous().view(batch_size, src_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Calculate attention scores
        attn_weights = torch.matmul(q, k.transpose(-2, -1))
        
        # Apply masks if provided
        if attn_mask is not None:
            attn_weights = attn_weights.masked_fill(
                attn_mask.unsqueeze(0).unsqueeze(0),
                float('-inf')
            )
        
        if key_padding_mask is not None:
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float('-inf')
            )
        
        # Apply softmax and dropout
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_probs = self.dropout_module(attn_weights)
        
        # Calculate output
        attn = torch.matmul(attn_probs, v)
        
        # Reshape and project output
        attn = attn.transpose(1, 2).contiguous().view(batch_size, tgt_len, embed_dim)
        output = self.out_proj(attn)
        
        return output, attn_weights

class TransformerEncoderLayer(nn.Module):
    """Transformer encoder layer."""
    
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout=dropout)
        
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        self.activation = F.gelu
    
    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of transformer encoder layer.
        
        Returns:
            output: Layer output
            attention: Self-attention weights
        """
        # Self attention
        src2, attention = self.self_attn(
            src, src, src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask
        )
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        
        # Feedforward
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        
        return src, attention

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
        x_conv = x.transpose(1, 2)
        conv_outputs = []
        for conv in self.convs:
            conv_out = conv(x_conv)
            conv_outputs.append(conv_out.transpose(1, 2))
        
        multi_scale = torch.cat(conv_outputs, dim=-1)
        gate_weights = self.gate(multi_scale)
        
        output = sum(gate_weights * conv_out for conv_out in conv_outputs)
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = 0.5 * (x + x.transpose(-2,-1))
        x = x.unsqueeze(1)

        x = self.bn1(F.gelu(self.conv1(x)))
        x = F.dropout(x, p=0.2, training=self.training)

        x = self.bn2(F.gelu(self.conv2(x)))
        x = F.dropout(x, p=0.2, training=self.training)

        x = self.bn3(F.gelu(self.conv3(x)))
        x = self.gap(x)
        
        return self.proj(x)

class MultiHeadAttention(nn.Module):
    """Multi-head attention with linear projections."""
    
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def forward(self, 
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = query.shape[0]
        
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        
        q = q.reshape(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, -1, self.embed_dim)
        
        return self.out_proj(attn_output), attn_weights

class TransformerEncoder(nn.Module):
    """Transformer encoder with multi-head attention and feed-forward layers."""
    
    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        attn_output, attn_weights = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        x = self.norm2(x + self.dropout(self.ff(x)))
        return x, attn_weights

class IMPACTModel(nn.Module):
    """
    IMPACT: Integrative Multimodal Pipeline for Advanced Connectivity and Time-series.
    A transformer-based model for analyzing fMRI time series data.
    """
    
    def __init__(self,
                 roi_dim: int,
                 ica_dim: int,
                 roi_hidden_dim: int = 256,
                 ica_hidden_dim: int = 256,
                 n_heads: int = 8,
                 n_layers: int = 2,
                 dropout: float = 0.12):
        super().__init__()
        
        # ROI stream
        self.roi_temporal = DynamicTemporalBlock(roi_dim, roi_hidden_dim)
        self.roi_corr = CorrelationEncoder(roi_dim, roi_hidden_dim)
        
        # ICA stream
        self.ica_temporal = DynamicTemporalBlock(ica_dim, ica_hidden_dim)
        
        # Transformer layers
        self.transformers = nn.ModuleList([
            TransformerEncoder(roi_hidden_dim + ica_hidden_dim, n_heads, dropout=dropout)
            for _ in range(n_layers)
        ])
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(roi_hidden_dim + ica_hidden_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 2)
        )
        
        self.attention_weights = None
    
    def forward(self, roi_ts: torch.Tensor, ica_ts: torch.Tensor, corr: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Process ROI time series
        roi_temp = self.roi_temporal(roi_ts)
        roi_corr = self.roi_corr(corr).unsqueeze(1).expand(-1, roi_temp.size(1), -1)
        roi_features = roi_temp + roi_corr
        
        # Process ICA time series
        ica_features = self.ica_temporal(ica_ts)
        
        # Combine features
        features = torch.cat([roi_features, ica_features], dim=-1)
        
        # Apply transformer layers
        self.attention_weights = []
        for transformer in self.transformers:
            features, attn = transformer(features)
            self.attention_weights.append(attn)
        
        # Global average pooling and classification
        features = features.mean(dim=1)
        logits = self.classifier(features)
        
        return logits, self.attention_weights
