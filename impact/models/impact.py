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

class IMPACTModel(nn.Module):
    """Main IMPACT model."""
    
    def __init__(
        self,
        roi_dim: int,
        ica_dim: int,
        n_classes: int = 2,
        embed_dim: int = 256,
        n_heads: int = 4,
        n_layers: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.roi_dim = roi_dim
        self.ica_dim = ica_dim
        self.embed_dim = embed_dim
        
        # Input projections
        self.roi_proj = nn.Linear(roi_dim, embed_dim)
        self.ica_proj = nn.Linear(ica_dim, embed_dim)
        self.conn_proj = nn.Sequential(
            nn.Linear(roi_dim * roi_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim)
        )
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(embed_dim)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                embed_dim,
                n_heads,
                embed_dim * 4,
                dropout
            ) for _ in range(n_layers)
        ])
        
        # Cross-modal fusion
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim)
        )
        
        # Output layers
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, n_classes)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights using Xavier uniform."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def forward(
        self,
        batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass of IMPACT model.
        
        Args:
            batch: Dictionary containing:
                - roi: ROI time series (batch_size, seq_len, roi_dim)
                - ica: ICA time series (batch_size, seq_len, ica_dim)
                - conn: Connectivity matrices (batch_size, n_windows, roi_dim, roi_dim)
        
        Returns:
            logits: Classification logits
            attention_weights: Dictionary of attention weights from each modality
        """
        roi_data = batch['roi']
        ica_data = batch['ica']
        conn_data = batch['conn']
        
        batch_size = roi_data.size(0)
        
        # Project inputs
        roi = self.roi_proj(roi_data)
        ica = self.ica_proj(ica_data)
        
        # Flatten and project connectivity matrices
        conn = conn_data.view(batch_size, -1, self.roi_dim * self.roi_dim)
        conn = self.conn_proj(conn)
        
        # Add positional encoding
        roi = self.pos_encoder(roi)
        ica = self.pos_encoder(ica)
        conn = self.pos_encoder(conn)
        
        # Process each modality through transformer
        attention_weights = {'roi': [], 'ica': [], 'conn': []}
        
        for layer in self.layers:
            roi, roi_attn = layer(roi)
            ica, ica_attn = layer(ica)
            conn, conn_attn = layer(conn)
            
            attention_weights['roi'].append(roi_attn)
            attention_weights['ica'].append(ica_attn)
            attention_weights['conn'].append(conn_attn)
        
        # Global pooling
        roi = roi.mean(dim=1)
        ica = ica.mean(dim=1)
        conn = conn.mean(dim=1)
        
        # Fuse modalities
        fused = torch.cat([roi, ica, conn], dim=-1)
        fused = self.fusion(fused)
        
        # Classification
        logits = self.classifier(fused)
        
        return logits, attention_weights
