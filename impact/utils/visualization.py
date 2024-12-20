"""
Visualization utilities for IMPACT model.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Union
import networkx as nx
from nilearn import plotting, image
import nibabel as nib
from scipy.stats import zscore
import pandas as pd

def plot_roi_timeseries(
    timeseries: np.ndarray,
    roi_labels: List[str],
    output_path: Optional[str] = None,
    title: str = 'ROI Time Series'
) -> None:
    """Plot ROI time series data."""
    plt.figure(figsize=(15, 8))
    for i in range(min(5, timeseries.shape[1])):  # Plot first 5 ROIs
        plt.plot(timeseries[:, i], label=roi_labels[i])
    
    plt.title(title)
    plt.xlabel('Time (TR)')
    plt.ylabel('BOLD Signal')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        plt.close()
    else:
        plt.show()

def plot_connectivity_matrix(
    conn_matrix: np.ndarray,
    roi_labels: List[str],
    output_path: Optional[str] = None,
    title: str = 'Connectivity Matrix'
) -> None:
    """Plot connectivity matrix as heatmap."""
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(conn_matrix, dtype=bool))
    
    sns.heatmap(
        conn_matrix,
        mask=mask,
        cmap='RdBu_r',
        center=0,
        square=True,
        xticklabels=roi_labels,
        yticklabels=roi_labels,
        cbar_kws={'label': 'Correlation'}
    )
    
    plt.title(title)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        plt.close()
    else:
        plt.show()

def plot_network_graph(
    conn_matrix: np.ndarray,
    roi_labels: List[str],
    threshold: float = 0.5,
    output_path: Optional[str] = None,
    title: str = 'Brain Network Graph'
) -> None:
    """Plot brain network as a graph."""
    # Create graph
    G = nx.Graph()
    
    # Add nodes
    for i, label in enumerate(roi_labels):
        G.add_node(i, label=label)
    
    # Add edges above threshold
    for i in range(len(roi_labels)):
        for j in range(i+1, len(roi_labels)):
            if abs(conn_matrix[i, j]) > threshold:
                G.add_edge(i, j, weight=conn_matrix[i, j])
    
    plt.figure(figsize=(12, 12))
    pos = nx.spring_layout(G)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=500, node_color='lightblue')
    
    # Draw edges with weights determining color
    edges = G.edges()
    weights = [G[u][v]['weight'] for u, v in edges]
    nx.draw_networkx_edges(
        G, pos,
        edgelist=edges,
        edge_color=weights,
        edge_cmap=plt.cm.RdBu_r,
        edge_vmin=-1,
        edge_vmax=1
    )
    
    # Add labels
    labels = nx.get_node_attributes(G, 'label')
    nx.draw_networkx_labels(G, pos, labels)
    
    plt.title(title)
    plt.colorbar(plt.cm.ScalarMappable(cmap=plt.cm.RdBu_r), label='Correlation')
    plt.axis('off')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        plt.close()
    else:
        plt.show()

def plot_attention_brain(
    attention_weights: np.ndarray,
    atlas: nib.Nifti1Image,
    output_path: Optional[str] = None,
    title: str = 'Brain Attention Map'
) -> None:
    """Plot attention weights on brain surface."""
    # Create brain map
    attention_map = np.zeros(atlas.shape)
    unique_labels = np.unique(atlas.get_fdata())[1:]  # Skip background
    
    for i, label in enumerate(unique_labels):
        mask = atlas.get_fdata() == label
        attention_map[mask] = attention_weights[i]
    
    attention_img = nib.Nifti1Image(attention_map, atlas.affine)
    
    # Plot on glass brain
    display = plotting.plot_glass_brain(None, title=title)
    display.add_overlay(attention_img, colorbar=True)
    
    if output_path:
        display.savefig(output_path)
        display.close()
    else:
        display.show()

def plot_dynamic_connectivity(
    conn_matrices: np.ndarray,
    window_info: List[Dict],
    roi_labels: List[str],
    output_path: Optional[str] = None,
    title: str = 'Dynamic Connectivity'
) -> None:
    """Plot dynamic connectivity patterns."""
    n_windows = len(conn_matrices)
    n_rois = len(roi_labels)
    
    # Compute mean connectivity over time
    mean_conn = np.mean(conn_matrices, axis=0)
    
    # Compute temporal variability
    std_conn = np.std(conn_matrices, axis=0)
    
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    # Plot mean connectivity
    sns.heatmap(
        mean_conn,
        ax=axes[0],
        cmap='RdBu_r',
        center=0,
        square=True,
        xticklabels=roi_labels,
        yticklabels=roi_labels,
        cbar_kws={'label': 'Mean Correlation'}
    )
    axes[0].set_title('Mean Connectivity')
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=90)
    axes[0].set_yticklabels(axes[0].get_yticklabels(), rotation=0)
    
    # Plot temporal variability
    sns.heatmap(
        std_conn,
        ax=axes[1],
        cmap='viridis',
        square=True,
        xticklabels=roi_labels,
        yticklabels=roi_labels,
        cbar_kws={'label': 'Standard Deviation'}
    )
    axes[1].set_title('Temporal Variability')
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=90)
    axes[1].set_yticklabels(axes[1].get_yticklabels(), rotation=0)
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        plt.close()
    else:
        plt.show()

def plot_quality_metrics(
    metrics: Dict[str, np.ndarray],
    roi_labels: List[str],
    output_path: Optional[str] = None
) -> None:
    """Plot quality metrics for ROIs."""
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    
    # SNR plot
    sns.barplot(
        x=roi_labels,
        y=metrics['snr'],
        ax=axes[0, 0]
    )
    axes[0, 0].set_title('Signal-to-Noise Ratio')
    axes[0, 0].set_xticklabels(axes[0, 0].get_xticklabels(), rotation=90)
    
    # Signal power plot
    sns.barplot(
        x=roi_labels,
        y=metrics['signal_power'],
        ax=axes[0, 1]
    )
    axes[0, 1].set_title('Signal Power')
    axes[0, 1].set_xticklabels(axes[0, 1].get_xticklabels(), rotation=90)
    
    # Temporal SNR plot
    sns.barplot(
        x=roi_labels,
        y=metrics['temporal_snr'],
        ax=axes[1, 0]
    )
    axes[1, 0].set_title('Temporal SNR')
    axes[1, 0].set_xticklabels(axes[1, 0].get_xticklabels(), rotation=90)
    
    # Power spectral density plot
    for i in range(min(5, len(roi_labels))):  # Plot first 5 ROIs
        axes[1, 1].plot(
            metrics['psd_freqs'],
            metrics['psd'][:, i],
            label=roi_labels[i]
        )
    axes[1, 1].set_title('Power Spectral Density')
    axes[1, 1].set_xlabel('Frequency (Hz)')
    axes[1, 1].set_ylabel('Power')
    axes[1, 1].legend()
    
    plt.suptitle('ROI Quality Metrics')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        plt.close()
    else:
        plt.show()

def create_visualization_report(
    subject_data: Dict,
    output_dir: Union[str, Path],
    prefix: str = ''
) -> None:
    """
    Create comprehensive visualization report for a subject.
    
    Args:
        subject_data: Dictionary containing processed subject data
        output_dir: Directory to save visualizations
        prefix: Prefix for output filenames
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. ROI time series
    plot_roi_timeseries(
        subject_data['roi_timeseries'],
        subject_data['metadata']['roi_labels'],
        output_dir / f'{prefix}roi_timeseries.png'
    )
    
    # 2. Static connectivity
    mean_conn = np.mean(subject_data['connectivity_matrices'], axis=0)
    plot_connectivity_matrix(
        mean_conn,
        subject_data['metadata']['roi_labels'],
        output_dir / f'{prefix}connectivity_matrix.png'
    )
    
    # 3. Network graph
    plot_network_graph(
        mean_conn,
        subject_data['metadata']['roi_labels'],
        output_dir / f'{prefix}network_graph.png'
    )
    
    # 4. Dynamic connectivity
    plot_dynamic_connectivity(
        subject_data['connectivity_matrices'],
        subject_data['window_info'],
        subject_data['metadata']['roi_labels'],
        output_dir / f'{prefix}dynamic_connectivity.png'
    )
    
    # 5. Quality metrics
    plot_quality_metrics(
        subject_data['roi_metrics'],
        subject_data['metadata']['roi_labels'],
        output_dir / f'{prefix}quality_metrics.png'
    )

def plot_group_differences(
    group1_data: np.ndarray,
    group2_data: np.ndarray,
    roi_labels: List[str],
    group_labels: List[str],
    output_path: Optional[str] = None,
    title: str = 'Group Differences'
) -> None:
    """Plot differences between two groups."""
    # Compute statistics
    t_stats = []
    p_values = []
    for i in range(group1_data.shape[1]):
        t_stat, p_val = scipy.stats.ttest_ind(
            group1_data[:, i],
            group2_data[:, i]
        )
        t_stats.append(t_stat)
        p_values.append(p_val)
    
    # Create plot
    plt.figure(figsize=(15, 6))
    x = np.arange(len(roi_labels))
    width = 0.35
    
    plt.bar(x - width/2, np.mean(group1_data, axis=0), width,
           label=group_labels[0], yerr=np.std(group1_data, axis=0))
    plt.bar(x + width/2, np.mean(group2_data, axis=0), width,
           label=group_labels[1], yerr=np.std(group2_data, axis=0))
    
    # Add significance markers
    sig_height = max(np.max(group1_data), np.max(group2_data)) * 1.1
    for i, p in enumerate(p_values):
        if p < 0.05:
            plt.text(i, sig_height, '*', ha='center')
        if p < 0.01:
            plt.text(i, sig_height*1.05, '*', ha='center')
        if p < 0.001:
            plt.text(i, sig_height*1.1, '*', ha='center')
    
    plt.xlabel('ROIs')
    plt.ylabel('Value')
    plt.title(title)
    plt.xticks(x, roi_labels, rotation=90)
    plt.legend()
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        plt.close()
    else:
        plt.show()
