"""
IMPACT fMRI Data Preprocessor

This module handles the preprocessing of fMRI data for the IMPACT pipeline, including:
- ROI time series extraction using the Harvard-Oxford atlas
- ICA component extraction
- Dynamic connectivity computation
- Quality metrics calculation
"""

import nibabel as nib
import numpy as np
from nilearn import datasets, masking
from nilearn.maskers import NiftiLabelsMasker
from nilearn.decomposition import CanICA
from scipy import signal
from pathlib import Path
import json
import warnings
from typing import Dict, List, Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FMRIPreprocessor:
    def __init__(
        self,
        atlas: str = 'harvard-oxford',
        n_ica_components: int = 5,
        window_size: int = 50,
        window_stride: int = 25,
        standardize: bool = True,
        high_pass: Optional[float] = None,
        low_pass: Optional[float] = None
    ):
        """
        Initialize the fMRI preprocessor.
        
        Args:
            atlas: Atlas to use for ROI extraction ('harvard-oxford' or path to custom atlas)
            n_ica_components: Number of ICA components to extract
            window_size: Size of sliding window for dynamic connectivity (in TRs)
            window_stride: Stride of sliding window (in TRs)
            standardize: Whether to standardize the time series
            high_pass: High-pass filter cutoff frequency (in Hz)
            low_pass: Low-pass filter cutoff frequency (in Hz)
        """
        self.n_ica_components = n_ica_components
        self.window_size = window_size
        self.window_stride = window_stride
        self.standardize = standardize
        self.high_pass = high_pass
        self.low_pass = low_pass
        
        # Load atlas
        if atlas == 'harvard-oxford':
            self.atlas = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
            self.roi_labels = [region for region in self.atlas.labels if region]
        else:
            # Load custom atlas
            self.atlas = nib.load(atlas)
            with open(Path(atlas).with_suffix('.json'), 'r') as f:
                self.roi_labels = json.load(f)['labels']
                
        # Initialize masker
        self.masker = NiftiLabelsMasker(
            labels_img=self.atlas.maps,
            standardize=standardize,
            high_pass=high_pass,
            low_pass=low_pass,
            memory='nilearn_cache',
            verbose=0
        )
        
        # Initialize ICA
        self.canica = CanICA(
            n_components=n_ica_components,
            random_state=42,
            standardize=standardize,
            memory='nilearn_cache',
            verbose=0
        )
        
    def extract_roi_timeseries(self, fmri_img: nib.Nifti1Image) -> np.ndarray:
        """Extract ROI time series from fMRI image."""
        return self.masker.fit_transform(fmri_img)
    
    def extract_ica_timeseries(self, fmri_img: nib.Nifti1Image) -> np.ndarray:
        """Extract ICA component time series from fMRI image."""
        try:
            return self.canica.fit_transform([fmri_img])[0]
        except Exception as e:
            logger.warning(f"ICA extraction failed: {str(e)}")
            # Return zero-filled array of correct shape
            roi_ts = self.extract_roi_timeseries(fmri_img)
            return np.zeros((roi_ts.shape[0], self.n_ica_components))
    
    def compute_dynamic_connectivity(
        self,
        timeseries: np.ndarray,
        tr: float
    ) -> Tuple[np.ndarray, List[Dict]]:
        """
        Compute dynamic connectivity matrices using sliding windows.
        
        Args:
            timeseries: Time series data of shape (timepoints, regions)
            tr: Repetition time in seconds
        
        Returns:
            windows: Array of connectivity matrices for each window
            window_info: List of dictionaries with window metadata
        """
        n_timepoints, n_regions = timeseries.shape
        n_windows = (n_timepoints - self.window_size) // self.window_stride + 1
        
        windows = np.zeros((n_windows, n_regions, n_regions))
        window_info = []
        
        for i in range(n_windows):
            start = i * self.window_stride
            end = start + self.window_size
            
            # Extract window
            window = timeseries[start:end]
            
            # Compute correlation matrix
            corr_matrix = np.corrcoef(window.T)
            windows[i] = corr_matrix
            
            # Store window metadata
            window_info.append({
                'start_tr': start,
                'end_tr': end,
                'start_time': start * tr,
                'end_time': end * tr
            })
        
        return windows, window_info
    
    def compute_quality_metrics(
        self,
        timeseries: np.ndarray,
        tr: float
    ) -> Dict[str, np.ndarray]:
        """
        Compute quality metrics for time series data.
        
        Args:
            timeseries: Time series data of shape (timepoints, regions)
            tr: Repetition time in seconds
            
        Returns:
            Dictionary of quality metrics
        """
        # Basic signal properties
        signal_power = np.mean(np.abs(timeseries), axis=0)
        noise_power = np.std(timeseries, axis=0)
        snr = np.where(noise_power > 0, signal_power / noise_power, 0)
        
        # Spectral properties
        freqs, psd = signal.welch(timeseries, fs=1/tr, nperseg=min(64, len(timeseries)))
        temporal_snr = np.mean(psd, axis=0)
        
        return {
            'signal_power': signal_power,
            'noise_power': noise_power,
            'snr': snr,
            'temporal_snr': temporal_snr,
            'psd_freqs': freqs,
            'psd': psd
        }
    
    def process_subject(
        self,
        fmri_path: str,
        tr: Optional[float] = None
    ) -> Dict:
        """
        Process a single subject's fMRI data.
        
        Args:
            fmri_path: Path to fMRI NIfTI file
            tr: Repetition time in seconds (if None, read from NIfTI header)
            
        Returns:
            Dictionary containing processed data and metadata
        """
        # Load fMRI data
        fmri_img = nib.load(fmri_path)
        if tr is None:
            tr = fmri_img.header.get_zooms()[-1]
        
        # Extract time series
        roi_ts = self.extract_roi_timeseries(fmri_img)
        ica_ts = self.extract_ica_timeseries(fmri_img)
        
        # Compute dynamic connectivity
        conn_matrices, window_info = self.compute_dynamic_connectivity(roi_ts, tr)
        
        # Compute quality metrics
        roi_metrics = self.compute_quality_metrics(roi_ts, tr)
        ica_metrics = self.compute_quality_metrics(ica_ts, tr)
        
        return {
            'roi_timeseries': roi_ts,
            'ica_timeseries': ica_ts,
            'connectivity_matrices': conn_matrices,
            'window_info': window_info,
            'roi_metrics': roi_metrics,
            'ica_metrics': ica_metrics,
            'metadata': {
                'tr': tr,
                'n_timepoints': len(roi_ts),
                'n_rois': len(self.roi_labels),
                'roi_labels': self.roi_labels,
                'n_ica_components': self.n_ica_components
            }
        }
