# IMPACT: Integrative Multimodal Pipeline for Advanced Connectivity and Time-series

## Overview

IMPACT is a transformer-based framework for analyzing resting-state fMRI data in Parkinson's Disease (PD). Unlike traditional approaches that treat fMRI data as static connectivity maps, IMPACT views brain activity as a dynamic sequence of states, enabling the capture of subtle temporal patterns that may be indicative of early disease processes.

### Key Innovations

1. **Dynamic Temporal Processing**: Captures transient disruptions in functional connectivity that are often averaged out in traditional static analyses.
2. **Multi-scale Integration**: Combines information across different temporal scales and spatial organizations of brain activity.
3. **Interpretable Architecture**: Provides insights into which brain regions and time periods contribute most to the model's decisions.

## Background

Parkinson's Disease diagnosis currently relies heavily on motor symptoms, which typically appear after significant neurodegeneration has occurred. Early detection through neuroimaging biomarkers could enable earlier intervention and better patient outcomes. IMPACT addresses this challenge by:

- Analyzing dynamic functional connectivity patterns in resting-state fMRI
- Integrating multiple streams of information (ROI time series, ICA components, connectivity matrices)
- Using attention mechanisms to identify disease-relevant temporal patterns
- Providing interpretable results that align with clinical understanding of PD

## Features

### Data Processing
- **Dataset Agnostic Processing**: 
  - Flexible data loading system supporting multiple dataset formats
  - Built-in support for Neurocon and Tao Wu datasets
  - Extensible design for adding new dataset loaders
  - Automated quality control and preprocessing

### Analysis Pipeline
- **Multimodal Integration**: 
  - Regional time courses from anatomically defined brain areas
  - ICA-derived network patterns
  - Time-varying functional connectivity estimates
  - Dynamic state analysis

### Model Architecture
- **Advanced Transformer Design**:
  - Multi-head attention for parallel feature processing
  - Dynamic temporal convolution blocks
  - Cross-modal fusion mechanism
  - Adaptive gating for temporal feature selection

### Visualization & Analysis
- **Comprehensive Visualization Suite**:
  - GradCAM visualization for feature importance
  - Attention weight analysis
  - Network-level statistical analysis
  - Dynamic connectivity visualization
  - ROI importance mapping

## Repository Structure

```
IMPACT/
├── impact/                      # Main package directory
│   ├── data/                   # Data processing modules
│   │   ├── processor.py        # Core fMRI processing logic
│   │   └── loaders.py         # Dataset-specific loaders
│   ├── models/                 # Model architectures
│   │   └── impact.py          # Main IMPACT model implementation
│   ├── training/              # Training infrastructure
│   │   └── trainer.py         # Training loop and utilities
│   ├── utils/                 # Utility functions
│   │   ├── visualization.py   # Visualization tools
│   │   └── metrics.py        # Evaluation metrics
│   ├── evaluate.py            # Evaluation scripts
│   └── train.py              # Training scripts
├── examples/                   # Usage examples
│   ├── train_model.py        # Example training script
│   └── process_datasets.py    # Data processing example
├── docs/                      # Documentation
│   ├── installation.md       # Installation guide
│   ├── usage.md             # Usage documentation
│   └── api/                 # API documentation
├── notebooks/                 # Jupyter notebooks
│   ├── demo.ipynb           # Demo notebook
│   └── analysis.ipynb       # Analysis examples
├── tests/                    # Unit tests
├── setup.py                  # Package installation
├── requirements.txt          # Dependencies
└── README.md                 # This file
```

## Installation

```bash
# Clone the repository
git clone https://github.com/salilp42/IMPACT.git
cd IMPACT

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install the package
pip install -e .
```

## Quick Start

### Processing a Dataset

```python
from impact.data.loaders import TaoWuLoader, NeuroconLoader

# For Tao Wu dataset
loader = TaoWuLoader(
    base_dir="/path/to/taowu/data",
    output_dir="/path/to/output"
)
loader.process_dataset()
```

### Training a Model

```python
# Example command-line usage
python -m examples.train_model \
    --dataset taowu \
    --data_dir /path/to/data \
    --processed_dir /path/to/processed \
    --output_dir /path/to/results \
    --visualize
```

## Extending IMPACT

### Adding Support for New Datasets

Create a new loader class that inherits from `BaseDatasetLoader`:

```python
from impact.data.loaders import BaseDatasetLoader

class CustomDatasetLoader(BaseDatasetLoader):
    def find_subjects(self):
        subjects = []
        # Implement dataset-specific logic
        return subjects
```

### Customizing the Model

The model architecture is modular and can be customized:
- Modify the temporal processing blocks in `DynamicTemporalBlock`
- Adjust the attention mechanism in `MultiHeadAttention`
- Add new feature streams to `IMPACTModel`

## Performance

IMPACT achieves high performance in PD detection:
- Neurocon dataset: AUC 0.977 (95% CI: 0.931–1.000)
- Tao Wu dataset: AUC 0.973 (95% CI: 0.924–0.994)

## Citation

If you use IMPACT in your research, please cite:

```bibtex
@article{impact2024,
  title={IMPACT: A Transformer-based Framework for Dynamic fMRI Analysis in Parkinson's Disease},
  author={Salil Patel},
  journal={},
  year={2024}
}
```

## Contributing

We welcome contributions! Please see our [contributing guidelines](CONTRIBUTING.md) for details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

