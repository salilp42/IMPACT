# IMPACT: Integrative Multimodal Pipeline for Advanced Connectivity and Timeseries

IMPACT is a transformer-based framework for analyzing resting-state fMRI data in Parkinson's Disease (PD). It treats fMRI data as a time-evolving sequence rather than a static map, enabling the capture of subtle temporal patterns that may be indicative of early disease processes.

## Features

- **Multimodal Integration**: Combines multiple feature streams including:
  - Regional time courses from anatomically defined brain areas
  - ICA-derived network patterns
  - Time-varying functional connectivity estimates

- **Advanced Architecture**: 
  - Transformer-based design for capturing long-range temporal dependencies
  - Selective attention to informative time intervals and brain regions
  - Multi-head attention mechanism for parallel feature processing

- **Interpretability**:
  - GradCAM visualization for feature importance
  - Attention weight analysis
  - Network-level statistical analysis

## Dataset

This project uses the [Neurocon dataset](https://fcon_1000.projects.nitrc.org/indi/retro/parkinsons.html) from NITRC, which includes resting-state fMRI data from Parkinson's Disease patients and healthy controls. To use this dataset:

1. Visit the [NITRC website](https://fcon_1000.projects.nitrc.org/indi/retro/parkinsons.html)
2. Register and accept the data usage agreement
3. Download the dataset
4. Organize the data as described in the example notebook

## Installation

```bash
# Clone the repository
git clone https://github.com/salilp42/IMPACT.git
cd IMPACT

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Project Structure

```
IMPACT/
├── impact/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py        # Data loading utilities
│   │   └── preprocessor.py  # Data preprocessing pipeline
│   ├── models/
│   │   ├── __init__.py
│   │   └── impact.py        # Main IMPACT model
│   └── utils/
│       ├── __init__.py
│       ├── visualization.py  # Plotting utilities
│       └── metrics.py       # Evaluation metrics
├── notebooks/
│   └── example.ipynb        # Usage examples
├── tests/
│   └── __init__.py
├── requirements.txt
└── README.md
```

## Usage

See `notebooks/example.ipynb` for a complete example. Here's a quick overview:

```python
from impact.data.loader import IMPACTDataLoader
from impact.models.impact import IMPACTModel

# Load and preprocess data
loader = IMPACTDataLoader(data_path="path/to/data")
train_loader, val_loader, test_loader = loader.get_dataloaders()

# Initialize and train model
model = IMPACTModel(
    roi_dim=train_loader.dataset.roi_data.size(-1),
    ica_dim=train_loader.dataset.ica_data.size(-1),
    embed_dim=256,
    n_heads=4,
    n_layers=3
)

# Train model (see example notebook for full training loop)
model.train()
...

# Make predictions
model.eval()
with torch.no_grad():
    logits, attention = model(inputs)
    predictions = torch.argmax(logits, dim=1)
```

## Training Scripts

The project includes two main scripts for training and evaluation:

1. `impact/train.py`: Main training script with the following features:
   - Command-line argument parsing
   - TensorBoard logging
   - Early stopping
   - Model checkpointing
   - Learning rate scheduling

2. `impact/evaluate.py`: Evaluation script that provides:
   - ROC curve plotting
   - Attention weight visualization
   - Brain importance mapping
   - Comprehensive metric computation

Run training:
```bash
python -m impact.train \
    --data_dir path/to/data \
    --output_dir outputs \
    --batch_size 8 \
    --lr 1e-4 \
    --epochs 100
```

Run evaluation:
```bash
python -m impact.evaluate \
    --model_path path/to/model.pt \
    --data_dir path/to/test_data \
    --output_dir evaluation
```

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

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.
