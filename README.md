# SAITS - Self-Attention-based Imputation for Time Series

A PyTorch implementation of SAITS (Self-Attention-based Imputation for Time Series) for multivariate time series data imputation.

## Overview

This project implements the SAITS model, which uses a dual-branch self-attention architecture to impute missing values in multivariate time series data. SAITS leverages the powerful attention mechanism to capture both temporal dependencies and feature correlations for accurate imputation.

### Key Features

- **Dual-branch self-attention architecture**: Two DMSA (Diagonal-Masked Self-Attention) blocks for progressive imputation refinement
- **Combined ORT and MIT losses**: Jointly optimizes observed reconstruction and masked imputation tasks
- **Flexible data preprocessing**: Utilities for normalization, sliding windows, and train/test splitting
- **Comprehensive evaluation metrics**: MAE, MSE, RMSE, and MRE for imputation quality assessment

## Installation

### Requirements

- Python 3.9+
- PyTorch 1.9+
- NumPy, Pandas, Scikit-learn

### Setup

```bash
# Clone the repository
git clone https://github.com/fnfloresra/Thesis-saits.git
cd Thesis-saits

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Basic Usage

```python
import torch
from saits import SAITS

# Create model
model = SAITS(
    n_features=10,      # Number of features in time series
    n_layers=2,         # Number of encoder layers
    d_model=128,        # Model dimension
    n_heads=4,          # Number of attention heads
    d_ff=256,           # Feed-forward dimension
    dropout=0.1         # Dropout rate
)

# Prepare data (batch_size, seq_len, n_features)
x = torch.randn(32, 100, 10)      # Input with missing values set to 0
mask = torch.ones(32, 100, 10)    # 1 = observed, 0 = missing
mask[:, 20:30, :5] = 0            # Mark some values as missing

# Impute missing values
imputed = model.impute(x, mask)
```

### Training Pipeline

```python
from torch.utils.data import DataLoader
from saits import SAITS, TimeSeriesDataset, SAITSTrainer

# Prepare dataset
train_dataset = TimeSeriesDataset(
    data=train_data,              # Shape: (n_samples, seq_len, n_features)
    masks=train_masks,            # Optional: existing missing value masks
    artificial_mask_rate=0.2,     # Rate of artificial masking for training
    mode='train'
)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Create and train model
model = SAITS(n_features=10)
trainer = SAITSTrainer(model, device='cuda', learning_rate=0.001)
history = trainer.train(
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=100,
    early_stopping_patience=20
)
```

### Evaluation

```python
from saits import evaluate_imputation

metrics = evaluate_imputation(
    ground_truth=original_data,    # Complete original data
    imputed=imputed_data,          # Model predictions
    missing_mask=mask              # 1 = missing (imputed), 0 = observed
)

print(f"MAE: {metrics['mae']:.4f}")
print(f"RMSE: {metrics['rmse']:.4f}")
```

## Project Structure

```
Thesis-saits/
├── saits/
│   ├── __init__.py          # Package initialization
│   ├── model.py             # SAITS model implementation
│   ├── data.py              # Data preprocessing utilities
│   ├── train.py             # Training pipeline
│   └── evaluate.py          # Evaluation metrics
├── tests/
│   ├── test_model.py        # Model tests
│   ├── test_data.py         # Data utilities tests
│   └── test_evaluate.py     # Evaluation tests
├── example.py               # Example usage script
├── requirements.txt         # Dependencies
└── README.md               # This file
```

## Model Architecture

The SAITS model consists of:

1. **Input Embedding**: Projects input features (concatenated with mask) to model dimension
2. **Positional Encoding**: Adds temporal position information
3. **First DMSA Block**: Self-attention layers for initial imputation (x̂₁)
4. **Second DMSA Block**: Refines imputation using output from first block (x̂₂)
5. **Combination Layer**: Combines outputs from both blocks (x̂₃)
6. **Final Output**: Replaces missing values with combined predictions

### Loss Function

The model is trained with a combined loss:
- **ORT Loss (Observed Reconstruction Task)**: Reconstruction error on observed values
- **MIT Loss (Masked Imputation Task)**: Imputation error on artificially masked values

## Running Tests

```bash
# Install pytest
pip install pytest

# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_model.py -v
```

## Running the Example

```bash
python example.py
```

This will:
1. Generate synthetic time series data with missing values
2. Train a SAITS model
3. Evaluate imputation quality on a test set

## References

- Du, W., et al. (2023). "SAITS: Self-Attention-based Imputation for Time Series." Expert Systems with Applications.

## License

This project is for thesis/educational purposes.
