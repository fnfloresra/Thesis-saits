"""
SAITS Multisite Transfer Learning Implementation
=================================================
Implementation of SAITS with multisite transfer learning for water quality data imputation.

Target: Stations 279, 280, 281, 282, 283
Focus: Tier (1) parameters (Cadmio, Plomo, Cobre) but impute all 14 parameters
Strategy: Pre-train on 4 source stations (279, 280, 281, 283), fine-tune on Station 282

Author: Generated for thesis research
Date: November 2025
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Check if PyPOTS is installed
try:
    from pypots.imputation import SAITS
    from pypots.optim import Adam
    print("[OK] PyPOTS imported successfully")
except ImportError:
    print("ERROR: PyPOTS not installed. Install with: pip install pypots")
    print("Also ensure you have: pip install torch torchvision")
    import sys
    sys.exit((1))

import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")

# ===================================================================================================
# CONFIGURATION PARAMETERS (Based on Strategy Document)
# ===================================================================================================

CONFIG = {
    # File paths
    'station_files': {
        '279': 'scaled_reindexed_station_279_weekly_limited (1).csv',
        '280': 'scaled_reindexed_station_280_weekly_limited (1).csv',
        '281': 'scaled_reindexed_station_281_weekly_limited (1).csv',
        '282': 'scaled_reindexed_station_282_weekly_limited (1).csv',
        '283': 'scaled_reindexed_station_283_weekly_limited (1).csv'
    },

    # Station configuration
    'source_stations': ['279', '280', '281', '283'],  # Pre-training sources
    'target_station': '282',  # Fine-tuning target

    # Data split (based on dates)
    'train_end_year': 2018,    # Training: 2003-2018
    'val_end_year': 2020,      # Validation: 2019-2020
    # Test: 2021-2023 (remainder)

    # Sequence parameters
    'n_steps': 52,             # 52 weeks = (1) year
    'stride': (1),               # Sliding window stride

    # SAITS hyperparameters
    'saits_config': {
        'n_steps': 52,
        'n_features': 14,      # Will be set dynamically
        'n_layers': 2,
        'd_model': 64,
        'n_heads': 4,
        'd_k': 16,
        'd_v': 16,
        'd_ffn': 128,
        'dropout': 0.1,
        'epochs': 50,
        'batch_size': 8,
        'ORT_weight': 0.05,
        'MIT_weight': 1.0,
        'learning_rate': 0.0001,
        'patience': 20,
    },

    # Fine-tuning hyperparameters
    'finetune_config': {
        'learning_rate': 0.0001,  # 10x smaller
        'epochs': 50,
        'patience': 20,
    },

    # Evaluation
    'test_mask_ratio': 0.15,   # Mask 15% for testing
    'random_seed': 42,

    # Tier (1) focus parameters
    'tier1_params': ['Cadmio total', 'Plomo total', 'Cobre total']
}

# ===================================================================================================
# UTILITY FUNCTIONS
# ===================================================================================================

def load_station_data(station_id, config):
    """Load data for a single station"""
    filename = config['station_files'][station_id]
    df = pd.read_csv(filename)
    df['week'] = pd.to_datetime(df['week'])
    return df

def create_temporal_splits(df, config):
    """
    Split data into train/val/test based on temporal periods

    Returns:
        train_df, val_df, test_df
    """
    df = df.copy()
    df['year'] = df['week'].dt.year

    train_mask = df['year'] <= config['train_end_year']
    val_mask = (df['year'] > config['train_end_year']) & (df['year'] <= config['val_end_year'])
    test_mask = df['year'] > config['val_end_year']

    train_df = df[train_mask].drop('year', axis=(1)).reset_index(drop=True)
    val_df = df[val_mask].drop('year', axis=(1)).reset_index(drop=True)
    test_df = df[test_mask].drop('year', axis=(1)).reset_index(drop=True)

    return train_df, val_df, test_df

def create_sequences(data, n_steps, stride=(1)):
    """
    Create sliding window sequences for SAITS

    Args:
        data: numpy array of shape (n_timesteps, n_features)
        n_steps: sequence length
        stride: sliding window stride

    Returns:
        sequences: array of shape (n_sequences, n_steps, n_features)
    """
    n_timesteps, n_features = data.shape

    if n_timesteps < n_steps:
        print(f"Warning: Data has {n_timesteps} timesteps but n_steps={n_steps}")
        return None

    sequences = []
    for i in range(0, n_timesteps - n_steps + (1), stride):
        seq = data[i:i + n_steps, :]
        sequences.append(seq)

    return np.array(sequences)

def prepare_station_sequences(station_id, config, split='train'):
    """
    Prepare sequences for a single station

    Returns:
        sequences array or None if insufficient data
    """
    df = load_station_data(station_id, config)
    train_df, val_df, test_df = create_temporal_splits(df, config)

    # Select appropriate split
    if split == 'train':
        split_df = train_df
    elif split == 'val':
        split_df = val_df
    elif split == 'test':
        split_df = test_df
    else:
        raise ValueError(f"Invalid split: {split}")

    # Extract numeric features (exclude 'week' column)
    feature_cols = [col for col in split_df.columns if col != 'week']
    data = split_df[feature_cols].values

    # Create sequences
    sequences = create_sequences(data, config['n_steps'], config['stride'])

    return sequences, feature_cols

# ===================================================================================================
# DATA PREPARATION
# ===================================================================================================

def prepare_multisite_data(config):
    """
    Prepare data for multisite transfer learning

    Returns:
        Dictionary with train/val/test data for source and target
    """
    print("\n" + "="*100)
    print("PREPARING MULTISITE DATA")
    print("="*100)

    data_dict = {
        'source_train': [],
        'source_val': [],
        'target_train': None,
        'target_val': None,
        'target_test': None,
        'feature_names': None
    }

    # Prepare source stations (for pre-training)
    print(f"\nPreparing SOURCE stations: {config['source_stations']}")
    for station_id in config['source_stations']:
        print(f"\n  Processing Station {station_id}...")

        # Training sequences
        train_seq, feature_names = prepare_station_sequences(station_id, config, 'train')
        if train_seq is not None:
            print(f"    Train sequences: {train_seq.shape}")
            data_dict['source_train'].append(train_seq)

        # Validation sequences
        val_seq, _ = prepare_station_sequences(station_id, config, 'val')
        if val_seq is not None:
            print(f"    Val sequences: {val_seq.shape}")
            data_dict['source_val'].append(val_seq)

        if data_dict['feature_names'] is None:
            data_dict['feature_names'] = feature_names

    # Stack source data
    if data_dict['source_train']:
        data_dict['source_train'] = np.vstack(data_dict['source_train'])
        print(f"\n  Pooled source TRAIN: {data_dict['source_train'].shape}")

    if data_dict['source_val']:
        data_dict['source_val'] = np.vstack(data_dict['source_val'])
        print(f"  Pooled source VAL: {data_dict['source_val'].shape}")

    # Prepare target station (for fine-tuning and testing)
    target_id = config['target_station']
    print(f"\nPreparing TARGET station: {target_id}")

    data_dict['target_train'], _ = prepare_station_sequences(target_id, config, 'train')
    data_dict['target_val'], _ = prepare_station_sequences(target_id, config, 'val')
    data_dict['target_test'], _ = prepare_station_sequences(target_id, config, 'test')

    print(f"  Target TRAIN: {data_dict['target_train'].shape}")
    print(f"  Target VAL: {data_dict['target_val'].shape}")
    print(f"  Target TEST: {data_dict['target_test'].shape}")

    # Update config with actual n_features
    config['saits_config']['n_features'] = len(data_dict['feature_names'])

    return data_dict

# ===================================================================================================
# SAITS PRE-TRAINING
# ===================================================================================================

def pretrain_saits(data_dict, config):
    """
    Pre-train SAITS on source stations

    Returns:
        Trained SAITS model
    """
    print("\n" + "="*100)
    print("PHASE (1): PRE-TRAINING SAITS ON SOURCE STATIONS")
    print("="*100)

    print("\nInitializing SAITS model with configuration:")
    for key, value in config['saits_config'].items():
        print(f"  {key}: {value}")

    # Initialize SAITS
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")

    # Prepare parameters and optimizer
    saits_params = config['saits_config'].copy()
    learning_rate = saits_params.pop('learning_rate', 0.001)

    saits_model = SAITS(
        **saits_params,
        optimizer=Adam(lr=learning_rate),
        device=device
    )

    print("\nStarting pre-training...")
    print(f"Training samples: {data_dict['source_train'].shape[0]}")
    print(f"Validation samples: {data_dict['source_val'].shape[0]}")

    # Train
    saits_model.fit(
        train_set={'X': data_dict['source_train'], 'X_ori': data_dict['source_train']},
        val_set={'X': data_dict['source_val'], 'X_ori': data_dict['source_val']}
    )

    print("\n[OK] Pre-training complete!")

    # Save pre-trained model
    saits_model.save('saits_pretrained_source_stations.pypots', overwrite=True)
    print("[OK] Model saved to: saits_pretrained_source_stations.pypots")

    return saits_model

# ===================================================================================================
# SAITS FINE-TUNING
# ===================================================================================================

def finetune_saits(pretrained_model, data_dict, config):
    """
    Fine-tune SAITS on target station

    Returns:
        Fine-tuned SAITS model
    """
    print("\n" + "="*100)
    print(f"PHASE 2: FINE-TUNING SAITS ON TARGET STATION {config['target_station']}")
    print("="*100)

    # Load pre-trained model (or use passed model)
    saits_model = pretrained_model

    # Update hyperparameters for fine-tuning
    print("\nUpdating hyperparameters for fine-tuning:")
    print(f"  Learning rate: {config['saits_config']['learning_rate']} → {config['finetune_config']['learning_rate']}")
    print(f"  Epochs: {config['saits_config']['epochs']} → {config['finetune_config']['epochs']}")
    print(f"  Patience: {config['saits_config']['patience']} → {config['finetune_config']['patience']}")

    # Note: PyPOTS may require re-initialization for parameter changes
    # Create new model instance with fine-tuning config
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    finetune_config = config['saits_config'].copy()
    finetune_config.update(config['finetune_config'])
    
    # Extract learning rate for optimizer
    learning_rate = finetune_config.pop('learning_rate', 0.001)

    saits_finetuned = SAITS(
        **finetune_config,
        optimizer=Adam(lr=learning_rate),
        device=device
    )

    # Transfer weights from pretrained model
    print("\nTransferring pre-trained weights...")
    saits_finetuned.model.load_state_dict(pretrained_model.model.state_dict())

    # Optional: Implement layer freezing here
    # For now, we'll do full fine-tuning with small learning rate
    print("Strategy: Full fine-tuning with reduced learning rate")

    print("\nStarting fine-tuning...")
    print(f"Training samples: {data_dict['target_train'].shape[0]}")
    print(f"Validation samples: {data_dict['target_val'].shape[0]}")

    # Fine-tune
    saits_finetuned.fit(
        train_set={'X': data_dict['target_train'], 'X_ori': data_dict['target_train']},
        val_set={'X': data_dict['target_val'], 'X_ori': data_dict['target_val']}
    )

    print("\n[OK] Fine-tuning complete!")

    # Save fine-tuned model
    saits_finetuned.save(f'saits_finetuned_station_{config["target_station"]}.pypots', overwrite=True)
    print(f"[OK] Model saved to: saits_finetuned_station_{config['target_station']}.pypots")

    return saits_finetuned

# ===================================================================================================
# EVALUATION
# ===================================================================================================

def create_test_masks(data, mask_ratio, random_seed):
    """
    Create masks for evaluation - mask observed values for testing

    Args:
        data: array of shape (n_samples, n_steps, n_features)
        mask_ratio: fraction of observed values to mask
        random_seed: for reproducibility

    Returns:
        masked_data: data with test values masked as NaN
        test_masks: boolean array indicating which values were masked for testing
        original_data: original data for ground truth
    """
    np.random.seed(random_seed)

    original_data = data.copy()
    masked_data = data.copy()
    test_masks = np.zeros_like(data, dtype=bool)

    # For each feature
    for feat_idx in range(data.shape[2]):
        # Find observed values (not originally NaN)
        observed_mask = ~np.isnan(data[:, :, feat_idx])
        observed_indices = np.where(observed_mask)

        n_observed = len(observed_indices[0])
        if n_observed == 0:
            continue

        # Randomly select mask_ratio of observed values
        n_to_mask = int(n_observed * mask_ratio)
        if n_to_mask == 0:
            continue

        mask_indices = np.random.choice(n_observed, size=n_to_mask, replace=False)

        # Apply masks
        for idx in mask_indices:
            sample_idx = observed_indices[0][idx]
            step_idx = observed_indices[(1)][idx]
            masked_data[sample_idx, step_idx, feat_idx] = np.nan
            test_masks[sample_idx, step_idx, feat_idx] = True

    return masked_data, test_masks, original_data

def calculate_metrics(y_true, y_pred):
    """Calculate MAE, RMSE, R2"""
    # Remove NaN values
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true_clean = y_true[mask]
    y_pred_clean = y_pred[mask]

    if len(y_true_clean) == 0:
        return np.nan, np.nan, np.nan, 0

    mae = mean_absolute_error(y_true_clean, y_pred_clean)
    rmse = np.sqrt(mean_squared_error(y_true_clean, y_pred_clean))

    # R² with variance check
    if np.var(y_true_clean) > 1e-10:
        r2 = r2_score(y_true_clean, y_pred_clean)
    else:
        r2 = np.nan

    return mae, rmse, r2, len(y_true_clean)

def evaluate_model(model, test_data, config, data_dict):
    """
    Evaluate model on test set

    Returns:
        DataFrame with metrics per parameter
    """
    print("\n" + "="*100)
    print(f"PHASE 3: EVALUATION ON STATION {config['target_station']} TEST SET")
    print("="*100)

    # Create test masks
    print(f"\nCreating test masks (masking {config['test_mask_ratio']*100:.1f}% of observed values)...")
    masked_test_data, test_masks, original_test_data = create_test_masks(
        test_data, 
        config['test_mask_ratio'], 
        config['random_seed']
    )

    print(f"Total test values masked: {np.sum(test_masks)}")

    # Impute
    print("\nGenerating imputations...")
    imputed_data = model.impute({'X': masked_test_data})

    # Calculate metrics per parameter
    results = []
    feature_names = data_dict['feature_names']

    print("\n" + "="*100)
    print("RESULTS BY PARAMETER")
    print("="*100)
    print(f"\n{'Parameter':<40} {'MAE':<12} {'RMSE':<12} {'R2':<10} {'N_Test'}")
    print("-"*100)

    for feat_idx, param_name in enumerate(feature_names):
        # Extract values for this parameter where we masked for testing
        y_true = original_test_data[:, :, feat_idx][test_masks[:, :, feat_idx]]
        y_pred = imputed_data[:, :, feat_idx][test_masks[:, :, feat_idx]]

        mae, rmse, r2, n_test = calculate_metrics(y_true, y_pred)

        results.append({
            'Parameter': param_name,
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2,
            'N_Test': n_test
        })

        param_short = param_name[:38]
        print(f"{param_short:<40} {mae:<12.6f} {rmse:<12.6f} {r2:<10.4f} {n_test:<8}")

    results_df = pd.DataFrame(results)

    # Overall statistics
    print("\n" + "="*100)
    print("OVERALL STATISTICS")
    print("="*100)
    print(f"Average MAE:  {results_df['MAE'].mean():.6f}")
    print(f"Average RMSE: {results_df['RMSE'].mean():.6f}")
    print(f"Average R2:   {results_df['R2'].mean():.6f}")

    # Tier (1) statistics
    tier1_results = results_df[results_df['Parameter'].isin(config['tier1_params'])]
    if len(tier1_results) > 0:
        print("\n" + "="*100)
        print("TIER (1) PARAMETERS (Cadmio, Plomo, Cobre)")
        print("="*100)
        print(tier1_results.to_string(index=False))
        print(f"\nTier (1) Average MAE:  {tier1_results['MAE'].mean():.6f}")
        print(f"Tier (1) Average RMSE: {tier1_results['RMSE'].mean():.6f}")
        print(f"Tier (1) Average R2:   {tier1_results['R2'].mean():.6f}")

    return results_df, imputed_data

# ===================================================================================================
# MAIN EXECUTION
# ===================================================================================================

def main():
    """Main execution function"""
    print("="*100)
    print("SAITS MULTISITE TRANSFER LEARNING")
    print("Water Quality Data Imputation")
    print("="*100)

    # (1). Prepare data
    data_dict = prepare_multisite_data(CONFIG)

    # 2. Pre-train SAITS
    pretrained_model = pretrain_saits(data_dict, CONFIG)

    # 3. Fine-tune SAITS
    finetuned_model = finetune_saits(pretrained_model, data_dict, CONFIG)

    # 4. Evaluate
    results_df, imputed_data = evaluate_model(
        finetuned_model,
        data_dict['target_test'],
        CONFIG,
        data_dict
    )

    # 5. Save results
    results_df.to_csv(f'saits_results_station_{CONFIG["target_station"]}.csv', index=False)
    print(f"\n[OK] Results saved to: saits_results_station_{CONFIG['target_station']}.csv")

    print("\n" + "="*100)
    print("EXECUTION COMPLETE")
    print("="*100)

    return results_df, finetuned_model

if __name__ == "__main__":
    results_df, model = main()

