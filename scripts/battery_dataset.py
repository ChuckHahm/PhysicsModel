"""
PyTorch Dataset for Battery Voltage and Temperature Time-Series Data
"""
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from typing import Tuple, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import logging

logger = logging.getLogger(__name__)


class BatteryTimeSeriesDataset(Dataset):
    """
    PyTorch Dataset for battery voltage forecasting.
    
    Groups data by ModuleId (battery) and SeriesId (time step),
    creating sequences for time-series forecasting.
    """
    
    def __init__(
        self,
        csv_path: str,
        sequence_length: int = 10,
        forecast_horizon: int = 1,
        use_temperature: bool = True,
        normalize: bool = True,
        min_sequence_length: int = 5,
        train: bool = True,
        train_split: float = 0.8,
        random_seed: int = 42,
        max_samples: Optional[int] = None
    ):
        """
        Args:
            csv_path: Path to VoltTemp.csv file
            sequence_length: Number of time steps to use as input
            forecast_horizon: Number of steps ahead to forecast
            use_temperature: Whether to include temperature as a feature
            normalize: Whether to normalize features
            min_sequence_length: Minimum sequence length for a battery to be included
            train: If True, use training split; if False, use validation split
            train_split: Fraction of data to use for training
            random_seed: Random seed for reproducibility
            max_samples: Maximum number of sequences to use (None for all). Useful for faster training.
        """
        self.csv_path = csv_path
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon
        self.use_temperature = use_temperature
        self.normalize = normalize
        self.min_sequence_length = min_sequence_length
        self.train = train
        self.max_samples = max_samples
        
        # Load and preprocess data
        self.data = self._load_data()
        self.sequences = self._create_sequences()
        
        # Split data
        self.sequences = self._split_data(train_split, random_seed)
        
        # Limit samples if specified
        if self.max_samples is not None and len(self.sequences) > self.max_samples:
            original_size = len(self.sequences)
            np.random.seed(random_seed)
            indices = np.random.choice(len(self.sequences), self.max_samples, replace=False)
            self.sequences = [self.sequences[i] for i in indices]
            logger.info(f"Limited to {self.max_samples} samples (from {original_size} total)")
        
        # Initialize scalers
        self.voltage_scaler = StandardScaler()
        self.temp_scaler = StandardScaler() if use_temperature else None
        
        # Fit scalers on training data
        if normalize:
            self._fit_scalers()
        
        # Apply normalization
        if normalize:
            self.sequences = self._normalize_sequences()
    
    def _load_data(self) -> pd.DataFrame:
        """Load CSV data and preprocess."""
        logger.info("Loading data...")
        df = pd.read_csv(self.csv_path)
        
        # Select relevant columns
        df = df[['ModuleId', 'SeriesId', 'CurrentVoltage', 'CurrentTemperature']].copy()
        
        # Handle missing temperature values (-1)
        df['CurrentTemperature'] = df['CurrentTemperature'].replace(-1, np.nan)
        
        # Sort by ModuleId and SeriesId
        df = df.sort_values(['ModuleId', 'SeriesId']).reset_index(drop=True)
        
        # Forward fill temperature within each battery
        df['CurrentTemperature'] = df.groupby('ModuleId')['CurrentTemperature'].ffill()
        
        # Backward fill any remaining NaN values
        df['CurrentTemperature'] = df.groupby('ModuleId')['CurrentTemperature'].bfill()
        
        # If still NaN, fill with median
        if df['CurrentTemperature'].isna().any():
            median_temp = df['CurrentTemperature'].median()
            df['CurrentTemperature'] = df['CurrentTemperature'].fillna(median_temp)
        
        logger.info(f"Loaded {len(df)} rows for {df['ModuleId'].nunique()} batteries")
        return df
    
    def _create_sequences(self) -> list:
        """Create sequences from time-series data grouped by battery."""
        sequences = []
        
        for module_id, group in self.data.groupby('ModuleId'):
            # Sort by SeriesId to ensure chronological order
            group = group.sort_values('SeriesId').reset_index(drop=True)
            
            # Skip batteries with insufficient data
            if len(group) < self.min_sequence_length + self.forecast_horizon:
                continue
            
            voltages = group['CurrentVoltage'].values
            temperatures = group['CurrentTemperature'].values if self.use_temperature else None
            
            # Create sequences
            for i in range(len(group) - self.sequence_length - self.forecast_horizon + 1):
                # Input sequence
                volt_seq = voltages[i:i + self.sequence_length]
                
                # Target (future voltage)
                target = voltages[i + self.sequence_length + self.forecast_horizon - 1]
                
                # Temperature sequence (if used)
                temp_seq = None
                if self.use_temperature:
                    temp_seq = temperatures[i:i + self.sequence_length]
                
                sequences.append({
                    'module_id': module_id,
                    'voltage_seq': volt_seq.astype(np.float32),
                    'temperature_seq': temp_seq.astype(np.float32) if temp_seq is not None else None,
                    'target': np.float32(target)
                })
        
        logger.info(f"Created {len(sequences)} sequences")
        return sequences
    
    def _split_data(self, train_split: float, random_seed: int) -> list:
        """Split sequences into train/validation sets."""
        np.random.seed(random_seed)
        
        # Get unique batteries
        unique_batteries = list(set(seq['module_id'] for seq in self.sequences))
        np.random.shuffle(unique_batteries)
        
        # Split batteries (not individual sequences) to avoid data leakage
        n_train = int(len(unique_batteries) * train_split)
        train_batteries = set(unique_batteries[:n_train])
        val_batteries = set(unique_batteries[n_train:])
        
        # Filter sequences based on battery assignment
        if self.train:
            split_sequences = [s for s in self.sequences if s['module_id'] in train_batteries]
        else:
            split_sequences = [s for s in self.sequences if s['module_id'] in val_batteries]
        
        logger.info(f"{'Training' if self.train else 'Validation'} set: {len(split_sequences)} sequences from {len(train_batteries if self.train else val_batteries)} batteries")
        return split_sequences
    
    def _fit_scalers(self):
        """Fit scalers on training data."""
        # Extract all voltage and temperature values from sequences
        all_voltages = np.concatenate([seq['voltage_seq'] for seq in self.sequences])
        all_voltages = np.append(all_voltages, [seq['target'] for seq in self.sequences])
        
        self.voltage_scaler.fit(all_voltages.reshape(-1, 1))
        
        if self.use_temperature and self.temp_scaler is not None:
            all_temps = np.concatenate([seq['temperature_seq'] for seq in self.sequences if seq['temperature_seq'] is not None])
            self.temp_scaler.fit(all_temps.reshape(-1, 1))
    
    def _normalize_sequences(self) -> list:
        """Normalize sequences using fitted scalers."""
        normalized_sequences = []
        
        for seq in self.sequences:
            volt_seq = self.voltage_scaler.transform(seq['voltage_seq'].reshape(-1, 1)).flatten()
            target = self.voltage_scaler.transform([[seq['target']]])[0, 0]
            
            temp_seq = None
            if self.use_temperature and seq['temperature_seq'] is not None:
                temp_seq = self.temp_scaler.transform(seq['temperature_seq'].reshape(-1, 1)).flatten()
            
            normalized_sequences.append({
                'module_id': seq['module_id'],
                'voltage_seq': volt_seq.astype(np.float32),
                'temperature_seq': temp_seq.astype(np.float32) if temp_seq is not None else None,
                'target': np.float32(target)
            })
        
        return normalized_sequences
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a sequence and its target."""
        seq = self.sequences[idx]
        
        # Build input tensor
        if self.use_temperature and seq['temperature_seq'] is not None:
            # Stack voltage and temperature: [sequence_length, 2]
            features = np.column_stack([seq['voltage_seq'], seq['temperature_seq']])
        else:
            # Only voltage: [sequence_length, 1]
            features = seq['voltage_seq'].reshape(-1, 1)
        
        x = torch.FloatTensor(features)
        y = torch.FloatTensor([seq['target']])
        
        return x, y
    
    def get_scalers(self):
        """Return scalers for inverse transformation."""
        return self.voltage_scaler, self.temp_scaler

