"""
Tests for battery_dataset.py - Data loading and preprocessing.
"""
import pytest
import torch
import numpy as np
from torch.utils.data import DataLoader
from battery_dataset import BatteryTimeSeriesDataset


class TestBatteryTimeSeriesDataset:
    """Test suite for BatteryTimeSeriesDataset."""
    
    def test_dataset_initialization(self, sample_csv_path, timer):
        """Test dataset initialization."""
        timer.start()
        dataset = BatteryTimeSeriesDataset(
            csv_path=sample_csv_path,
            sequence_length=10,
            forecast_horizon=1,
            use_temperature=True,
            normalize=True,
            train=True
        )
        timer.checkpoint("Dataset initialization")
        
        assert dataset is not None
        assert len(dataset) > 0
        assert dataset.sequence_length == 10
        assert dataset.forecast_horizon == 1
        assert dataset.use_temperature is True
    
    def test_data_loading(self, sample_csv_path, timer):
        """Test data loading step."""
        timer.start()
        dataset = BatteryTimeSeriesDataset(
            csv_path=sample_csv_path,
            sequence_length=10,
            train=True
        )
        timer.checkpoint("Data loading")
        
        assert dataset.data is not None
        assert len(dataset.data) > 0
        assert 'ModuleId' in dataset.data.columns
        assert 'SeriesId' in dataset.data.columns
        assert 'CurrentVoltage' in dataset.data.columns
        assert 'CurrentTemperature' in dataset.data.columns
    
    def test_sequence_creation(self, sample_csv_path, timer):
        """Test sequence creation."""
        timer.start()
        dataset = BatteryTimeSeriesDataset(
            csv_path=sample_csv_path,
            sequence_length=10,
            train=True
        )
        timer.checkpoint("Sequence creation")
        
        assert len(dataset.sequences) > 0
        # Check sequence structure
        seq = dataset.sequences[0]
        assert 'module_id' in seq
        assert 'voltage_seq' in seq
        assert 'target' in seq
        assert len(seq['voltage_seq']) == 10
    
    def test_train_val_split(self, sample_csv_path, timer):
        """Test train/validation split."""
        timer.start()
        train_dataset = BatteryTimeSeriesDataset(
            csv_path=sample_csv_path,
            sequence_length=10,
            train=True
        )
        timer.checkpoint("Train dataset creation")
        
        val_dataset = BatteryTimeSeriesDataset(
            csv_path=sample_csv_path,
            sequence_length=10,
            train=False
        )
        timer.checkpoint("Validation dataset creation")
        
        # Check that datasets are different
        assert len(train_dataset) > 0
        assert len(val_dataset) > 0
        
        # Check that batteries are split (not sequences)
        train_batteries = set(seq['module_id'] for seq in train_dataset.sequences)
        val_batteries = set(seq['module_id'] for seq in val_dataset.sequences)
        
        # Batteries should not overlap
        assert len(train_batteries.intersection(val_batteries)) == 0
    
    def test_normalization(self, sample_csv_path, timer):
        """Test feature normalization."""
        timer.start()
        dataset = BatteryTimeSeriesDataset(
            csv_path=sample_csv_path,
            sequence_length=10,
            normalize=True,
            train=True
        )
        timer.checkpoint("Normalization")
        
        # Check scalers exist
        assert dataset.voltage_scaler is not None
        assert dataset.temp_scaler is not None
        
        # Check that data is normalized (mean ~0, std ~1)
        all_voltages = np.concatenate([
            seq['voltage_seq'] for seq in dataset.sequences
        ])
        mean_voltage = np.mean(all_voltages)
        std_voltage = np.std(all_voltages)
        
        assert abs(mean_voltage) < 0.1  # Should be close to 0
        assert abs(std_voltage - 1.0) < 0.1  # Should be close to 1
    
    def test_without_temperature(self, sample_csv_path, timer):
        """Test dataset without temperature features."""
        timer.start()
        dataset = BatteryTimeSeriesDataset(
            csv_path=sample_csv_path,
            sequence_length=10,
            use_temperature=False,
            train=True
        )
        timer.checkpoint("Dataset without temperature")
        
        assert dataset.use_temperature is False
        assert dataset.temp_scaler is None
        
        # Check that sequences don't have temperature
        x, y = dataset[0]
        assert x.shape[1] == 1  # Only voltage feature
    
    def test_get_item(self, sample_csv_path, timer):
        """Test __getitem__ method."""
        timer.start()
        dataset = BatteryTimeSeriesDataset(
            csv_path=sample_csv_path,
            sequence_length=10,
            use_temperature=True,
            train=True
        )
        timer.checkpoint("Dataset creation")
        
        x, y = dataset[0]
        timer.checkpoint("Get item")
        
        assert isinstance(x, torch.Tensor)
        assert isinstance(y, torch.Tensor)
        assert x.shape == (10, 2)  # sequence_length x features
        assert y.shape == (1,)
    
    def test_dataloader(self, sample_csv_path, timer):
        """Test DataLoader integration."""
        timer.start()
        dataset = BatteryTimeSeriesDataset(
            csv_path=sample_csv_path,
            sequence_length=10,
            train=True
        )
        timer.checkpoint("Dataset creation")
        
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
        timer.checkpoint("DataLoader creation")
        
        # Get a batch
        for x, y in dataloader:
            timer.checkpoint("First batch")
            assert x.shape[0] <= 4  # batch size
            assert x.shape[1] == 10  # sequence length
            assert x.shape[2] == 2  # features (voltage + temperature)
            assert y.shape[0] <= 4
            assert y.shape[1] == 1
            break
    
    def test_max_samples(self, sample_csv_path, timer):
        """Test max_samples parameter."""
        timer.start()
        dataset_full = BatteryTimeSeriesDataset(
            csv_path=sample_csv_path,
            sequence_length=10,
            train=True,
            max_samples=None
        )
        timer.checkpoint("Full dataset")
        
        dataset_limited = BatteryTimeSeriesDataset(
            csv_path=sample_csv_path,
            sequence_length=10,
            train=True,
            max_samples=10
        )
        timer.checkpoint("Limited dataset")
        
        assert len(dataset_limited) <= 10
        if len(dataset_full) > 10:
            assert len(dataset_limited) < len(dataset_full)
    
    def test_missing_temperature_handling(self, sample_csv_path, timer):
        """Test handling of missing temperature values."""
        timer.start()
        dataset = BatteryTimeSeriesDataset(
            csv_path=sample_csv_path,
            sequence_length=10,
            use_temperature=True,
            train=True
        )
        timer.checkpoint("Dataset with missing temp handling")
        
        # Check that no NaN values remain
        for seq in dataset.sequences:
            if seq['temperature_seq'] is not None:
                assert not np.isnan(seq['temperature_seq']).any()
    
    def test_scalers_retrieval(self, sample_csv_path, timer):
        """Test get_scalers method."""
        timer.start()
        dataset = BatteryTimeSeriesDataset(
            csv_path=sample_csv_path,
            sequence_length=10,
            train=True
        )
        timer.checkpoint("Dataset creation")
        
        voltage_scaler, temp_scaler = dataset.get_scalers()
        timer.checkpoint("Get scalers")
        
        assert voltage_scaler is not None
        assert temp_scaler is not None

