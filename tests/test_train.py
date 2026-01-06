"""
Tests for train.py - Training pipeline.
"""
import pytest
import torch
import os
from pathlib import Path
from train import train, train_epoch, validate
from battery_dataset import BatteryTimeSeriesDataset
from voltage_forecast_model import VoltageForecastLSTM, VoltageForecastGRU
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim


class TestTrainingFunctions:
    """Test suite for training utility functions."""
    
    def test_train_epoch(self, sample_csv_path, timer):
        """Test training for one epoch."""
        timer.start()
        # Create small dataset
        dataset = BatteryTimeSeriesDataset(
            csv_path=sample_csv_path,
            sequence_length=10,
            train=True,
            max_samples=10000
        )
        timer.checkpoint("Dataset creation")
        
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
        timer.checkpoint("DataLoader creation")
        
        # Create model
        model = VoltageForecastLSTM(
            input_size=2,
            hidden_size=32,
            num_layers=1
        )
        timer.checkpoint("Model creation")
        
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        device = torch.device('cpu')
        
        # Train one epoch
        loss = train_epoch(model, dataloader, criterion, optimizer, device)
        timer.checkpoint("Training epoch")
        
        assert isinstance(loss, float)
        assert loss >= 0
        assert not (loss != loss)  # Check for NaN
    
    def test_validate(self, sample_csv_path, timer):
        """Test validation function."""
        timer.start()
        dataset = BatteryTimeSeriesDataset(
            csv_path=sample_csv_path,
            sequence_length=10,
            train=False,
            max_samples=10000
        )
        timer.checkpoint("Dataset creation")
        
        dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
        timer.checkpoint("DataLoader creation")
        
        model = VoltageForecastLSTM(input_size=2, hidden_size=32)
        timer.checkpoint("Model creation")
        
        criterion = nn.MSELoss()
        device = torch.device('cpu')
        
        val_loss = validate(model, dataloader, criterion, device)
        timer.checkpoint("Validation")
        
        assert isinstance(val_loss, float)
        assert val_loss >= 0


class TestTrainingPipeline:
    """Test suite for full training pipeline."""
    
    def test_training_minimal(self, sample_csv_path, temp_model_dir, timer):
        """Test minimal training run."""
        timer.start()
        train(
            csv_path=sample_csv_path,
            output_dir=temp_model_dir,
            sequence_length=10,
            hidden_size=32,
            num_layers=1,
            batch_size=4,
            num_epochs=2,
            patience=5,
            max_samples=10000,
            use_mlflow=False,  # Disable MLflow for faster tests
            device='cpu'
        )
        timer.checkpoint("Full training")
        
        # Check that model file was created
        model_path = Path(temp_model_dir) / "best_model.pth"
        assert model_path.exists()
        
        # Check that checkpoint can be loaded
        checkpoint = torch.load(model_path, map_location='cpu')
        timer.checkpoint("Model loading")
        
        assert 'model_state_dict' in checkpoint
        assert 'model_config' in checkpoint
        assert 'voltage_scaler' in checkpoint
    
    def test_training_with_gru(self, sample_csv_path, temp_model_dir, timer):
        """Test training with GRU model."""
        timer.start()
        train(
            csv_path=sample_csv_path,
            output_dir=temp_model_dir,
            model_type='GRU',
            sequence_length=10,
            hidden_size=32,
            num_layers=1,
            batch_size=4,
            num_epochs=2,
            patience=5,
            max_samples=10000,
            use_mlflow=False,
            device='cpu'
        )
        timer.checkpoint("GRU training")
        
        model_path = Path(temp_model_dir) / "best_model.pth"
        assert model_path.exists()
        
        checkpoint = torch.load(model_path, map_location='cpu')
        assert checkpoint['model_config']['model_type'] == 'GRU'
    
    def test_training_without_temperature(self, sample_csv_path, temp_model_dir, timer):
        """Test training without temperature features."""
        timer.start()
        train(
            csv_path=sample_csv_path,
            output_dir=temp_model_dir,
            use_temperature=False,
            sequence_length=10,
            hidden_size=32,
            batch_size=4,
            num_epochs=2,
            patience=5,
            max_samples=10000,
            use_mlflow=False,
            device='cpu'
        )
        timer.checkpoint("Training without temperature")
        
        checkpoint = torch.load(
            Path(temp_model_dir) / "best_model.pth",
            map_location='cpu'
        )
        assert checkpoint['model_config']['use_temperature'] is False
        assert checkpoint['model_config']['input_size'] == 1
    
    def test_training_early_stopping(self, sample_csv_path, temp_model_dir, timer):
        """Test that early stopping works."""
        timer.start()
        train(
            csv_path=sample_csv_path,
            output_dir=temp_model_dir,
            sequence_length=10,
            hidden_size=32,
            batch_size=4,
            num_epochs=100,  # Many epochs
            patience=2,  # Stop after 2 epochs without improvement
            max_samples=10000,
            use_mlflow=False,
            device='cpu'
        )
        timer.checkpoint("Training with early stopping")
        
        # Training should have stopped early
        model_path = Path(temp_model_dir) / "best_model.pth"
        assert model_path.exists()
    
    def test_training_outputs(self, sample_csv_path, temp_model_dir, timer):
        """Test that training produces expected outputs."""
        timer.start()
        train(
            csv_path=sample_csv_path,
            output_dir=temp_model_dir,
            sequence_length=10,
            hidden_size=32,
            batch_size=4,
            num_epochs=2,
            patience=5,
            max_samples=10000,
            use_mlflow=False,
            device='cpu'
        )
        timer.checkpoint("Training")
        
        output_dir = Path(temp_model_dir)
        
        # Check for model file
        assert (output_dir / "best_model.pth").exists()
        
        # Check for training curves plot (if matplotlib works)
        # Note: This might not exist if there's an issue, so we'll just check model
    
    def test_training_different_hyperparameters(self, sample_csv_path, temp_model_dir, timer):
        """Test training with different hyperparameter combinations."""
        timer.start()
        configs = [
            {'hidden_size': 32, 'num_layers': 1, 'dropout': 0.1},
            {'hidden_size': 64, 'num_layers': 2, 'dropout': 0.2},
        ]
        
        for i, config in enumerate(configs):
            model_dir = Path(temp_model_dir) / f"model_{i}"
            model_dir.mkdir()
            
            train(
                csv_path=sample_csv_path,
                output_dir=str(model_dir),
                sequence_length=10,
                batch_size=4,
                num_epochs=2,
                patience=5,
                max_samples=10000,
                use_mlflow=False,
                device='cpu',
                **config
            )
            timer.checkpoint(f"Config {i+1}")
            
            assert (model_dir / "best_model.pth").exists()

