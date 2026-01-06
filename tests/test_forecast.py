"""
Tests for forecast.py - Forecasting functionality.
"""
import pytest
import torch
import pandas as pd
from pathlib import Path
from forecast import load_model, forecast_voltage
from train import train


class TestModelLoading:
    """Test suite for model loading."""
    
    def test_load_model(self, sample_csv_path, temp_model_dir, timer):
        """Test loading a trained model."""
        timer.start()
        # First train a model
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
            use_mlflow=False,
            device='cpu'
        )
        timer.checkpoint("Model training")
        
        model_path = Path(temp_model_dir) / "best_model.pth"
        model, checkpoint, device = load_model(str(model_path), 'cpu')
        timer.checkpoint("Model loading")
        
        assert model is not None
        assert checkpoint is not None
        assert 'model_config' in checkpoint
        assert 'voltage_scaler' in checkpoint
        assert model.training is False  # Should be in eval mode


class TestForecasting:
    """Test suite for forecasting functionality."""
    
    def test_forecast_voltage_all_batteries(self, sample_csv_path, temp_model_dir, timer):
        """Test forecasting for all batteries."""
        timer.start()
        # Train a model first
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
        timer.checkpoint("Model training")
        
        model_path = Path(temp_model_dir) / "best_model.pth"
        
        # Run forecasting
        results = forecast_voltage(
            model_path=str(model_path),
            csv_path=sample_csv_path,
            sequence_length=10,
            device='cpu'
        )
        timer.checkpoint("Forecasting")
        
        assert results is not None
        assert len(results) > 0
        assert 'ModuleId' in results.columns
        assert 'ForecastVoltage' in results.columns
        assert 'VoltageChange' in results.columns
    
    def test_forecast_specific_battery(self, sample_csv_path, temp_model_dir, timer):
        """Test forecasting for a specific battery."""
        timer.start()
        # Train a model
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
        timer.checkpoint("Model training")
        
        # Get a module ID from the data
        df = pd.read_csv(sample_csv_path)
        module_id = df['ModuleId'].iloc[0]
        
        model_path = Path(temp_model_dir) / "best_model.pth"
        
        results = forecast_voltage(
            model_path=str(model_path),
            csv_path=sample_csv_path,
            module_id=module_id,
            sequence_length=10,
            device='cpu'
        )
        timer.checkpoint("Forecasting specific battery")
        
        assert results is not None
        assert len(results) == 1
        assert results['ModuleId'].iloc[0] == module_id
    
    def test_forecast_output_file(self, sample_csv_path, temp_model_dir, temp_output_dir, timer):
        """Test saving forecasts to file."""
        timer.start()
        # Train a model
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
        timer.checkpoint("Model training")
        
        output_file = Path(temp_output_dir) / "forecasts.csv"
        model_path = Path(temp_model_dir) / "best_model.pth"
        
        forecast_voltage(
            model_path=str(model_path),
            csv_path=sample_csv_path,
            sequence_length=10,
            output_file=str(output_file),
            device='cpu'
        )
        timer.checkpoint("Forecasting and saving")
        
        assert output_file.exists()
        
        # Check file contents
        saved_results = pd.read_csv(output_file)
        assert len(saved_results) > 0
        assert 'ForecastVoltage' in saved_results.columns
    
    def test_forecast_without_temperature(self, sample_csv_path, temp_model_dir, timer):
        """Test forecasting with model trained without temperature."""
        timer.start()
        # Train model without temperature
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
        timer.checkpoint("Model training")
        
        model_path = Path(temp_model_dir) / "best_model.pth"
        
        results = forecast_voltage(
            model_path=str(model_path),
            csv_path=sample_csv_path,
            sequence_length=10,
            device='cpu'
        )
        timer.checkpoint("Forecasting")
        
        assert results is not None
        assert len(results) > 0
    
    def test_forecast_output_structure(self, sample_csv_path, temp_model_dir, timer):
        """Test that forecast output has correct structure."""
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
        timer.checkpoint("Model training")
        
        model_path = Path(temp_model_dir) / "best_model.pth"
        results = forecast_voltage(
            model_path=str(model_path),
            csv_path=sample_csv_path,
            sequence_length=10,
            device='cpu'
        )
        timer.checkpoint("Forecasting")
        
        # Check required columns
        required_columns = [
            'ModuleId', 'LastSeriesId', 'LastVoltage',
            'ForecastVoltage', 'VoltageChange'
        ]
        for col in required_columns:
            assert col in results.columns
        
        # Check data types
        assert pd.api.types.is_numeric_dtype(results['ForecastVoltage'])
        assert pd.api.types.is_numeric_dtype(results['VoltageChange'])

