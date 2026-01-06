"""
Tests for voltage_forecast_model.py - Model architecture and forward pass.
"""
import pytest
import torch
import numpy as np
from voltage_forecast_model import VoltageForecastLSTM, VoltageForecastGRU


class TestVoltageForecastLSTM:
    """Test suite for VoltageForecastLSTM model."""
    
    def test_lstm_initialization(self, timer):
        """Test LSTM model initialization."""
        timer.start()
        model = VoltageForecastLSTM(
            input_size=2,
            hidden_size=64,
            num_layers=2,
            dropout=0.2,
            use_temperature=True
        )
        timer.checkpoint("LSTM initialization")
        
        assert model is not None
        assert model.input_size == 2
        assert model.hidden_size == 64
        assert model.num_layers == 2
        assert model.use_temperature is True
    
    def test_lstm_forward_pass(self, timer):
        """Test LSTM forward pass."""
        timer.start()
        model = VoltageForecastLSTM(
            input_size=2,
            hidden_size=64,
            num_layers=2,
            dropout=0.2
        )
        timer.checkpoint("Model creation")
        
        # Create dummy input: (batch_size, sequence_length, input_size)
        batch_size = 8
        sequence_length = 10
        x = torch.randn(batch_size, sequence_length, 2)
        
        timer.checkpoint("Input creation")
        
        output = model(x)
        timer.checkpoint("Forward pass")
        
        assert output.shape == (batch_size, 1)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_lstm_without_temperature(self, timer):
        """Test LSTM model without temperature."""
        timer.start()
        model = VoltageForecastLSTM(
            input_size=1,
            hidden_size=32,
            num_layers=1,
            use_temperature=False
        )
        timer.checkpoint("Model creation")
        
        x = torch.randn(4, 10, 1)
        output = model(x)
        timer.checkpoint("Forward pass")
        
        assert output.shape == (4, 1)
    
    def test_lstm_different_batch_sizes(self, timer):
        """Test LSTM with different batch sizes."""
        timer.start()
        model = VoltageForecastLSTM(input_size=2, hidden_size=32)
        timer.checkpoint("Model creation")
        
        for batch_size in [1, 4, 16, 32]:
            x = torch.randn(batch_size, 10, 2)
            output = model(x)
            timer.checkpoint(f"Batch size {batch_size}")
            assert output.shape == (batch_size, 1)
    
    def test_lstm_gradient_flow(self, timer):
        """Test that gradients flow through the model."""
        timer.start()
        model = VoltageForecastLSTM(input_size=2, hidden_size=32)
        timer.checkpoint("Model creation")
        
        x = torch.randn(4, 10, 2, requires_grad=True)
        y = torch.randn(4, 1)
        
        output = model(x)
        timer.checkpoint("Forward pass")
        
        loss = torch.nn.functional.mse_loss(output, y)
        loss.backward()
        timer.checkpoint("Backward pass")
        
        # Check that gradients exist
        assert x.grad is not None
        for param in model.parameters():
            assert param.grad is not None


class TestVoltageForecastGRU:
    """Test suite for VoltageForecastGRU model."""
    
    def test_gru_initialization(self, timer):
        """Test GRU model initialization."""
        timer.start()
        model = VoltageForecastGRU(
            input_size=2,
            hidden_size=64,
            num_layers=2,
            dropout=0.2,
            use_temperature=True
        )
        timer.checkpoint("GRU initialization")
        
        assert model is not None
        assert model.input_size == 2
        assert model.hidden_size == 64
        assert model.num_layers == 2
    
    def test_gru_forward_pass(self, timer):
        """Test GRU forward pass."""
        timer.start()
        model = VoltageForecastGRU(
            input_size=2,
            hidden_size=64,
            num_layers=2
        )
        timer.checkpoint("Model creation")
        
        x = torch.randn(8, 10, 2)
        output = model(x)
        timer.checkpoint("Forward pass")
        
        assert output.shape == (8, 1)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_gru_vs_lstm_output_shape(self, timer):
        """Test that GRU and LSTM produce same output shape."""
        timer.start()
        lstm = VoltageForecastLSTM(input_size=2, hidden_size=32)
        gru = VoltageForecastGRU(input_size=2, hidden_size=32)
        timer.checkpoint("Model creation")
        
        x = torch.randn(4, 10, 2)
        
        lstm_out = lstm(x)
        timer.checkpoint("LSTM forward")
        
        gru_out = gru(x)
        timer.checkpoint("GRU forward")
        
        assert lstm_out.shape == gru_out.shape
        assert lstm_out.shape == (4, 1)


class TestModelComparison:
    """Tests comparing different model configurations."""
    
    def test_different_hidden_sizes(self, timer):
        """Test models with different hidden sizes."""
        timer.start()
        for hidden_size in [32, 64, 128]:
            model = VoltageForecastLSTM(
                input_size=2,
                hidden_size=hidden_size
            )
            timer.checkpoint(f"Hidden size {hidden_size}")
            
            x = torch.randn(4, 10, 2)
            output = model(x)
            assert output.shape == (4, 1)
    
    def test_different_num_layers(self, timer):
        """Test models with different numbers of layers."""
        timer.start()
        for num_layers in [1, 2, 3]:
            model = VoltageForecastLSTM(
                input_size=2,
                hidden_size=32,
                num_layers=num_layers
            )
            timer.checkpoint(f"Layers {num_layers}")
            
            x = torch.randn(4, 10, 2)
            output = model(x)
            assert output.shape == (4, 1)
    
    def test_model_parameter_count(self, timer):
        """Test parameter counting."""
        timer.start()
        model = VoltageForecastLSTM(
            input_size=2,
            hidden_size=64,
            num_layers=2
        )
        timer.checkpoint("Model creation")
        
        num_params = sum(p.numel() for p in model.parameters())
        timer.checkpoint("Parameter counting")
        
        assert num_params > 0
        # LSTM with these settings should have several thousand parameters
        assert num_params > 1000

