"""
LSTM-based Model for Battery Voltage Forecasting
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class VoltageForecastLSTM(nn.Module):
    """
    LSTM model for forecasting battery voltage based on historical voltage and temperature.
    """
    
    def __init__(
        self,
        input_size: int = 2,  # voltage + temperature
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        use_temperature: bool = True
    ):
        """
        Args:
            input_size: Number of input features (1 for voltage only, 2 for voltage+temperature)
            hidden_size: Number of LSTM hidden units
            num_layers: Number of LSTM layers
            dropout: Dropout probability
            use_temperature: Whether temperature is used as input
        """
        super(VoltageForecastLSTM, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_temperature = use_temperature
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, hidden_size // 4)
        self.fc3 = nn.Linear(hidden_size // 4, 1)
        
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
        
        Returns:
            Predicted voltage of shape (batch_size, 1)
        """
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)  # (batch_size, sequence_length, hidden_size)
        
        # Use the last output from the sequence
        last_output = lstm_out[:, -1, :]  # (batch_size, hidden_size)
        
        # Fully connected layers
        out = self.fc1(last_output)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.fc2(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.fc3(out)  # (batch_size, 1)
        
        return out


class VoltageForecastGRU(nn.Module):
    """
    GRU-based model as an alternative to LSTM (faster, sometimes better for simpler patterns).
    """
    
    def __init__(
        self,
        input_size: int = 2,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        use_temperature: bool = True
    ):
        super(VoltageForecastGRU, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_temperature = use_temperature
        
        # GRU layers
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, hidden_size // 4)
        self.fc3 = nn.Linear(hidden_size // 4, 1)
        
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        gru_out, _ = self.gru(x)
        last_output = gru_out[:, -1, :]
        
        out = self.fc1(last_output)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.fc2(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.fc3(out)
        
        return out

