"""
Script for making voltage forecasts using trained model.
"""
# Import device_utils first to set up CUDA warning filters before torch is imported
from device_utils import get_compatible_device

import torch
import numpy as np
import pandas as pd
import argparse
import os
from pathlib import Path
import logging

from battery_dataset import BatteryTimeSeriesDataset
from voltage_forecast_model import VoltageForecastLSTM, VoltageForecastGRU

logger = logging.getLogger(__name__)


def load_model(model_path: str, device: str = None):
    """Load trained model from checkpoint."""
    device = get_compatible_device(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint['model_config']
    
    # Create model
    if config['model_type'].upper() == "LSTM":
        model = VoltageForecastLSTM(
            input_size=config['input_size'],
            hidden_size=config['hidden_size'],
            num_layers=config['num_layers'],
            dropout=config['dropout'],
            use_temperature=config['use_temperature']
        )
    elif config['model_type'].upper() == "GRU":
        model = VoltageForecastGRU(
            input_size=config['input_size'],
            hidden_size=config['hidden_size'],
            num_layers=config['num_layers'],
            dropout=config['dropout'],
            use_temperature=config['use_temperature']
        )
    else:
        raise ValueError(f"Unknown model type: {config['model_type']}")
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, checkpoint, device


def forecast_voltage(
    model_path: str,
    csv_path: str,
    module_id: int = None,
    sequence_length: int = 10,
    output_file: str = None,
    device: str = None
):
    """
    Forecast voltage for batteries.
    
    Args:
        model_path: Path to trained model checkpoint
        csv_path: Path to VoltTemp.csv
        module_id: Specific battery ID to forecast (None for all)
        sequence_length: Input sequence length (should match training)
        output_file: Path to save forecasts (CSV)
        device: Device to use
    """
    # Load model
    logger.info("Loading model...")
    model, checkpoint, device = load_model(model_path, device)
    config = checkpoint['model_config']
    voltage_scaler = checkpoint['voltage_scaler']
    temp_scaler = checkpoint.get('temp_scaler')
    
    # Load data
    logger.info("Loading data...")
    df = pd.read_csv(csv_path)
    df = df[['ModuleId', 'SeriesId', 'CurrentVoltage', 'CurrentTemperature']].copy()
    
    # Handle missing temperature
    df['CurrentTemperature'] = df['CurrentTemperature'].replace(-1, np.nan)
    df = df.sort_values(['ModuleId', 'SeriesId']).reset_index(drop=True)
    df['CurrentTemperature'] = df.groupby('ModuleId')['CurrentTemperature'].ffill().bfill()
    if df['CurrentTemperature'].isna().any():
        df['CurrentTemperature'] = df['CurrentTemperature'].fillna(df['CurrentTemperature'].median())
    
    # Filter by module_id if specified
    if module_id is not None:
        df = df[df['ModuleId'] == module_id].copy()
        if len(df) == 0:
            logger.warning(f"No data found for ModuleId {module_id}")
            return
    
    # Prepare sequences for forecasting
    forecasts = []
    
    for mod_id, group in df.groupby('ModuleId'):
        group = group.sort_values('SeriesId').reset_index(drop=True)
        
        if len(group) < sequence_length:
            continue
        
        # Get the last sequence_length points for forecasting
        last_voltages = group['CurrentVoltage'].values[-sequence_length:]
        last_temps = group['CurrentTemperature'].values[-sequence_length:] if config['use_temperature'] else None
        
        # Normalize
        volt_seq_norm = voltage_scaler.transform(last_voltages.reshape(-1, 1)).flatten()
        temp_seq_norm = None
        if config['use_temperature'] and temp_scaler is not None:
            temp_seq_norm = temp_scaler.transform(last_temps.reshape(-1, 1)).flatten()
        
        # Create input tensor
        if config['use_temperature'] and temp_seq_norm is not None:
            features = np.column_stack([volt_seq_norm, temp_seq_norm])
        else:
            features = volt_seq_norm.reshape(-1, 1)
        
        x = torch.FloatTensor(features).unsqueeze(0).to(device)  # Add batch dimension
        
        # Forecast
        with torch.no_grad():
            pred_norm = model(x).cpu().numpy()[0, 0]
        
        # Denormalize
        pred_voltage = voltage_scaler.inverse_transform([[pred_norm]])[0, 0]
        
        last_series_id = group['SeriesId'].iloc[-1]
        last_voltage = group['CurrentVoltage'].iloc[-1]
        last_temp = group['CurrentTemperature'].iloc[-1] if config['use_temperature'] else None
        
        forecasts.append({
            'ModuleId': mod_id,
            'LastSeriesId': last_series_id,
            'LastVoltage': last_voltage,
            'LastTemperature': last_temp,
            'ForecastVoltage': pred_voltage,
            'VoltageChange': pred_voltage - last_voltage
        })
    
    # Create results dataframe
    results_df = pd.DataFrame(forecasts)
    
    # Log summary
    logger.info(f"Forecasts for {len(results_df)} batteries:")
    logger.info(f"\n{results_df.head(20)}")
    logger.info(f"Average voltage change: {results_df['VoltageChange'].mean():.6f} V")
    logger.info(f"Std voltage change: {results_df['VoltageChange'].std():.6f} V")
    
    # Save to file
    if output_file:
        results_df.to_csv(output_file, index=False)
        logger.info(f"Forecasts saved to: {output_file}")
    
    return results_df


def main():
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    parser = argparse.ArgumentParser(description='Forecast battery voltage using trained model')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--csv_path', type=str, default='data/VoltTemp.csv',
                       help='Path to VoltTemp.csv file')
    parser.add_argument('--module_id', type=int, default=None,
                       help='Specific battery ID to forecast (None for all)')
    parser.add_argument('--sequence_length', type=int, default=10,
                       help='Input sequence length (should match training)')
    parser.add_argument('--output_file', type=str, default=None,
                       help='Path to save forecasts (CSV)')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    forecast_voltage(
        model_path=args.model_path,
        csv_path=args.csv_path,
        module_id=args.module_id,
        sequence_length=args.sequence_length,
        output_file=args.output_file,
        device=args.device
    )


if __name__ == "__main__":
    main()

