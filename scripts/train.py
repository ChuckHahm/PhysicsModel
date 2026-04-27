"""
Training script for battery voltage forecasting model.
"""
# Import device_utils first to set up CUDA warning filters before torch is imported
from device_utils import get_compatible_device

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import argparse
import os
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging
import mlflow
import mlflow.pytorch
from datetime import datetime

from battery_dataset import BatteryTimeSeriesDataset
from voltage_forecast_model import VoltageForecastLSTM, VoltageForecastGRU

logger = logging.getLogger(__name__)


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for x, y in tqdm(dataloader, desc="Training"):
        x, y = x.to(device), y.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        predictions = model(x)
        loss = criterion(predictions, y)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches if num_batches > 0 else 0.0


def validate(model, dataloader, criterion, device):
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for x, y in tqdm(dataloader, desc="Validating"):
            x, y = x.to(device), y.to(device)
            predictions = model(x)
            loss = criterion(predictions, y)
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches if num_batches > 0 else 0.0


def train(
    csv_path: str,
    output_dir: str = "./models",
    sequence_length: int = 10,
    forecast_horizon: int = 1,
    use_temperature: bool = True,
    model_type: str = "LSTM",
    hidden_size: int = 64,
    num_layers: int = 2,
    dropout: float = 0.2,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    num_epochs: int = 50,
    patience: int = 10,
    device: str = None,
    max_samples: int = None,
    mlflow_experiment_name: str = "battery_voltage_forecasting",
    use_mlflow: bool = True
):
    """
    Train the voltage forecasting model.
    
    Args:
        csv_path: Path to VoltTemp.csv
        output_dir: Directory to save model and plots
        sequence_length: Input sequence length
        forecast_horizon: Steps ahead to forecast
        use_temperature: Whether to use temperature features
        model_type: "LSTM" or "GRU"
        hidden_size: LSTM/GRU hidden size
        num_layers: Number of LSTM/GRU layers
        dropout: Dropout rate
        batch_size: Training batch size
        learning_rate: Learning rate
        num_epochs: Maximum number of epochs
        patience: Early stopping patience
        device: Device to use (cuda/cpu)
        max_samples: Maximum number of training sequences (None for all)
        mlflow_experiment_name: MLflow experiment name
        use_mlflow: Whether to use MLflow tracking (default: True)
    """
    # Set device
    device = get_compatible_device(device)
    logger.info(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up log file (always create for consistency, but only log to MLflow if enabled)
    log_file_path = os.path.join(output_dir, 'training.log')
    file_handler = None
    if use_mlflow:
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        
        # Add file handler to root logger to capture all logs
        root_logger = logging.getLogger()
        root_logger.addHandler(file_handler)
        
        # Initialize MLflow
        mlflow.set_experiment(mlflow_experiment_name)
    
    # Create datasets
    logger.info("Creating datasets...")
    train_dataset = BatteryTimeSeriesDataset(
        csv_path=csv_path,
        sequence_length=sequence_length,
        forecast_horizon=forecast_horizon,
        use_temperature=use_temperature,
        normalize=True,
        train=True,
        max_samples=max_samples
    )
    
    val_dataset = BatteryTimeSeriesDataset(
        csv_path=csv_path,
        sequence_length=sequence_length,
        forecast_horizon=forecast_horizon,
        use_temperature=use_temperature,
        normalize=True,
        train=False
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # Determine input size
    input_size = 2 if use_temperature else 1
    
    # Create model
    if model_type.upper() == "LSTM":
        model = VoltageForecastLSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            use_temperature=use_temperature
        ).to(device)
    elif model_type.upper() == "GRU":
        model = VoltageForecastGRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            use_temperature=use_temperature
        ).to(device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    logger.info(f"Model: {model_type}")
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Parameters: {num_params:,}")
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Start MLflow run (if enabled)
    if use_mlflow:
        mlflow_run_context = mlflow.start_run()
    else:
        # Create a no-op context manager
        from contextlib import nullcontext
        mlflow_run_context = nullcontext()
    
    with mlflow_run_context:
        # Log hyperparameters (if MLflow enabled)
        if use_mlflow:
            mlflow.log_params({
                'sequence_length': sequence_length,
                'forecast_horizon': forecast_horizon,
                'use_temperature': use_temperature,
                'model_type': model_type,
                'hidden_size': hidden_size,
                'num_layers': num_layers,
                'dropout': dropout,
                'batch_size': batch_size,
                'learning_rate': learning_rate,
                'num_epochs': num_epochs,
                'patience': patience,
                'max_samples': max_samples if max_samples else 'all',
                'num_params': num_params,
                'train_samples': len(train_dataset),
                'val_samples': len(val_dataset)
            })
            
            # Add tags for run metadata and filtering
            mlflow.set_tag("model_type", model_type)
            mlflow.set_tag("dataset_size", len(train_dataset))
            mlflow.set_tag("device", str(device))
            mlflow.set_tag("status", "initialized")
        
        # Training loop
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        patience_counter = 0
        
        logger.info("Starting training...")
        if use_mlflow:
            mlflow.set_tag("status", "training")
        for epoch in range(num_epochs):
            logger.info(f"Epoch {epoch + 1}/{num_epochs}")
            
            # Train
            train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
            train_losses.append(train_loss)
            
            # Validate
            val_loss = validate(model, val_loader, criterion, device)
            val_losses.append(val_loss)
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            logger.info(f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            
            # Log metrics to MLflow (if enabled)
            if use_mlflow:
                mlflow.log_metrics({
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'epoch': epoch + 1
                }, step=epoch + 1)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                model_path = os.path.join(output_dir, 'best_model.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'model_config': {
                        'input_size': input_size,
                        'hidden_size': hidden_size,
                        'num_layers': num_layers,
                        'dropout': dropout,
                        'use_temperature': use_temperature,
                        'model_type': model_type
                    },
                    'voltage_scaler': train_dataset.voltage_scaler,
                    'temp_scaler': train_dataset.temp_scaler if use_temperature else None
                }, model_path)
                logger.info(f"Saved best model (Val Loss: {val_loss:.6f})")
                
                # Log best model to MLflow
                mlflow.log_artifact(model_path, artifact_path="models")
                mlflow.pytorch.log_model(model, "best_model")
                mlflow.log_metric('best_val_loss', best_val_loss)
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break
        
        # Plot training curves
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (MSE)')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        plot_path = os.path.join(output_dir, 'training_curves.png')
        plt.savefig(plot_path)
        plt.close()
        
        # Log plot and log file to MLflow (if enabled)
        if use_mlflow:
            mlflow.log_artifact(plot_path, artifact_path="plots")
            mlflow.log_artifact(log_file_path, artifact_path="logs")
            
            # Update status and add final metrics as tags
            mlflow.set_tag("status", "completed")
            mlflow.set_tag("best_val_loss", f"{best_val_loss:.6f}")
            logger.info(f"MLflow run ID: {mlflow.active_run().info.run_id}")
        
        logger.info("Training completed!")
        logger.info(f"Best validation loss: {best_val_loss:.6f}")
        logger.info(f"Model saved to: {os.path.join(output_dir, 'best_model.pth')}")


def main():
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    parser = argparse.ArgumentParser(description='Train battery voltage forecasting model')
    parser.add_argument('--csv_path', type=str, default='data/VoltTemp.csv',
                       help='Path to VoltTemp.csv file')
    parser.add_argument('--output_dir', type=str, default='./models',
                       help='Directory to save model and outputs')
    parser.add_argument('--sequence_length', type=int, default=10,
                       help='Input sequence length')
    parser.add_argument('--forecast_horizon', type=int, default=1,
                       help='Steps ahead to forecast')
    parser.add_argument('--use_temperature', action='store_true', default=True,
                       help='Use temperature as feature')
    parser.add_argument('--no_temperature', dest='use_temperature', action='store_false',
                       help='Do not use temperature as feature')
    parser.add_argument('--model_type', type=str, default='LSTM', choices=['LSTM', 'GRU'],
                       help='Model type: LSTM or GRU')
    parser.add_argument('--hidden_size', type=int, default=64,
                       help='Hidden size of LSTM/GRU')
    parser.add_argument('--num_layers', type=int, default=2,
                       help='Number of LSTM/GRU layers')
    parser.add_argument('--dropout', type=float, default=0.2,
                       help='Dropout rate')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=50,
                       help='Number of epochs')
    parser.add_argument('--patience', type=int, default=10,
                       help='Early stopping patience')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum number of training sequences (None for all). Useful for faster training.')
    parser.add_argument('--mlflow_experiment', type=str, default='battery_voltage_forecasting',
                       help='MLflow experiment name')
    parser.add_argument('--use_mlflow', action='store_true', default=True,
                       help='Enable MLflow tracking (default: True)')
    parser.add_argument('--no_mlflow', dest='use_mlflow', action='store_false',
                       help='Disable MLflow tracking')
    
    args = parser.parse_args()
    
    train(
        csv_path=args.csv_path,
        output_dir=args.output_dir,
        sequence_length=args.sequence_length,
        forecast_horizon=args.forecast_horizon,
        use_temperature=args.use_temperature,
        model_type=args.model_type,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        patience=args.patience,
        device=args.device,
        max_samples=args.max_samples,
        mlflow_experiment_name=args.mlflow_experiment,
        use_mlflow=args.use_mlflow
    )


if __name__ == "__main__":
    main()

