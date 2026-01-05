"""
Optuna hyperparameter optimization script for battery voltage forecasting.
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
import logging
import joblib

# Conditional Optuna import
try:
    import optuna
    from optuna.trial import TrialState
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    optuna = None
    TrialState = None

from battery_dataset import BatteryTimeSeriesDataset
from voltage_forecast_model import VoltageForecastLSTM, VoltageForecastGRU
import train as train_module

logger = logging.getLogger(__name__)


def train_and_evaluate(
    csv_path: str,
    trial: optuna.Trial,
    sequence_length: int,
    forecast_horizon: int,
    use_temperature: bool,
    model_type: str,
    hidden_size: int,
    num_layers: int,
    dropout: float,
    batch_size: int,
    learning_rate: float,
    num_epochs: int,
    patience: int,
    device: str,
    max_samples: int = None,
    n_jobs: int = 1
) -> float:
    """
    Train a model with given hyperparameters and return validation loss.
    
    Returns:
        Best validation loss
    """
    # Create datasets
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
        num_workers=n_jobs,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=n_jobs,
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
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=False
    )
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # Train
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            predictions = model(x)
            loss = criterion(predictions, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        # Validate
        model.eval()
        val_loss = 0.0
        num_batches = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                predictions = model(x)
                loss = criterion(predictions, y)
                val_loss += loss.item()
                num_batches += 1
        
        val_loss = val_loss / num_batches if num_batches > 0 else float('inf')
        scheduler.step(val_loss)
        
        # Report intermediate value for pruning
        trial.report(val_loss, epoch)
        
        # Handle pruning based on the intermediate value
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
        
        # Update best validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= patience:
            break
    
    return best_val_loss


def objective(trial: optuna.Trial, args) -> float:
    """
    Optuna objective function.
    """
    # Suggest hyperparameters
    sequence_length = trial.suggest_int('sequence_length', 5, 30, step=1)
    hidden_size = trial.suggest_int('hidden_size', 32, 256, step=16)
    num_layers = trial.suggest_int('num_layers', 1, 4, step=1)
    dropout = trial.suggest_float('dropout', 0.1, 0.5, step=0.1)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    model_type = trial.suggest_categorical('model_type', ['LSTM', 'GRU'])
    
    # Set device
    device = get_compatible_device(args.device)
    
    # Log suggested parameters
    logger.info(f"Trial {trial.number}: sequence_length={sequence_length}, "
                f"hidden_size={hidden_size}, num_layers={num_layers}, "
                f"dropout={dropout:.2f}, batch_size={batch_size}, "
                f"learning_rate={learning_rate:.6f}, model_type={model_type}")
    
    try:
        val_loss = train_and_evaluate(
            csv_path=args.csv_path,
            trial=trial,
            sequence_length=sequence_length,
            forecast_horizon=args.forecast_horizon,
            use_temperature=args.use_temperature,
            model_type=model_type,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_size=batch_size,
            learning_rate=learning_rate,
            num_epochs=args.num_epochs,
            patience=args.patience,
            device=device,
            max_samples=args.max_samples,
            n_jobs=args.n_jobs
        )
        
        return val_loss
    except Exception as e:
        logger.error(f"Trial {trial.number} failed: {e}")
        raise


def optimize(
    csv_path: str,
    output_dir: str = "./optimization",
    n_trials: int = 50,
    n_jobs: int = 1,
    study_name: str = "battery_voltage_optimization",
    csv_path_study: str = None,
    **kwargs
):
    """
    Run Optuna optimization.
    
    Args:
        csv_path: Path to data CSV
        output_dir: Directory to save optimization results
        n_trials: Number of optimization trials
        n_jobs: Number of parallel jobs
        study_name: Name of the Optuna study
        csv_path_study: Path to save study database (None for in-memory)
        **kwargs: Additional arguments for training
    
    Raises:
        ImportError: If Optuna is not installed
    """
    if not OPTUNA_AVAILABLE:
        raise ImportError(
            "Optuna is required for optimization but is not installed. "
            "Install it with: pip install optuna"
        )
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Create or load study
    if csv_path_study:
        study = optuna.create_study(
            direction='minimize',
            storage=f'sqlite:///{csv_path_study}',
            study_name=study_name,
            load_if_exists=True,
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)
        )
    else:
        study = optuna.create_study(
            study_name=study_name,
            direction='minimize',
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)
        )
    
    logger.info(f"Starting optimization with {n_trials} trials...")
    logger.info(f"Study name: {study_name}")
    
    # Create args object for objective function
    class Args:
        pass
    args = Args()
    for key, value in kwargs.items():
        setattr(args, key, value)
    args.csv_path = csv_path
    
    # Optimize
    study.optimize(
        lambda trial: objective(trial, args),
        n_trials=n_trials,
        n_jobs=n_jobs,
        show_progress_bar=True
    )
    
    # Print results
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
    
    logger.info("Optimization finished!")
    logger.info(f"Number of finished trials: {len(study.trials)}")
    logger.info(f"Number of pruned trials: {len(pruned_trials)}")
    logger.info(f"Number of complete trials: {len(complete_trials)}")
    
    logger.info("\nBest trial:")
    trial = study.best_trial
    logger.info(f"  Value (validation loss): {trial.value:.6f}")
    logger.info("\n  Params:")
    for key, value in trial.params.items():
        logger.info(f"    {key}: {value}")
    
    # Save best parameters
    best_params_path = os.path.join(output_dir, 'best_params.json')
    import json
    with open(best_params_path, 'w') as f:
        json.dump(trial.params, f, indent=2)
    logger.info(f"\nBest parameters saved to: {best_params_path}")
    
    # Save study
    study_path = os.path.join(output_dir, 'study.pkl')
    joblib.dump(study, study_path)
    logger.info(f"Study saved to: {study_path}")
    
    # Create optimization report
    try:
        import optuna.visualization as vis
        
        # Plot optimization history
        fig = vis.plot_optimization_history(study)
        fig.write_image(os.path.join(output_dir, 'optimization_history.png'))
        
        # Plot parameter importances
        try:
            fig = vis.plot_param_importances(study)
            fig.write_image(os.path.join(output_dir, 'param_importances.png'))
        except:
            logger.warning("Could not generate parameter importances plot")
        
        # Plot parallel coordinate
        try:
            fig = vis.plot_parallel_coordinate(study)
            fig.write_image(os.path.join(output_dir, 'parallel_coordinate.png'))
        except:
            logger.warning("Could not generate parallel coordinate plot")
        
        logger.info(f"Optimization plots saved to: {output_dir}")
    except ImportError:
        logger.warning("Plotly/Kaleido not available. Skipping visualization generation.")
        logger.info("Install with: pip install plotly kaleido")
    
    return study, trial


def main():
    parser = argparse.ArgumentParser(description='Optimize hyperparameters using Optuna')
    parser.add_argument('--csv_path', type=str, default='data/VoltTemp.csv',
                       help='Path to VoltTemp.csv file')
    parser.add_argument('--output_dir', type=str, default='./optimization',
                       help='Directory to save optimization results')
    parser.add_argument('--n_trials', type=int, default=50,
                       help='Number of optimization trials')
    parser.add_argument('--n_jobs', type=int, default=1,
                       help='Number of parallel jobs')
    parser.add_argument('--study_name', type=str, default='battery_voltage_optimization',
                       help='Name of the Optuna study')
    parser.add_argument('--study_db', type=str, default=None,
                       help='Path to SQLite database for study persistence')
    parser.add_argument('--sequence_length', type=int, default=10,
                       help='Fixed sequence length (if not optimizing)')
    parser.add_argument('--forecast_horizon', type=int, default=1,
                       help='Steps ahead to forecast')
    parser.add_argument('--use_temperature', action='store_true', default=True,
                       help='Use temperature as feature')
    parser.add_argument('--no_temperature', dest='use_temperature', action='store_false',
                       help='Do not use temperature as feature')
    parser.add_argument('--num_epochs', type=int, default=30,
                       help='Number of epochs per trial')
    parser.add_argument('--patience', type=int, default=7,
                       help='Early stopping patience')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum number of training sequences per trial')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--train_final', action='store_true',
                       help='Train final model with best parameters after optimization')
    parser.add_argument('--final_output_dir', type=str, default='./models',
                       help='Directory to save final trained model')
    parser.add_argument('--skip_optimization', action='store_true',
                       help='Skip Optuna optimization and train a single model with default/specified parameters')
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Skip optimization if requested
    if args.skip_optimization:
        logger.info("Skipping Optuna optimization, training single model with specified/default parameters...")
        
        # Use default parameters or specified ones
        default_params = {
            'sequence_length': args.sequence_length,
            'hidden_size': 64,
            'num_layers': 2,
            'dropout': 0.2,
            'batch_size': 32,
            'learning_rate': 0.001,
            'model_type': 'LSTM'
        }
        
        # Train final model with default parameters
        logger.info("Training model with parameters:")
        for key, value in default_params.items():
            logger.info(f"  {key}: {value}")
        
        train_module.train(
            csv_path=args.csv_path,
            output_dir=args.final_output_dir,
            sequence_length=default_params['sequence_length'],
            forecast_horizon=args.forecast_horizon,
            use_temperature=args.use_temperature,
            model_type=default_params['model_type'],
            hidden_size=default_params['hidden_size'],
            num_layers=default_params['num_layers'],
            dropout=default_params['dropout'],
            batch_size=default_params['batch_size'],
            learning_rate=default_params['learning_rate'],
            num_epochs=args.num_epochs,
            patience=args.patience,
            device=args.device,
            max_samples=args.max_samples
        )
        return
    
    # Check if Optuna is available
    if not OPTUNA_AVAILABLE:
        logger.error("Optuna is required for optimization but is not installed.")
        logger.error("Install it with: pip install optuna")
        logger.error("Or use --skip_optimization to train without optimization.")
        return
    
    # Run optimization
    study, best_trial = optimize(
        csv_path=args.csv_path,
        output_dir=args.output_dir,
        n_trials=args.n_trials,
        n_jobs=args.n_jobs,
        study_name=args.study_name,
        csv_path_study=args.study_db,
        forecast_horizon=args.forecast_horizon,
        use_temperature=args.use_temperature,
        num_epochs=args.num_epochs,
        patience=args.patience,
        max_samples=args.max_samples,
        device=args.device
    )
    
    # Train final model if requested
    if args.train_final:
        logger.info("\nTraining final model with best parameters...")
        train_module.train(
            csv_path=args.csv_path,
            output_dir=args.final_output_dir,
            sequence_length=best_trial.params['sequence_length'],
            forecast_horizon=args.forecast_horizon,
            use_temperature=args.use_temperature,
            model_type=best_trial.params['model_type'],
            hidden_size=best_trial.params['hidden_size'],
            num_layers=best_trial.params['num_layers'],
            dropout=best_trial.params['dropout'],
            batch_size=best_trial.params['batch_size'],
            learning_rate=best_trial.params['learning_rate'],
            num_epochs=args.num_epochs * 2,  # Use more epochs for final training
            patience=args.patience,
            device=args.device,
            max_samples=None  # Use full dataset for final training
        )


if __name__ == "__main__":
    main()

