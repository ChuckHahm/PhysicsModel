"""
Evaluation and visualization script for forecast performance metrics.
"""
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import argparse
import os
import logging
from pathlib import Path

from battery_dataset import BatteryTimeSeriesDataset
from voltage_forecast_model import VoltageForecastLSTM, VoltageForecastGRU
import forecast

logger = logging.getLogger(__name__)

# Set style for better looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


def calculate_metrics(y_true, y_pred):
    """
    Calculate performance metrics.
    
    Args:
        y_true: True values
        y_pred: Predicted values
    
    Returns:
        Dictionary of metrics
    """
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    
    # Mean Absolute Percentage Error (MAPE)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    
    # R² score
    r2 = r2_score(y_true, y_pred)
    
    # Mean Error (bias)
    mean_error = np.mean(y_pred - y_true)
    
    # Max error
    max_error = np.max(np.abs(y_pred - y_true))
    
    metrics = {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'MAPE': mape,
        'R2': r2,
        'Mean Error': mean_error,
        'Max Error': max_error
    }
    
    return metrics


def evaluate_model(
    model_path: str,
    csv_path: str,
    sequence_length: int = 10,
    forecast_horizon: int = 1,
    use_temperature: bool = True,
    max_samples: int = None,
    device: str = None
):
    """
    Evaluate model on validation data and return predictions and actuals.
    
    Returns:
        predictions: Array of predictions
        actuals: Array of actual values
        module_ids: Array of module IDs
        metrics: Dictionary of performance metrics
    """
    # Load model
    logger.info("Loading model...")
    model, checkpoint, device = forecast.load_model(model_path, device)
    config = checkpoint['model_config']
    voltage_scaler = checkpoint['voltage_scaler']
    temp_scaler = checkpoint.get('temp_scaler')
    
    # Create validation dataset
    logger.info("Creating validation dataset...")
    val_dataset = BatteryTimeSeriesDataset(
        csv_path=csv_path,
        sequence_length=sequence_length,
        forecast_horizon=forecast_horizon,
        use_temperature=use_temperature,
        normalize=True,
        train=False
    )
    
    # Make predictions
    logger.info("Making predictions...")
    predictions = []
    actuals = []
    module_ids = []
    
    model.eval()
    with torch.no_grad():
        for i in range(len(val_dataset)):
            x, y = val_dataset[i]
            x = x.unsqueeze(0).to(device)
            
            pred_norm = model(x).cpu().numpy()[0, 0]
            actual_norm = y.item()
            
            # Denormalize
            pred = voltage_scaler.inverse_transform([[pred_norm]])[0, 0]
            actual = voltage_scaler.inverse_transform([[actual_norm]])[0, 0]
            
            predictions.append(pred)
            actuals.append(actual)
            
            # Get module_id if available
            if hasattr(val_dataset, 'sequences') and i < len(val_dataset.sequences):
                module_ids.append(val_dataset.sequences[i]['module_id'])
            else:
                module_ids.append(None)
    
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    # Calculate metrics
    metrics = calculate_metrics(actuals, predictions)
    
    logger.info("Evaluation Metrics:")
    for key, value in metrics.items():
        logger.info(f"  {key}: {value:.6f}")
    
    return predictions, actuals, module_ids, metrics


def plot_predicted_vs_actual(predictions, actuals, output_path: str = None):
    """Plot predicted vs actual values."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Scatter plot
    ax.scatter(actuals, predictions, alpha=0.5, s=20)
    
    # Perfect prediction line
    min_val = min(actuals.min(), predictions.min())
    max_val = max(actuals.max(), predictions.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    ax.set_xlabel('Actual Voltage (V)', fontsize=12)
    ax.set_ylabel('Predicted Voltage (V)', fontsize=12)
    ax.set_title('Predicted vs Actual Voltage', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add R² to plot
    r2 = r2_score(actuals, predictions)
    ax.text(0.05, 0.95, f'R² = {r2:.4f}', transform=ax.transAxes,
            fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved plot to: {output_path}")
    else:
        plt.show()
    plt.close()


def plot_residuals(predictions, actuals, output_path: str = None):
    """Plot residuals (errors) vs predicted values."""
    residuals = predictions - actuals
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 10))
    
    # Residuals vs Predicted
    axes[0].scatter(predictions, residuals, alpha=0.5, s=20)
    axes[0].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[0].set_xlabel('Predicted Voltage (V)', fontsize=12)
    axes[0].set_ylabel('Residuals (Predicted - Actual)', fontsize=12)
    axes[0].set_title('Residuals vs Predicted Values', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # Histogram of residuals
    axes[1].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
    axes[1].axvline(x=0, color='r', linestyle='--', lw=2)
    axes[1].set_xlabel('Residuals (Predicted - Actual)', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title('Distribution of Residuals', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # Add statistics
    mean_residual = np.mean(residuals)
    std_residual = np.std(residuals)
    axes[1].text(0.05, 0.95, f'Mean: {mean_residual:.4f}\nStd: {std_residual:.4f}',
                transform=axes[1].transAxes, fontsize=11,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved plot to: {output_path}")
    else:
        plt.show()
    plt.close()


def plot_error_distribution(predictions, actuals, output_path: str = None):
    """Plot distribution of absolute errors."""
    errors = np.abs(predictions - actuals)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Histogram
    axes[0].hist(errors, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    axes[0].axvline(x=np.mean(errors), color='r', linestyle='--', lw=2, label=f'Mean: {np.mean(errors):.4f}')
    axes[0].axvline(x=np.median(errors), color='g', linestyle='--', lw=2, label=f'Median: {np.median(errors):.4f}')
    axes[0].set_xlabel('Absolute Error (V)', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('Distribution of Absolute Errors', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Box plot
    axes[1].boxplot(errors, vert=True)
    axes[1].set_ylabel('Absolute Error (V)', fontsize=12)
    axes[1].set_title('Box Plot of Absolute Errors', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved plot to: {output_path}")
    else:
        plt.show()
    plt.close()


def plot_metrics_summary(metrics, output_path: str = None):
    """Plot summary of performance metrics."""
    # Extract metrics for plotting (exclude R2 as it's on different scale)
    metric_names = ['MAE', 'RMSE', 'MAPE', 'Max Error']
    metric_values = [metrics[name] for name in metric_names]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars = ax.bar(metric_names, metric_values, color=['steelblue', 'coral', 'lightgreen', 'orange'])
    ax.set_ylabel('Value', fontsize=12)
    ax.set_title('Performance Metrics Summary', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, value in zip(bars, metric_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{value:.4f}', ha='center', va='bottom', fontsize=10)
    
    # Add R² as text
    ax.text(0.02, 0.98, f'R² Score: {metrics["R2"]:.4f}', transform=ax.transAxes,
           fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved plot to: {output_path}")
    else:
        plt.show()
    plt.close()


def plot_time_series_sample(predictions, actuals, module_ids, n_samples: int = 5, output_path: str = None):
    """
    Plot time series samples for a few batteries.
    Note: This is a simplified version - for full time series, we'd need the full history.
    """
    # Group by module_id if available
    if module_ids and module_ids[0] is not None:
        df = pd.DataFrame({
            'ModuleId': module_ids,
            'Predicted': predictions,
            'Actual': actuals
        })
        
        unique_modules = df['ModuleId'].unique()[:n_samples]
        n_modules = len(unique_modules)
        
        fig, axes = plt.subplots(n_modules, 1, figsize=(12, 3 * n_modules))
        if n_modules == 1:
            axes = [axes]
        
        for idx, mod_id in enumerate(unique_modules):
            module_data = df[df['ModuleId'] == mod_id]
            axes[idx].plot(module_data['Actual'].values, 'o-', label='Actual', linewidth=2, markersize=4)
            axes[idx].plot(module_data['Predicted'].values, 's-', label='Predicted', linewidth=2, markersize=4)
            axes[idx].set_ylabel('Voltage (V)', fontsize=10)
            axes[idx].set_title(f'Module {mod_id}', fontsize=12, fontweight='bold')
            axes[idx].legend()
            axes[idx].grid(True, alpha=0.3)
        
        axes[-1].set_xlabel('Sample Index', fontsize=12)
        plt.suptitle('Sample Time Series Predictions', fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved plot to: {output_path}")
        else:
            plt.show()
        plt.close()
    else:
        logger.warning("Module IDs not available for time series plotting")


def create_evaluation_report(
    model_path: str,
    csv_path: str,
    output_dir: str = "./evaluation",
    sequence_length: int = 10,
    forecast_horizon: int = 1,
    use_temperature: bool = True,
    max_samples: int = None,
    device: str = None
):
    """
    Create comprehensive evaluation report with all visualizations.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Evaluate model
    predictions, actuals, module_ids, metrics = evaluate_model(
        model_path=model_path,
        csv_path=csv_path,
        sequence_length=sequence_length,
        forecast_horizon=forecast_horizon,
        use_temperature=use_temperature,
        max_samples=max_samples,
        device=device
    )
    
    # Create visualizations
    logger.info("Creating visualizations...")
    
    plot_predicted_vs_actual(predictions, actuals, 
                            os.path.join(output_dir, 'predicted_vs_actual.png'))
    
    plot_residuals(predictions, actuals,
                  os.path.join(output_dir, 'residuals.png'))
    
    plot_error_distribution(predictions, actuals,
                          os.path.join(output_dir, 'error_distribution.png'))
    
    plot_metrics_summary(metrics,
                        os.path.join(output_dir, 'metrics_summary.png'))
    
    plot_time_series_sample(predictions, actuals, module_ids,
                          output_path=os.path.join(output_dir, 'time_series_samples.png'))
    
    # Save metrics to CSV
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(os.path.join(output_dir, 'metrics.csv'), index=False)
    logger.info(f"Metrics saved to: {os.path.join(output_dir, 'metrics.csv')}")
    
    # Save predictions and actuals
    results_df = pd.DataFrame({
        'Predicted': predictions,
        'Actual': actuals,
        'Error': predictions - actuals,
        'Absolute Error': np.abs(predictions - actuals),
        'ModuleId': module_ids
    })
    results_df.to_csv(os.path.join(output_dir, 'predictions.csv'), index=False)
    logger.info(f"Predictions saved to: {os.path.join(output_dir, 'predictions.csv')}")
    
    logger.info(f"\nEvaluation report created in: {output_dir}")
    
    return metrics, predictions, actuals


def main():
    parser = argparse.ArgumentParser(description='Evaluate model and create performance visualizations')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--csv_path', type=str, default='data/VoltTemp.csv',
                       help='Path to VoltTemp.csv file')
    parser.add_argument('--output_dir', type=str, default='./evaluation',
                       help='Directory to save evaluation results')
    parser.add_argument('--sequence_length', type=int, default=10,
                       help='Input sequence length (should match training)')
    parser.add_argument('--forecast_horizon', type=int, default=1,
                       help='Forecast horizon')
    parser.add_argument('--use_temperature', action='store_true', default=True,
                       help='Use temperature as feature')
    parser.add_argument('--no_temperature', dest='use_temperature', action='store_false',
                       help='Do not use temperature as feature')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum number of samples to evaluate')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    create_evaluation_report(
        model_path=args.model_path,
        csv_path=args.csv_path,
        output_dir=args.output_dir,
        sequence_length=args.sequence_length,
        forecast_horizon=args.forecast_horizon,
        use_temperature=args.use_temperature,
        max_samples=args.max_samples,
        device=args.device
    )


if __name__ == "__main__":
    main()

