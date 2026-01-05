# Battery Voltage Forecasting

This project implements a PyTorch-based time-series forecasting system for predicting battery voltage decrease over time using LSTM/GRU neural networks. The model uses historical voltage and temperature data from lithium-ion batteries.

## Features

- **PyTorch Dataset Framework**: Custom `BatteryTimeSeriesDataset` class for efficient data loading and processing
- **LSTM/GRU Models**: Deep learning models for time-series forecasting
- **Temperature Integration**: Uses both voltage and temperature as input features
- **Data Preprocessing**: Handles missing values, normalization, and sequence creation
- **Training Pipeline**: Complete training script with validation, early stopping, and checkpointing
- **Forecasting**: Script to generate voltage forecasts for existing or new batteries

## Data Structure

The input CSV file (`data/VoltTemp.csv`) should have the following columns:
- `ModuleId`: Battery identifier
- `SeriesId`: Time interval identifier (sequential within each battery)
- `CurrentVoltage`: Voltage values (target variable)0
- `CurrentTemperature`: Temperature values (feature)

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure you have the data file at `data/VoltTemp.csv`

## Usage

### Training

Train a model with default parameters:
```bash
python train.py --csv_path data/VoltTemp.csv --output_dir ./models
```

Train with custom parameters:
```bash
python train.py \
    --csv_path data/VoltTemp.csv \
    --output_dir ./models \
    --sequence_length 20 \
    --hidden_size 128 \
    --num_layers 3 \
    --batch_size 64 \
    --learning_rate 0.0001 \
    --num_epochs 100 \
    --model_type LSTM \
    --use_temperature
```

Train with limited samples for faster experimentation:
```bash
python train.py \
    --csv_path data/VoltTemp.csv \
    --output_dir ./models \
    --max_samples 10000 \
    --num_epochs 20
```

### Hyperparameter Optimization with Optuna

Use Optuna to automatically find optimal hyperparameters:

```bash
python optimize.py \
    --csv_path data/VoltTemp.csv \
    --output_dir ./optimization \
    --n_trials 50 \
    --num_epochs 20 \
    --max_samples 5000
```

Optimize and automatically train final model:
```bash
python optimize.py \
    --csv_path data/VoltTemp.csv \
    --output_dir ./optimization \
    --n_trials 50 \
    --num_epochs 20 \
    --train_final \
    --final_output_dir ./models
```

Skip optimization and train with default parameters (Optuna not required):
```bash
python optimize.py \
    --csv_path data/VoltTemp.csv \
    --skip_optimization \
    --final_output_dir ./models \
    --num_epochs 20
```

**Optuna Parameters:**
- `--n_trials`: Number of optimization trials (default: 50)
- `--study_name`: Name of the Optuna study (default: "battery_voltage_optimization")
- `--study_db`: Path to SQLite database for study persistence (optional)
- `--n_jobs`: Number of parallel jobs (default: 1)
- `--train_final`: Train final model with best parameters after optimization
- `--final_output_dir`: Directory to save final trained model
- `--skip_optimization`: Skip Optuna optimization and train a single model with default parameters (Optuna not required)
- Other training parameters (--num_epochs, --patience, --max_samples, etc.) apply to each trial

**Note:** Optuna is optional. If Optuna is not installed and you try to run optimization, the script will prompt you to install it. Use `--skip_optimization` to train without Optuna.

**Optimized Hyperparameters:**
- `sequence_length`: 5-30
- `hidden_size`: 32-256 (step 16)
- `num_layers`: 1-4
- `dropout`: 0.1-0.5 (step 0.1)
- `batch_size`: [16, 32, 64, 128]
- `learning_rate`: 1e-5 to 1e-2 (log scale)
- `model_type`: LSTM or GRU

The optimization uses Median Pruner for efficient pruning of poor-performing trials.

**Key Parameters:**
- `--sequence_length`: Number of historical time steps to use as input (default: 10)
- `--forecast_horizon`: Steps ahead to forecast (default: 1)
- `--hidden_size`: LSTM/GRU hidden layer size (default: 64)
- `--num_layers`: Number of LSTM/GRU layers (default: 2)
- `--dropout`: Dropout rate (default: 0.2)
- `--batch_size`: Training batch size (default: 32)
- `--learning_rate`: Learning rate (default: 0.001)
- `--model_type`: "LSTM" or "GRU" (default: "LSTM")
- `--use_temperature`: Include temperature as feature (default: True)
- `--no_temperature`: Exclude temperature features
- `--max_samples`: Maximum number of training sequences (None for all). Useful for faster experimentation (default: None)
- `--mlflow_experiment`: MLflow experiment name (default: "battery_voltage_forecasting")

### Forecasting

Forecast voltage for all batteries:
```bash
python forecast.py \
    --model_path ./models/best_model.pth \
    --csv_path data/VoltTemp.csv \
    --output_file forecasts.csv
```

Forecast for a specific battery:
```bash
python forecast.py \
    --model_path ./models/best_model.pth \
    --csv_path data/VoltTemp.csv \
    --module_id 67115833 \
    --output_file forecast_67115833.csv
```

### Model Evaluation and Visualization

Evaluate model performance and generate comprehensive visualizations:

```bash
python evaluate.py \
    --model_path ./models/best_model.pth \
    --csv_path data/VoltTemp.csv \
    --output_dir ./evaluation
```

**Evaluation Parameters:**
- `--output_dir`: Directory to save evaluation results and plots (default: "./evaluation")
- `--max_samples`: Maximum number of samples to evaluate (None for all)
- Other parameters match training (sequence_length, forecast_horizon, etc.)

**Generated Outputs:**
- `predicted_vs_actual.png`: Scatter plot of predicted vs actual values
- `residuals.png`: Residual plots (errors vs predicted, distribution)
- `error_distribution.png`: Distribution and box plot of absolute errors
- `metrics_summary.png`: Bar chart of performance metrics
- `time_series_samples.png`: Sample time series predictions for individual batteries
- `metrics.csv`: Performance metrics in CSV format
- `predictions.csv`: All predictions with errors

**Performance Metrics Calculated:**
- MAE (Mean Absolute Error)
- MSE (Mean Squared Error)
- RMSE (Root Mean Squared Error)
- MAPE (Mean Absolute Percentage Error)
- R² (Coefficient of Determination)
- Mean Error (Bias)
- Max Error

## Model Architecture

The model uses an LSTM or GRU network with the following structure:
- **Input Layer**: Takes sequences of voltage (and optionally temperature)
- **LSTM/GRU Layers**: 2-3 layers with configurable hidden size
- **Fully Connected Layers**: 3-layer MLP for final prediction
- **Output**: Single value representing forecasted voltage

## Data Processing

1. **Missing Value Handling**: Temperature values of -1 are treated as missing and filled using forward/backward fill within each battery's time series
2. **Normalization**: Features are standardized using StandardScaler
3. **Sequence Creation**: Data is organized into sequences of fixed length for time-series learning
4. **Train/Validation Split**: Batteries are split (not individual sequences) to avoid data leakage

## MLflow Experiment Tracking

The training script automatically tracks experiments using MLflow:

- **Hyperparameters**: All training parameters (sequence length, model type, learning rate, etc.)
- **Metrics**: Training and validation loss for each epoch
- **Artifacts**: Model checkpoints, training curves plots
- **Models**: Best model saved to MLflow model registry

View experiments:
```bash
# Start MLflow UI
mlflow ui

# Or specify a different port
mlflow ui --port 5000
```

Access the MLflow UI at `http://localhost:5000` to compare experiments, view metrics, and download models.

## Output

Training produces:
- `best_model.pth`: Model checkpoint with weights, scalers, and configuration
- `training_curves.png`: Plot of training and validation loss over epochs
- MLflow run with all hyperparameters, metrics, and artifacts

Forecasting produces:
- CSV file with columns:
  - `ModuleId`: Battery identifier
  - `LastSeriesId`: Last time step in history
  - `LastVoltage`: Last observed voltage
  - `LastTemperature`: Last observed temperature
  - `ForecastVoltage`: Predicted future voltage
  - `VoltageChange`: Predicted change in voltage

## Example

```python
from battery_dataset import BatteryTimeSeriesDataset
from torch.utils.data import DataLoader

# Create dataset
dataset = BatteryTimeSeriesDataset(
    csv_path='data/VoltTemp.csv',
    sequence_length=10,
    forecast_horizon=1,
    use_temperature=True,
    train=True
)

# Create data loader
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Iterate over batches
for x, y in loader:
    # x: (batch_size, sequence_length, 2) - voltage and temperature sequences
    # y: (batch_size, 1) - target voltage
    pass
```

## Notes

- The dataset uses battery-level splitting (not sequence-level) to prevent data leakage
- Temperature values of -1 are treated as missing and imputed
- Models are saved with scalers for proper denormalization during inference
- Early stopping is implemented to prevent overfitting
- Gradient clipping is used for training stability

## Requirements

- Python 3.7+
- PyTorch 2.0+
- NumPy, Pandas, Scikit-learn
- Matplotlib, Seaborn, tqdm
- MLflow 2.0+ (for experiment tracking)
- Optuna 3.0+ (for hyperparameter optimization)
- Joblib (for study persistence)

Optional (for visualization):
- Plotly, Kaleido (for Optuna visualization plots)

## GPU Compatibility

The code automatically detects GPU compatibility and falls back to CPU if your GPU is not supported by the current PyTorch installation. PyTorch 2.0+ requires CUDA compute capability sm_70 or higher. If you have an older GPU (e.g., sm_61), the code will automatically use CPU instead, and you won't see compatibility warnings.

To force CPU usage, use the `--device cpu` argument in any script.

