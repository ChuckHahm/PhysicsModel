# CLAUDE.md

## Project Overview

Battery voltage forecasting system using PyTorch LSTM/GRU neural networks. Predicts battery voltage decrease over time using historical voltage and temperature data from lithium-ion batteries.

## Project Structure

```
scripts/           - Main Python source code
  battery_dataset.py       - Custom PyTorch Dataset (BatteryTimeSeriesDataset)
  voltage_forecast_model.py - LSTM/GRU model definitions
  train.py                 - Training with MLflow tracking, early stopping
  forecast.py              - Voltage forecast generation
  evaluate.py              - Model evaluation and visualization
  optimize.py              - Hyperparameter optimization (Optuna)
  device_utils.py          - GPU compatibility utilities
  run_tests.py             - Test runner script
data/              - CSV and NumPy data files (VoltTemp.csv is primary)
models/            - Trained model checkpoints (.pth)
optimization/      - Optuna optimization outputs
notebooks/         - Jupyter notebooks for exploration/analysis
tests/             - pytest test suite
doc/               - Documentation
mlruns/            - MLflow experiment artifacts
```

## Development Commands

### Install dependencies
```bash
pip install -r requirements.txt
```

### Training
```bash
python scripts/train.py --csv_path data/VoltTemp.csv --output_dir ./models
```

### Quick training (limited samples)
```bash
python scripts/train.py --csv_path data/VoltTemp.csv --output_dir ./models --max_samples 10000 --num_epochs 20
```

### Hyperparameter optimization
```bash
python scripts/optimize.py --csv_path data/VoltTemp.csv --output_dir ./optimization --n_trials 50
```

### Forecasting
```bash
python scripts/forecast.py --model_path ./models/best_model.pth --csv_path data/VoltTemp.csv --output_file forecasts.csv
```

### Evaluation
```bash
python scripts/evaluate.py --model_path ./models/best_model.pth --csv_path data/VoltTemp.csv --output_dir ./evaluation
```

### Run tests
```bash
pytest tests/
```

### MLflow UI
```bash
mlflow ui --port 5000
```

## Data Format

Primary data file: `data/VoltTemp.csv`
- `ModuleId`: Battery identifier
- `SeriesId`: Time interval identifier (sequential per battery)
- `CurrentVoltage`: Voltage values (target variable)
- `CurrentTemperature`: Temperature values (feature)

## Key Architecture Details

- Missing temperature values (-1) are imputed via forward/backward fill per battery
- Features standardized with StandardScaler (saved with model checkpoints)
- Train/validation split at battery level (not sequence level) to prevent data leakage
- GPU auto-detection with CPU fallback for older GPUs (requires CUDA sm_70+)
- Use `device_utils.get_compatible_device()` for device handling

## Code Conventions

- Type hints on function parameters and return types
- PEP 8 style
- Docstrings on classes and functions
- MLflow tracking for all hyperparameters and metrics
- Model checkpoints (.pth) include weights, scalers, and config
