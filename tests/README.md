# Test Suite Documentation

This directory contains comprehensive tests for the Battery Voltage Forecasting project. Each test logs the time duration for individual steps and operations.

## Test Structure

- `conftest.py`: Shared fixtures and timing utilities
- `test_battery_dataset.py`: Tests for data loading and preprocessing
- `test_voltage_forecast_model.py`: Tests for model architecture
- `test_train.py`: Tests for training pipeline
- `test_forecast.py`: Tests for forecasting functionality
- `test_evaluate.py`: Tests for model evaluation
- `test_optimize.py`: Tests for hyperparameter optimization
- `test_device_utils.py`: Tests for device detection utilities

## Running Tests

### Run all tests:
```bash
python run_tests.py
```

### Run specific test file:
```bash
pytest tests/test_battery_dataset.py -v
```

### Run specific test:
```bash
pytest tests/test_battery_dataset.py::TestBatteryTimeSeriesDataset::test_dataset_initialization -v
```

### Run with timing summary:
```bash
pytest tests/ -v --tb=short
```

## Timing Information

Each test automatically logs:
- ⏱️ Individual test duration
- ⏱️ Checkpoint timings within tests
- Summary of fastest/slowest tests at the end

The timing plugin tracks:
- Total test execution time
- Average test time
- Slowest 10 tests
- Fastest 10 tests

## Test Coverage

### Data Pipeline (`test_battery_dataset.py`)
- Dataset initialization
- Data loading from CSV
- Sequence creation
- Train/validation split
- Feature normalization
- Missing value handling
- DataLoader integration

### Model Architecture (`test_voltage_forecast_model.py`)
- LSTM initialization and forward pass
- GRU initialization and forward pass
- Different configurations (hidden sizes, layers)
- Gradient flow verification
- Parameter counting

### Training (`test_train.py`)
- Training epoch function
- Validation function
- Full training pipeline
- Early stopping
- Model checkpointing
- Different model types (LSTM/GRU)
- With/without temperature features

### Forecasting (`test_forecast.py`)
- Model loading
- Voltage forecasting for all batteries
- Forecasting for specific batteries
- Output file generation
- Output structure validation

### Evaluation (`test_evaluate.py`)
- Metrics calculation (MAE, MSE, RMSE, R², etc.)
- Model evaluation on validation set
- Evaluation report generation
- Output file validation

### Device Utils (`test_device_utils.py`)
- Device detection
- CPU/CUDA handling
- GPU compatibility fallback

## Test Fixtures

- `sample_csv_path`: Provides test data (uses real data if available, otherwise creates synthetic)
- `temp_model_dir`: Temporary directory for model outputs
- `temp_output_dir`: Temporary directory for test outputs
- `timer`: Per-test timer for tracking operation durations

## Notes

- Tests use synthetic data if `data/VoltTemp.csv` doesn't exist
- MLflow is disabled in tests for faster execution
- Tests use CPU device by default for consistency
- Model training uses minimal epochs and samples for speed
- Test outputs are cleaned up automatically via pytest fixtures

