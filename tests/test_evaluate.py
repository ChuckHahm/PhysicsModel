"""
Tests for evaluate.py - Model evaluation and metrics.
"""
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from evaluate import (
    calculate_metrics,
    evaluate_model,
    create_evaluation_report
)
from train import train


class TestMetricsCalculation:
    """Test suite for metrics calculation."""
    
    def test_calculate_metrics(self, timer):
        """Test metrics calculation."""
        timer.start()
        # Create dummy predictions and actuals
        np.random.seed(42)
        y_true = np.random.randn(100) + 10
        y_pred = y_true + np.random.randn(100) * 0.1
        
        metrics = calculate_metrics(y_true, y_pred)
        timer.checkpoint("Metrics calculation")
        
        assert 'MAE' in metrics
        assert 'MSE' in metrics
        assert 'RMSE' in metrics
        assert 'MAPE' in metrics
        assert 'R2' in metrics
        assert 'Mean Error' in metrics
        assert 'Max Error' in metrics
        
        # Check that metrics are reasonable
        assert metrics['MAE'] >= 0
        assert metrics['RMSE'] >= 0
        assert metrics['R2'] <= 1.0
    
    def test_perfect_predictions(self, timer):
        """Test metrics with perfect predictions."""
        timer.start()
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = y_true.copy()
        
        metrics = calculate_metrics(y_true, y_pred)
        timer.checkpoint("Perfect predictions")
        
        assert metrics['MAE'] == 0.0
        assert metrics['MSE'] == 0.0
        assert metrics['RMSE'] == 0.0
        assert abs(metrics['R2'] - 1.0) < 1e-6


class TestModelEvaluation:
    """Test suite for model evaluation."""
    
    def test_evaluate_model(self, sample_csv_path, temp_model_dir, timer):
        """Test model evaluation."""
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
        
        # Evaluate model
        predictions, actuals, module_ids, metrics = evaluate_model(
            model_path=str(model_path),
            csv_path=sample_csv_path,
            sequence_length=10,
            max_samples=10000,
            device='cpu'
        )
        timer.checkpoint("Model evaluation")
        
        assert len(predictions) > 0
        assert len(actuals) > 0
        assert len(predictions) == len(actuals)
        assert isinstance(metrics, dict)
        assert 'MAE' in metrics
    
    def test_evaluate_model_without_temperature(self, sample_csv_path, temp_model_dir, timer):
        """Test evaluation with model trained without temperature."""
        timer.start()
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
        
        predictions, actuals, _, metrics = evaluate_model(
            model_path=str(model_path),
            csv_path=sample_csv_path,
            use_temperature=False,
            sequence_length=10,
            max_samples=10000,
            device='cpu'
        )
        timer.checkpoint("Evaluation")
        
        assert len(predictions) > 0
        assert 'MAE' in metrics


class TestEvaluationReport:
    """Test suite for evaluation report generation."""
    
    def test_create_evaluation_report(self, sample_csv_path, temp_model_dir, temp_output_dir, timer):
        """Test creating full evaluation report."""
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
        
        model_path = Path(temp_model_dir) / "best_model.pth"
        
        # Create evaluation report
        metrics, predictions, actuals = create_evaluation_report(
            model_path=str(model_path),
            csv_path=sample_csv_path,
            output_dir=temp_output_dir,
            sequence_length=10,
            max_samples=10000,
            device='cpu'
        )
        timer.checkpoint("Report creation")
        
        assert metrics is not None
        assert len(predictions) > 0
        assert len(actuals) > 0
        
        # Check that output files were created
        output_dir = Path(temp_output_dir)
        assert (output_dir / "metrics.csv").exists()
        assert (output_dir / "predictions.csv").exists()
        
        # Check metrics file
        metrics_df = pd.read_csv(output_dir / "metrics.csv")
        assert len(metrics_df) > 0
        assert 'MAE' in metrics_df.columns
    
    def test_evaluation_report_files(self, sample_csv_path, temp_model_dir, temp_output_dir, timer):
        """Test that all expected files are created."""
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
        
        create_evaluation_report(
            model_path=str(model_path),
            csv_path=sample_csv_path,
            output_dir=temp_output_dir,
            sequence_length=10,
            max_samples=10000,
            device='cpu'
        )
        timer.checkpoint("Report creation")
        
        output_dir = Path(temp_output_dir)
        
        # Check for CSV files
        assert (output_dir / "metrics.csv").exists()
        assert (output_dir / "predictions.csv").exists()
        
        # Check CSV contents
        predictions_df = pd.read_csv(output_dir / "predictions.csv")
        assert len(predictions_df) > 0
        assert 'Predicted' in predictions_df.columns
        assert 'Actual' in predictions_df.columns
        assert 'Error' in predictions_df.columns

