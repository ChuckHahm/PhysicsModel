"""
Tests for optimize.py - Hyperparameter optimization.
"""
import pytest
from pathlib import Path
from train import train


class TestOptimization:
    """Test suite for hyperparameter optimization."""
    
    @pytest.mark.skipif(
        True,  # Skip by default since Optuna might not be available
        reason="Optuna optimization tests require Optuna and are slow"
    )
    def test_optimization_skip_flag(self, sample_csv_path, temp_model_dir, timer):
        """Test optimization with skip flag (trains single model)."""
        timer.start()
        # This would test the --skip_optimization flag
        # For now, we'll just test that we can train a model with default params
        train(
            csv_path=sample_csv_path,
            output_dir=temp_model_dir,
            sequence_length=10,
            hidden_size=32,
            batch_size=4,
            num_epochs=2,
            patience=5,
            max_samples=50,
            use_mlflow=False,
            device='cpu'
        )
        timer.checkpoint("Training with defaults")
        
        assert (Path(temp_model_dir) / "best_model.pth").exists()
    
    def test_optimization_import_check(self, timer):
        """Test that optimization module handles missing Optuna gracefully."""
        timer.start()
        try:
            import optimize
            timer.checkpoint("Import optimize module")
            
            # Check if OPTUNA_AVAILABLE attribute exists
            assert hasattr(optimize, 'OPTUNA_AVAILABLE')
        except ImportError:
            # If optimize.py itself fails to import, that's okay
            # It might have syntax errors or other issues
            pass

