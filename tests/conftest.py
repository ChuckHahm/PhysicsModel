"""
Pytest configuration and shared fixtures with timing utilities.
"""
import pytest
import time
import os
import pandas as pd
import numpy as np
import tempfile
from pathlib import Path
from typing import Generator
import logging

# Set up logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Global storage for test timings
_test_timings = {}


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item):
    """Called before each test runs."""
    item.start_time = time.time()


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_teardown(item):
    """Called after each test runs."""
    if hasattr(item, 'start_time'):
        duration = time.time() - item.start_time
        test_name = item.nodeid
        _test_timings[test_name] = duration
        logger.info(f"⏱️  {test_name}: {duration:.4f}s")




@pytest.fixture(scope="function")
def timer():
    """Fixture to time individual operations within a test."""
    class Timer:
        def __init__(self):
            self.start_time = None
            self.checkpoints = []
        
        def start(self):
            """Start timing."""
            self.start_time = time.time()
            return self
        
        def checkpoint(self, name: str):
            """Record a checkpoint."""
            if self.start_time is None:
                self.start()
            elapsed = time.time() - self.start_time
            self.checkpoints.append((name, elapsed))
            logger.info(f"  ⏱️  Checkpoint '{name}': {elapsed:.4f}s")
            return elapsed
        
        def elapsed(self):
            """Get total elapsed time."""
            if self.start_time is None:
                return 0.0
            return time.time() - self.start_time
    
    return Timer()


@pytest.fixture(scope="session")
def test_data_dir():
    """Get the test data directory."""
    return Path(__file__).parent.parent / "data"


@pytest.fixture(scope="session")
def sample_csv_path(test_data_dir, tmp_path_factory):
    """Create a sample CSV file for testing."""
    # Check if real data exists
    real_csv = test_data_dir / "VoltTemp.csv"
    if real_csv.exists():
        return str(real_csv)
    
    # Create synthetic test data
    tmp_dir = tmp_path_factory.mktemp("test_data")
    csv_path = tmp_dir / "test_VoltTemp.csv"
    
    # Generate synthetic battery data
    np.random.seed(42)
    data = []
    
    # Create 5 batteries with different characteristics
    for module_id in range(1, 6):
        n_points = np.random.randint(20, 50)
        base_voltage = np.random.uniform(3.5, 4.2)
        base_temp = np.random.uniform(20, 30)
        
        for series_id in range(1, n_points + 1):
            # Simulate voltage decrease over time
            voltage = base_voltage - (series_id / n_points) * 0.1 + np.random.normal(0, 0.01)
            # Simulate temperature variation
            temp = base_temp + np.random.normal(0, 2)
            
            data.append({
                'ModuleId': module_id,
                'SeriesId': series_id,
                'CurrentVoltage': voltage,
                'CurrentTemperature': temp
            })
    
    df = pd.DataFrame(data)
    df.to_csv(csv_path, index=False)
    logger.info(f"Created synthetic test data at {csv_path}")
    
    return str(csv_path)


@pytest.fixture(scope="function")
def temp_model_dir(tmp_path):
    """Create a temporary directory for model outputs."""
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    return str(model_dir)


@pytest.fixture(scope="function")
def temp_output_dir(tmp_path):
    """Create a temporary directory for test outputs."""
    output_dir = tmp_path / "test_outputs"
    output_dir.mkdir()
    return str(output_dir)


def pytest_sessionfinish(session, exitstatus):
    """Print timing summary at the end of test session."""
    if _test_timings:
        logger.info("\n" + "="*80)
        logger.info("TEST TIMING SUMMARY")
        logger.info("="*80)
        
        # Sort by duration
        sorted_timings = sorted(_test_timings.items(), key=lambda x: x[1], reverse=True)
        
        total_time = sum(_test_timings.values())
        logger.info(f"\nTotal test time: {total_time:.4f}s")
        logger.info(f"Number of tests: {len(_test_timings)}")
        if len(_test_timings) > 0:
            logger.info(f"Average test time: {total_time/len(_test_timings):.4f}s")
        
        logger.info("\nSlowest tests:")
        for test_name, duration in sorted_timings[:10]:
            logger.info(f"  {duration:.4f}s - {test_name}")
        
        if len(sorted_timings) > 10:
            logger.info("\nFastest tests:")
            for test_name, duration in sorted_timings[-10:]:
                logger.info(f"  {duration:.4f}s - {test_name}")
        
        logger.info("="*80 + "\n")

