#!/usr/bin/env python3
"""
Test runner script for Battery Voltage Forecasting project.
Runs all tests with timing information and generates a summary report.
"""
import subprocess
import sys
import time
from pathlib import Path


def main():
    """Run pytest with timing and generate summary."""
    print("="*80)
    print("Battery Voltage Forecasting - Test Suite")
    print("="*80)
    print()
    
    # Get test directory
    test_dir = Path(__file__).parent / "tests"
    
    # Build pytest command
    cmd = [
        sys.executable, "-m", "pytest",
        str(test_dir),
        "-v",  # Verbose output
        "--tb=short",  # Short traceback format
        "-s",  # Don't capture output (show print statements)
        "--color=yes",  # Colored output
    ]
    
    # Add timing plugin
    cmd.extend(["-p", "no:cacheprovider"])  # Disable cache to avoid conflicts
    
    print(f"Running tests from: {test_dir}")
    print(f"Command: {' '.join(cmd)}")
    print()
    print("-"*80)
    
    # Run tests
    start_time = time.time()
    result = subprocess.run(cmd)
    elapsed_time = time.time() - start_time
    
    print("-"*80)
    print()
    print("="*80)
    print(f"Test execution completed in {elapsed_time:.2f} seconds")
    print("="*80)
    
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())

