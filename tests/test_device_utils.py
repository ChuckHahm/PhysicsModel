"""
Tests for device_utils.py - Device detection and GPU compatibility.
"""
import pytest
from device_utils import get_compatible_device


class TestDeviceUtils:
    """Test suite for device utilities."""
    
    def test_get_compatible_device_default(self, timer):
        """Test getting default device."""
        timer.start()
        device = get_compatible_device(None)
        timer.checkpoint("Device detection")
        
        assert device is not None
        # Should return either 'cpu' or 'cuda'
        assert device.type in ['cpu', 'cuda']
    
    def test_get_compatible_device_cpu(self, timer):
        """Test forcing CPU device."""
        timer.start()
        device = get_compatible_device('cpu')
        timer.checkpoint("CPU device")
        
        assert device.type == 'cpu'
    
    def test_get_compatible_device_cuda_if_available(self, timer):
        """Test CUDA device if available."""
        timer.start()
        import torch
        if torch.cuda.is_available():
            device = get_compatible_device('cuda')
            timer.checkpoint("CUDA device")
            assert device.type == 'cuda'
        else:
            # If CUDA not available, should fall back gracefully
            device = get_compatible_device('cuda')
            timer.checkpoint("CUDA fallback")
            # Should still return a valid device (might be CPU)
            assert device is not None

