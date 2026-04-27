"""
Utility functions for device detection and GPU compatibility checking.
"""
import warnings
import logging

# Suppress CUDA compatibility warnings before importing torch
# This prevents the warning from appearing when torch.cuda is initialized
warnings.filterwarnings(
    "ignore",
    message=".*not compatible.*",
    category=UserWarning,
    module="torch.cuda"
)
warnings.filterwarnings(
    "ignore",
    message=".*CUDA capability.*",
    category=UserWarning,
    module="torch.cuda"
)

import torch

logger = logging.getLogger(__name__)

# PyTorch 2.0+ supports CUDA compute capabilities sm_70 and above
# This list may need to be updated for newer PyTorch versions
SUPPORTED_COMPUTE_CAPABILITIES = [
    (7, 0),  # sm_70
    (7, 5),  # sm_75
    (8, 0),  # sm_80
    (8, 6),  # sm_86
    (9, 0),  # sm_90
    (10, 0), # sm_100
    (12, 0), # sm_120
]


def _is_compute_capability_supported(major: int, minor: int) -> bool:
    """
    Check if a CUDA compute capability is supported by the current PyTorch build.
    
    Args:
        major: Major version of compute capability
        minor: Minor version of compute capability
    
    Returns:
        True if supported, False otherwise
    """
    # Check if (major, minor) is in the supported list
    # Also check if it's >= minimum supported (sm_70 = 7.0)
    if (major, minor) in SUPPORTED_COMPUTE_CAPABILITIES:
        return True
    
    # Check if it's at least sm_70
    if major > 7 or (major == 7 and minor >= 0):
        # Might be a newer capability not in our list, assume supported
        return True
    
    return False


def get_compatible_device(device: str = None) -> torch.device:
    """
    Get a compatible device (CPU or CUDA).
    
    This function checks if CUDA is available and compatible with the current
    PyTorch installation. If the GPU is incompatible (e.g., compute capability
    too low), it falls back to CPU and logs a warning.
    
    Args:
        device: Optional device string ('cuda', 'cpu', or None for auto-detect)
                If 'cuda' is specified but GPU is incompatible, falls back to CPU
    
    Returns:
        torch.device: Compatible device
    """
    # If user explicitly wants CPU, return it immediately
    if device is not None and device.lower() == 'cpu':
        return torch.device("cpu")
    
    # Check if user wants CUDA or auto-detect
    wants_cuda = device is None or (device is not None and device.lower() in ['cuda', 'cuda:0'])
    
    # Auto-detect device or user wants CUDA
    if not torch.cuda.is_available():
        if wants_cuda:
            logger.warning("CUDA requested but not available. Falling back to CPU.")
        return torch.device("cpu")
    
    # CUDA is available, check if it's actually compatible
    # Suppress the incompatibility warning while checking
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*not compatible.*", category=UserWarning)
        warnings.filterwarnings("ignore", message=".*CUDA capability.*", category=UserWarning)
        
        # Get device compute capability
        # This might fail if CUDA context can't be initialized
        try:
            capability = torch.cuda.get_device_capability(0)
            major, minor = capability
            
            # Check if this compute capability is supported
            if not _is_compute_capability_supported(major, minor):
                logger.warning(
                    f"GPU with compute capability {major}.{minor} is not compatible "
                    f"with current PyTorch installation (requires sm_70+). "
                    f"Falling back to CPU."
                )
                return torch.device("cpu")
        except RuntimeError as e:
            # Error getting device capability - likely incompatible GPU
            error_msg = str(e).lower()
            if any(keyword in error_msg for keyword in ['no kernel image', 'cuda error', 'accelerator', 'kernel image', 'not compatible']):
                logger.warning(
                    f"GPU is not compatible with current PyTorch installation. "
                    f"Error during capability check: {e}. Falling back to CPU."
                )
            else:
                logger.warning(f"Could not determine GPU compute capability: {e}. Falling back to CPU.")
            return torch.device("cpu")
        except Exception as e:
            # Any other error getting capability, assume incompatible
            logger.warning(f"Unexpected error checking GPU capability: {e}. Falling back to CPU.")
            return torch.device("cpu")
        
        # If we get here, compute capability check passed
        # Still verify CUDA actually works by creating a test tensor
        try:
            test_tensor = torch.zeros(1, device='cuda')
            # Try a simple operation to ensure kernels are available
            _ = test_tensor + 1
            del test_tensor
            torch.cuda.empty_cache()
            
            # GPU is compatible and working
            return torch.device("cuda")
        except RuntimeError as e:
            # Catch CUDA errors including AcceleratorError (which is a RuntimeError subclass)
            error_msg = str(e).lower()
            if any(keyword in error_msg for keyword in ['no kernel image', 'cuda error', 'accelerator', 'kernel image']):
                logger.warning(
                    f"GPU is not compatible with current PyTorch installation. "
                    f"Error during tensor test: {e}. Falling back to CPU."
                )
            else:
                logger.warning(f"Error testing CUDA device: {e}. Falling back to CPU.")
            return torch.device("cpu")
        except Exception as e:
            # Catch any other unexpected errors
            error_msg = str(e).lower()
            if any(keyword in error_msg for keyword in ['cuda', 'gpu', 'device']):
                logger.warning(f"CUDA-related error: {e}. Falling back to CPU.")
            else:
                logger.warning(f"Unexpected error testing CUDA: {e}. Falling back to CPU.")
            return torch.device("cpu")

