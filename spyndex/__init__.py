"""spyndex - Awesome Spectral Indices in Python"""

__version__ = "0.7.0"
__author__ = "David Montero Loaiza <dml.mont@gmail.com>"
__all__ = []

from . import datasets, plot
from .axioms import bands, constants, indices
from .spyndex import *

# S3 integration (optional import to avoid dependency issues)
try:
    from . import s3
    __all__.extend(['s3'])
    
    # Expose commonly used S3 classes at package level for convenience
    from .s3 import (
        S3Config,
        S3DataLoader, 
        S3SpectralProcessor,
        create_s3_config,
        process_s3_spectral_data
    )
    __all__.extend([
        'S3Config', 'S3DataLoader', 'S3SpectralProcessor',
        'create_s3_config', 'process_s3_spectral_data'
    ])
    
except ImportError:
    # S3 dependencies not available - this is fine for standard usage
    pass
