"""
COCO128 Dataset Package

This package provides tools for extracting, loading, and working with the
COCO128 dataset.
"""

# Version info
__version__ = "1.0.0"

# Package-level exports
__all__ = [
    "COCO128Dataset",
    "load_dataset",
    "COCO128Extractor",
]

# Direct imports
try:
    from .loader import COCO128Dataset, load_dataset
    from .extract_coco128 import COCO128Extractor
except ImportError as e:
    if "torch" in str(e):
        raise ImportError(
            "PyTorch is required for dataset loading. "
            "Install with: pip install torch torchvision"
        ) from e
    raise
