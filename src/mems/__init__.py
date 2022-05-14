"""Top-level package for mems."""

__all__ = ("__version__", "__author__", "__email__", "archs", "muvb", "utils")
__author__ = "Rupesh Kumar Srivastava"
__email__ = "rupesh@nnaisense.com"

try:
    from .__version import __version__ as __version__
except ImportError:
    print("Please install the package to ensure correct behavior.\nFrom root folder:\n\tpip install -e .")
    __version__ = "undefined"
