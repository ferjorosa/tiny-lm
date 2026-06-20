"""Feed-forward network variants."""

from .standard import FeedForward
from .swiglu import SwiGLU

__all__ = ["FeedForward", "SwiGLU"]
