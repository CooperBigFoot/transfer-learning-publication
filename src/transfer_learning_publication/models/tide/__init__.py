"""TiDE model implementation module.

TiDE (Time-series Dense Encoder) is a model architecture based on the paper:
"Long-term Forecasting with TiDE: Time-series Dense Encoder"
https://arxiv.org/pdf/2304.08424
"""

from .config import TiDEConfig
from .lightning import LitTiDE
from .model import TiDEModel, TiDEResBlock

__all__ = ["TiDEConfig", "TiDEModel", "TiDEResBlock", "LitTiDE"]
