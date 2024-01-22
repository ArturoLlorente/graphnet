"""Modules for configuring and building models.

`graphnet.models` allows for configuring and building complex GNN models using
simple, physics-oriented components. This module provides modular components
subclassing `torch.nn.Module`, meaning that users only need to import a few,
existing, purpose-built components and chain them together to form a complete
GNN
"""


from .model import Model
from .standard_model import StandardModel
from .standard_model_prediction import StandardModelPred
from .standard_model_stacking import StandardModelStacking
from .standard_model_prediction_tito import StandardModelPredTito
from .standard_average_model import StandardAverageModel