# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# !pwd

# +
import os
import sys
import pandas as pd
from tqdm import tqdm
import os
from typing import Any, Dict, List, Optional
import numpy as np
import gc

KAGGLE_ENV = False
    

# +
from collections import OrderedDict
import os
from typing import Dict, List, Optional, Tuple, Union, Callable

import numpy as np
import pandas as pd


#from graphnet.data.dataset import Dataset
#from graphnet.data.sqlite import SQLiteDataset
from graphnet.data.parquet import ParquetDataset

from torch_geometric.data import Batch, Data
from torch.utils.data import DataLoader


def collate_fn(graphs: List[Data]) -> Batch:
    """Remove graphs with less than two DOM hits.

    Should not occur in "production.
    """
    graphs = [g for g in graphs if g.n_pulses > 1]
    return Batch.from_data_list(graphs)

# @TODO: Remove in favour of DataLoader{,.from_dataset_config}
def make_dataloader2(
    db: str,
    pulsemaps: Union[str, List[str]],
    features: List[str],
    truth: List[str],
    batch_ids = List[int],
    max_len = 0,
    max_pulse = 200,
    min_pulse = 200,
    *,
    batch_size: int,
    shuffle: bool,
    selection: Optional[List[int]] = None,
    num_workers: int = 10,
    persistent_workers: bool = False,
    node_truth: List[str] = None,
    truth_table: str = "truth",
    node_truth_table: Optional[str] = None,
    string_selection: List[int] = None,
    loss_weight_table: Optional[str] = None,
    loss_weight_column: Optional[str] = None,
    index_column: str = "event_no",
    labels: Optional[Dict[str, Callable]] = None,
) -> DataLoader:
    """Construct `DataLoader` instance."""
    # Check(s)
    if isinstance(pulsemaps, str):
        pulsemaps = [pulsemaps]

    dataset = ParquetDataset2(
        path=db,
        pulsemaps=pulsemaps,
        features=features,
        truth=truth,
        batch_ids=batch_ids,
        selection=selection,
        node_truth=node_truth,
        truth_table=truth_table,
        node_truth_table=node_truth_table,
        string_selection=string_selection,
        loss_weight_table=loss_weight_table,
        loss_weight_column=loss_weight_column,
        index_column=index_column,
        max_len=max_len,
        max_pulse=max_pulse,
        min_pulse=min_pulse,
    )

    # adds custom labels to dataset
    if isinstance(labels, dict):
        for label in labels.keys():
            dataset.add_label(key=label, fn=labels[label])

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        persistent_workers=persistent_workers,
        prefetch_factor=2,
    )

    return dataloader, dataset


# +
"""Standard model class(es)."""

from typing import Any, Dict, List, Optional, Union

import torch
from torch import Tensor
from torch.nn import ModuleList
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch_geometric.data import Data

from graphnet.models.coarsening import Coarsening
from graphnet.utilities.config import save_model_config
from graphnet.models.detector.detector import Detector
from graphnet.models.gnn.gnn import GNN
from graphnet.models.model import Model
from graphnet.models.task import Task

"""Standard model class(es)."""



class StandardModel2(Model):
    """Main class for standard models in graphnet.

    This class chains together the different elements of a complete GNN-based
    model (detector read-in, GNN architecture, and task-specific read-outs).
    """

    @save_model_config
    def __init__(
        self,
        *,
        detector: Detector,
        gnn: GNN,
        tasks: Union[Task, List[Task]],
        max_epochs: 0,
        coarsening: Optional[Coarsening] = None,
        optimizer_class: type = Adam,
        optimizer_kwargs: Optional[Dict] = None,
        scheduler_class: Optional[type] = None,
        scheduler_kwargs: Optional[Dict] = None,
        scheduler_config: Optional[Dict] = None,
    ) -> None:
        """Construct `StandardModel`."""
        # Base class constructor
        super().__init__()

        # Check(s)
        if isinstance(tasks, Task):
            tasks = [tasks]
        assert isinstance(tasks, (list, tuple))
        assert all(isinstance(task, Task) for task in tasks)
        assert isinstance(detector, Detector)
        assert isinstance(gnn, GNN)
        assert coarsening is None or isinstance(coarsening, Coarsening)

        # Member variable(s)
        self._detector = detector
        self._gnn = gnn
        self._tasks = ModuleList(tasks)
        self._coarsening = coarsening
        self._optimizer_class = optimizer_class
        self._optimizer_kwargs = optimizer_kwargs or dict()
        self._scheduler_class = scheduler_class
        self._scheduler_kwargs = scheduler_kwargs or dict()
        self._scheduler_config = scheduler_config or dict()
        self._dataset = dataset
        self._max_epochs = max_epochs

    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure the model's optimizer(s)."""
        optimizer = self._optimizer_class(
            self.parameters(), **self._optimizer_kwargs
        )
        config = {
            "optimizer": optimizer,
        }
        if self._scheduler_class is not None:
            scheduler = self._scheduler_class(
                optimizer, **self._scheduler_kwargs
            )
            config.update(
                {
                    "lr_scheduler": {
                        "scheduler": scheduler,
                        **self._scheduler_config,
                    },
                }
            )
        return config

    def forward(self, data: Data) -> List[Union[Tensor, Data]]:
        """Forward pass, chaining model components."""
        #import pdb;pdb.set_trace()
        if self._coarsening:
            data = self._coarsening(data)
        data = self._detector(data)
        x = self._gnn(data)
        preds = [task(x) for task in self._tasks]
        #preds = [self._tasks[0](x)]
        return preds

    def training_step(self, train_batch: Data, batch_idx: int) -> Tensor:
        """Perform training step."""
        preds = self(train_batch)
        vlosses = self._tasks[1].compute_loss(preds[1], train_batch)
        vloss = torch.sum(vlosses)
        
        tlosses = self._tasks[0].compute_loss(preds[0], train_batch)
        tloss = torch.sum(tlosses)

        #x = self.current_epoch/self._max_epochs
        #x = 0.5 + x/2
        #y = 1-x
        #loss = vloss*y + tloss*x
        loss = vloss + tloss
        return {"loss": loss, 'vloss': vloss, 'tloss': tloss}

    def validation_step(self, val_batch: Data, batch_idx: int) -> Tensor:
        """Perform validation step."""
        preds = self(val_batch)
        vlosses = self._tasks[1].compute_loss(preds[1], val_batch)
        vloss = torch.sum(vlosses)
        
        tlosses = self._tasks[0].compute_loss(preds[0], val_batch)
        tloss = torch.sum(tlosses)
        loss = vloss + tloss
        return {"loss": loss, 'vloss': vloss, 'tloss': tloss}

    def _get_batch_size(self, data: Data) -> int:
        return torch.numel(torch.unique(data.batch))

    def inference(self) -> None:
        """Activate inference mode."""
        for task in self._tasks:
            task.inference()

    def train(self, mode: bool = True) -> "Model":
        """Deactivate inference mode."""
        super().train(mode)
        if mode:
            for task in self._tasks:
                task.train_eval()
        return self

    def predict(
        self,
        dataloader: DataLoader,
        gpus: Optional[Union[List[int], int]] = None,
        distribution_strategy: Optional[str] = None,
    ) -> List[Tensor]:
        """Return predictions for `dataloader`."""
        self.inference()
        return super().predict(
            dataloader=dataloader,
            gpus=gpus,
            distribution_strategy=distribution_strategy,
        )
    
    def training_epoch_end(self, training_step_outputs):
        loss = torch.stack([x["loss"] for x in training_step_outputs]).mean()
        vloss = torch.stack([x["vloss"] for x in training_step_outputs]).mean()
        tloss = torch.stack([x["tloss"] for x in training_step_outputs]).mean()
        self.log_dict(
            {"trn_loss": loss, "trn_vloss": vloss, "trn_tloss": tloss},
            prog_bar=True,
            sync_dist=True,
        )
        print(f'epoch:{self.current_epoch}, train loss:{loss.item()}, tloss:{tloss.item()}, vloss:{vloss.item()}')
        self._dataset.reset_epoch()
        
    def validation_epoch_end(self, validation_step_outputs):
        loss = torch.stack([x["loss"] for x in validation_step_outputs]).mean()
        vloss = torch.stack([x["vloss"] for x in validation_step_outputs]).mean()
        tloss = torch.stack([x["tloss"] for x in validation_step_outputs]).mean()
        self.log_dict(
            {"val_loss": loss, "val_vloss": vloss, "val_tloss": tloss},
            prog_bar=True,
            sync_dist=True,
        )
        print(f'epoch:{self.current_epoch}, valid loss:{loss.item()}, tloss:{tloss.item()}, vloss:{vloss.item()}')



# +
from typing import Callable, Optional, Union

import torch
from torch import Tensor

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj, OptTensor, PairOptTensor, PairTensor

from torch_geometric.nn.inits import reset

try:
    from torch_cluster import knn
except ImportError:
    knn = None



class EdgeConv0(MessagePassing):
    r"""The edge convolutional operator from the `"Dynamic Graph CNN for
    Learning on Point Clouds" <https://arxiv.org/abs/1801.07829>`_ paper

    .. math::
        \mathbf{x}^{\prime}_i = \sum_{j \in \mathcal{N}(i)}
        h_{\mathbf{\Theta}}(\mathbf{x}_i \, \Vert \,
        \mathbf{x}_j - \mathbf{x}_i),

    where :math:`h_{\mathbf{\Theta}}` denotes a neural network, *.i.e.* a MLP.

    Args:
        nn (torch.nn.Module): A neural network :math:`h_{\mathbf{\Theta}}` that
            maps pair-wise concatenated node features :obj:`x` of shape
            :obj:`[-1, 2 * in_channels]` to shape :obj:`[-1, out_channels]`,
            *e.g.*, defined by :class:`torch.nn.Sequential`.
        aggr (string, optional): The aggregation scheme to use
            (:obj:`"add"`, :obj:`"mean"`, :obj:`"max"`).
            (default: :obj:`"max"`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.

    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F_{in})` or
          :math:`((|\mathcal{V}|, F_{in}), (|\mathcal{V}|, F_{in}))`
          if bipartite,
          edge indices :math:`(2, |\mathcal{E}|)`
        - **output:** node features :math:`(|\mathcal{V}|, F_{out})` or
          :math:`(|\mathcal{V}_t|, F_{out})` if bipartite
    """
    def __init__(self, nn: Callable, aggr: str = 'max', **kwargs):
        super().__init__(aggr=aggr, **kwargs)
        self.nn = nn
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.nn)

    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: PairTensor = (x, x)
        # propagate_type: (x: PairTensor)
        return self.propagate(edge_index, x=x, size=None)

    def message(self, x_i: Tensor, x_j: Tensor) -> Tensor:
        return self.nn(torch.cat([x_i, x_j - x_i, x_j], dim=-1)) ##edgeConv0

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(nn={self.nn})'
    
NODE_EDGE_FEA_RATIO = 0.7

class EdgeConv1(MessagePassing):
    r"""The edge convolutional operator from the `"Dynamic Graph CNN for
    Learning on Point Clouds" <https://arxiv.org/abs/1801.07829>`_ paper

    .. math::
        \mathbf{x}^{\prime}_i = \sum_{j \in \mathcal{N}(i)}
        h_{\mathbf{\Theta}}(\mathbf{x}_i \, \Vert \,
        \mathbf{x}_j - \mathbf{x}_i),

    where :math:`h_{\mathbf{\Theta}}` denotes a neural network, *.i.e.* a MLP.

    Args:
        nn (torch.nn.Module): A neural network :math:`h_{\mathbf{\Theta}}` that
            maps pair-wise concatenated node features :obj:`x` of shape
            :obj:`[-1, 2 * in_channels]` to shape :obj:`[-1, out_channels]`,
            *e.g.*, defined by :class:`torch.nn.Sequential`.
        aggr (string, optional): The aggregation scheme to use
            (:obj:`"add"`, :obj:`"mean"`, :obj:`"max"`).
            (default: :obj:`"max"`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.

    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F_{in})` or
          :math:`((|\mathcal{V}|, F_{in}), (|\mathcal{V}|, F_{in}))`
          if bipartite,
          edge indices :math:`(2, |\mathcal{E}|)`
        - **output:** node features :math:`(|\mathcal{V}|, F_{out})` or
          :math:`(|\mathcal{V}_t|, F_{out})` if bipartite
    """
    def __init__(self, nn: Callable, aggr: str = 'max', **kwargs):
        super().__init__(aggr=aggr, **kwargs)
        self.nn = nn
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.nn)

    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: PairTensor = (x, x)
        # propagate_type: (x: PairTensor)
        return self.propagate(edge_index, x=x, size=None)

    def message(self, x_i: Tensor, x_j: Tensor) -> Tensor:
        edge_cnt = round(x_i.shape[-1]*NODE_EDGE_FEA_RATIO)
        if edge_cnt < 20:
            edge_cnt = 4                   #TODO　通常edge素性の数は20以上、それより少ないときは最初のレイヤーなのでXYZTの4つ
        edge_ij = torch.cat([(x_j - x_i)[:,:edge_cnt],x_j[:,edge_cnt:]], axis=-1)
        return self.nn(torch.cat([x_i, edge_ij], dim=-1)) ##edgeConv1

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(nn={self.nn})'


# +
"""Class(es) implementing layers to be used in `graphnet` models."""

from typing import Any, Callable, Optional, Sequence, Union

from torch.functional import Tensor
#from torch_geometric.nn import EdgeConv
from torch_geometric.nn.pool import knn_graph
from torch_geometric.typing import Adj
from pytorch_lightning import LightningModule

from torch.nn.modules import TransformerEncoder, TransformerEncoderLayer
from torch_geometric.utils import to_dense_batch
from torch.nn.modules.normalization import LayerNorm #lNorm

USE_TRANS_IN_DYN1=True

class dynTrans1(EdgeConv0, LightningModule):
    """Dynamical edge convolution layer."""

    def __init__(
        self,
        layer_sizes,
        aggr: str = "max",
        nb_neighbors: int = 8,
        features_subset: Optional[Union[Sequence[int], slice]] = None,
        **kwargs: Any,
    ):
        """Construct `DynEdgeConv`.

        Args:
            nn: The MLP/torch.Module to be used within the `EdgeConv`.
            aggr: Aggregation method to be used with `EdgeConv`.
            nb_neighbors: Number of neighbours to be clustered after the
                `EdgeConv` operation.
            features_subset: Subset of features in `Data.x` that should be used
                when dynamically performing the new graph clustering after the
                `EdgeConv` operation. Defaults to all features.
            **kwargs: Additional features to be passed to `EdgeConv`.
        """
        # Check(s)
        if features_subset is None:
            features_subset = slice(None)  # Use all features
        assert isinstance(features_subset, (list, slice))
                
        layers = []
        for ix, (nb_in, nb_out) in enumerate(
            zip(layer_sizes[:-1], layer_sizes[1:])
        ):
            if ix == 0:
                nb_in *= 3 # edgeConv1
            layers.append(torch.nn.Linear(nb_in, nb_out))
            layers.append(torch.nn.LeakyReLU())
        d_model = nb_out
        # Base class constructor
        super().__init__(nn=torch.nn.Sequential(*layers), aggr=aggr, **kwargs)

        # Additional member variables
        self.nb_neighbors = nb_neighbors
        self.features_subset = features_subset
        

        self.norm_first=False
        
        self.norm1 = LayerNorm(d_model, eps=1e-5) #lNorm
        
        # Transformer layer(s)
        if USE_TRANS_IN_DYN1:
            encoder_layer = TransformerEncoderLayer(d_model=d_model, nhead=8, batch_first=True, dropout=DROPOUT, norm_first=self.norm_first)
            self._transformer_encoder = TransformerEncoder(encoder_layer, num_layers=1)        
        
        

    def forward(
        self, x: Tensor, edge_index: Adj, batch: Optional[Tensor] = None
    ) -> Tensor:
        """Forward pass."""
        # Standard EdgeConv forward pass
        #print('        before forward in DynEdgeConv2', x.shape)
        
        if self.norm_first:
            x = self.norm1(x) # lNorm
            
        x_out = super().forward(x, edge_index)
        
        # 最初のレイヤー以外はスキップコネクションを入れる
        if x_out.shape[-1] == x.shape[-1] and SERIAL_CONNECTION:
            #print('adding residual connection')
            x = x + x_out
        else:
            #print('skip residual connection')
            x = x_out
            
        if not self.norm_first:
            x = self.norm1(x) # lNorm

        # Recompute adjacency
        edge_index = None

        # Transformer layer
        if USE_TRANS_IN_DYN1:
            #print('        before to_dense_batch in DynEdgeConv2', x.shape)
            x, mask = to_dense_batch(x, batch)
            #print('        before Transformer in DynEdgeConv2', x.shape)
            x = self._transformer_encoder(x, src_key_padding_mask=~mask)
            #print('        before mask in DynEdgeConv2', x.shape)
            x = x[mask]

        return x, edge_index
# +
"""Implementation of the DynEdge GNN model architecture."""
from typing import List, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor, LongTensor
from torch_geometric.data import Data
from torch_scatter import scatter_max, scatter_mean, scatter_min, scatter_sum

from graphnet.models.components.layers import DynEdgeConv
from graphnet.utilities.config import save_model_config
from graphnet.models.gnn.gnn import GNN
from graphnet.models.utils import calculate_xyzt_homophily

from torch.nn.modules import TransformerEncoder, TransformerEncoderLayer
from torch_geometric.utils import to_dense_batch
from torch import nn

GLOBAL_POOLINGS = {
    "min": scatter_min,
    "max": scatter_max,
    "sum": scatter_sum,
    "mean": scatter_mean,
}

class DynEdge(GNN):
    """DynEdge (dynamical edge convolutional) model."""

    @save_model_config
    def __init__(
        self,
        nb_inputs: int,
        *,
        nb_neighbours: int = 8,
        features_subset: Optional[Union[List[int], slice]] = None,
        dynedge_layer_sizes: Optional[List[Tuple[int, ...]]] = None,
        post_processing_layer_sizes: Optional[List[int]] = None,
        readout_layer_sizes: Optional[List[int]] = None,
        global_pooling_schemes: Optional[Union[str, List[str]]] = None,
        add_global_variables_after_pooling: bool = False,
    ):
        """Construct `DynEdge`.

        Args:
            nb_inputs: Number of input features on each node.
            nb_neighbours: Number of neighbours to used in the k-nearest
                neighbour clustering which is performed after each (dynamical)
                edge convolution.
            features_subset: The subset of latent features on each node that
                are used as metric dimensions when performing the k-nearest
                neighbours clustering. Defaults to [0,1,2].
            dynedge_layer_sizes: The layer sizes, or latent feature dimenions,
                used in the `DynEdgeConv` layer. Each entry in
                `dynedge_layer_sizes` corresponds to a single `DynEdgeConv`
                layer; the integers in the corresponding tuple corresponds to
                the layer sizes in the multi-layer perceptron (MLP) that is
                applied within each `DynEdgeConv` layer. That is, a list of
                size-two tuples means that all `DynEdgeConv` layers contain a
                two-layer MLP.
                Defaults to [(128, 256), (336, 256), (336, 256), (336, 256)].
            post_processing_layer_sizes: Hidden layer sizes in the MLP
                following the skip-concatenation of the outputs of each
                `DynEdgeConv` layer. Defaults to [336, 256].
            readout_layer_sizes: Hidden layer sizes in the MLP following the
                post-processing _and_ optional global pooling. As this is the
                last layer(s) in the model, the last layer in the read-out
                yields the output of the `DynEdge` model. Defaults to [128,].
            global_pooling_schemes: The list global pooling schemes to use.
                Options are: "min", "max", "mean", and "sum".
            add_global_variables_after_pooling: Whether to add global variables
                after global pooling. The alternative is to  added (distribute)
                them to the individual nodes before any convolutional
                operations.
        """
        # Latent feature subset for computing nearest neighbours in DynEdge.
        if features_subset is None:
            features_subset = slice(0, 4) #4D

        # DynEdge layer sizes
        if dynedge_layer_sizes is None: #nb_nearest_neighboursと合わせて変更
            dynedge_layer_sizes = DYNEDGE_LAYER_SIZE

        assert isinstance(dynedge_layer_sizes, list)
        assert len(dynedge_layer_sizes)
        assert all(isinstance(sizes, tuple) for sizes in dynedge_layer_sizes)
        assert all(len(sizes) > 0 for sizes in dynedge_layer_sizes)
        assert all(
            all(size > 0 for size in sizes) for sizes in dynedge_layer_sizes
        )

        self._dynedge_layer_sizes = dynedge_layer_sizes

        # Post-processing layer sizes
        if post_processing_layer_sizes is None:
            post_processing_layer_sizes = [
                336,
                256,
            ]

        assert isinstance(post_processing_layer_sizes, list)
        assert len(post_processing_layer_sizes)
        assert all(size > 0 for size in post_processing_layer_sizes)

        self._post_processing_layer_sizes = post_processing_layer_sizes

        # Read-out layer sizes
        if readout_layer_sizes is None:
            readout_layer_sizes = [
                256,
                128,
            ]

        assert isinstance(readout_layer_sizes, list)
        assert len(readout_layer_sizes)
        assert all(size > 0 for size in readout_layer_sizes)

        self._readout_layer_sizes = readout_layer_sizes
        


        # Global pooling scheme(s)
        if isinstance(global_pooling_schemes, str):
            global_pooling_schemes = [global_pooling_schemes]

        if isinstance(global_pooling_schemes, list):
            for pooling_scheme in global_pooling_schemes:
                assert (
                    pooling_scheme in GLOBAL_POOLINGS
                ), f"Global pooling scheme {pooling_scheme} not supported."
        else:
            assert global_pooling_schemes is None

        self._global_pooling_schemes = global_pooling_schemes

        if add_global_variables_after_pooling:
            assert self._global_pooling_schemes, (
                "No global pooling schemes were request, so cannot add global"
                " variables after pooling."
            )
        self._add_global_variables_after_pooling = (
            add_global_variables_after_pooling
        )

        # Base class constructor
        super().__init__(nb_inputs, self._readout_layer_sizes[-1])

        # Remaining member variables()
        self._activation = torch.nn.LeakyReLU()
        self._nb_inputs = nb_inputs
        self._nb_global_variables = 5 + nb_inputs
        self._nb_neighbours = nb_neighbours
        self._features_subset = features_subset

        self._construct_layers()

    def _construct_layers(self) -> None:
        """Construct layers (torch.nn.Modules)."""
        # Convolutional operations
        nb_input_features = self._nb_inputs
        if USE_G:
            if not self._add_global_variables_after_pooling:
                nb_input_features += self._nb_global_variables

        self._conv_layers = torch.nn.ModuleList()
        nb_latent_features = nb_input_features
        for sizes in self._dynedge_layer_sizes:
            conv_layer = dynTrans1(
                [nb_latent_features] + list(sizes),
                aggr="max",
                nb_neighbors=self._nb_neighbours,
                features_subset=self._features_subset,
            )
            self._conv_layers.append(conv_layer)
            nb_latent_features = sizes[-1]

        # Post-processing operations
        if SERIAL_CONNECTION:
            nb_latent_features = self._dynedge_layer_sizes[-1][-1]
        else:
            nb_latent_features = (
                sum(sizes[-1] for sizes in self._dynedge_layer_sizes)
                + nb_input_features
            )

        if USE_PP:
            post_processing_layers = []
            layer_sizes = [nb_latent_features] + list(
                self._post_processing_layer_sizes
            )
            for nb_in, nb_out in zip(layer_sizes[:-1], layer_sizes[1:]):
                post_processing_layers.append(torch.nn.Linear(nb_in, nb_out))
                post_processing_layers.append(self._activation)
            last_posting_layer_output_dim = nb_out

            self._post_processing = torch.nn.Sequential(*post_processing_layers)
        else:
            last_posting_layer_output_dim = nb_latent_features

        # Read-out operations
        nb_poolings = (
            len(self._global_pooling_schemes)
            if self._global_pooling_schemes
            else 1
        )
        nb_latent_features = last_posting_layer_output_dim * nb_poolings
        if USE_G:
            if self._add_global_variables_after_pooling:  
                nb_latent_features += self._nb_global_variables

        readout_layers = []
        layer_sizes = [nb_latent_features] + list(self._readout_layer_sizes)
        for nb_in, nb_out in zip(layer_sizes[:-1], layer_sizes[1:]):
            readout_layers.append(torch.nn.Linear(nb_in, nb_out))
            readout_layers.append(self._activation)

        self._readout = torch.nn.Sequential(*readout_layers)
        

        # Transformer layer(s)
        if USE_TRANS_IN_LAST:
            encoder_layer = TransformerEncoderLayer(d_model=last_posting_layer_output_dim, nhead=8, batch_first=True, dropout=DROPOUT, norm_first=False)
            self._transformer_encoder = TransformerEncoder(encoder_layer, num_layers=USE_TRANS_IN_LAST)        


    def _global_pooling(self, x: Tensor, batch: LongTensor) -> Tensor:
        """Perform global pooling."""
        assert self._global_pooling_schemes
        pooled = []
        for pooling_scheme in self._global_pooling_schemes:
            pooling_fn = GLOBAL_POOLINGS[pooling_scheme]
            pooled_x = pooling_fn(x, index=batch, dim=0)
            if isinstance(pooled_x, tuple) and len(pooled_x) == 2:
                # `scatter_{min,max}`, which return also an argument, vs.
                # `scatter_{mean,sum}`
                pooled_x, _ = pooled_x
            pooled.append(pooled_x)

        return torch.cat(pooled, dim=1)

    def _calculate_global_variables(
        self,
        x: Tensor,
        edge_index: LongTensor,
        batch: LongTensor,
        *additional_attributes: Tensor,
    ) -> Tensor:
        """Calculate global variables."""
        # Calculate homophily (scalar variables)
        h_x, h_y, h_z, h_t = calculate_xyzt_homophily(x, edge_index, batch)

        # Calculate mean features
        global_means = scatter_mean(x, batch, dim=0)

        # Add global variables
        global_variables = torch.cat(
            [
                global_means,
                h_x,
                h_y,
                h_z,
                h_t,
            ]
            + [attr.unsqueeze(dim=1) for attr in additional_attributes],
            dim=1,
        )

        return global_variables

    def forward(self, data: Data) -> Tensor:
        """Apply learnable forward pass."""
        # Convenience variables
        x, edge_index, batch = data.x, data.edge_index, data.batch

        #assert len(edge_index) == len(self._conv_layers)
        #import pdb;pdb.set_trace()
        
        if USE_G:
            global_variables = self._calculate_global_variables(
                x,
                edge_index[0],
                batch,
                torch.log10(data.n_pulses),
            )

            #print('before _add_global_variables_after_pooling', x.shape)
            # Distribute global variables out to each node
            if not self._add_global_variables_after_pooling:
                distribute = (
                    batch.unsqueeze(dim=1) == torch.unique(batch).unsqueeze(dim=0)
                ).type(torch.float)

                global_variables_distributed = torch.sum(
                    distribute.unsqueeze(dim=2)
                    * global_variables.unsqueeze(dim=0),
                    dim=1,
                )

                x = torch.cat((x, global_variables_distributed), dim=1)


        #print('before dynEdge', x.shape)
        # DynEdge-convolutions
        if SERIAL_CONNECTION:
            for conv_layer_index, conv_layer in enumerate(self._conv_layers):
                x, _edge_index = conv_layer(x, data.edge_index[0], batch)
        else:
            skip_connections = [x]
            for conv_layer_index, conv_layer in enumerate(self._conv_layers):
                x, _edge_index = conv_layer(x, data.edge_index[0], batch)
                #print('    dynEdge output skip_connections', x.shape)
                skip_connections.append(x)

            # Skip-cat
            x = torch.cat(skip_connections, dim=1)

        # Transformer layer
        if USE_TRANS_IN_LAST:
            #print('before to_dense_batch', x.shape)
            x, mask = to_dense_batch(x, batch)
            #print('before Transformer', x.shape)
            x = self._transformer_encoder(x, src_key_padding_mask=~mask)
            #print('before mask', x.shape)
            x = x[mask]
        
        # Post-processing
        if USE_PP:
            #print('before _post_processing', x.shape)
            x = self._post_processing(x)
        
        # (Optional) Global pooling
        #import pdb;pdb.set_trace()
        #print('before Global pooling', x.shape)
        if self._global_pooling_schemes:
            x = self._global_pooling(x, batch=batch)
            if USE_G:
                if self._add_global_variables_after_pooling:
                    x = torch.cat(
                        [
                            x,
                            global_variables,
                        ],
                        dim=1,
                    )

        #print('before Read-out', x.shape)
        # Read-out
        x = self._readout(x)
        #print('final', x.shape)

        return x
# +
"""Class(es) for building/connecting graphs."""

from typing import List
import random

import torch
from torch_geometric.nn import knn_graph, radius_graph
from torch_geometric.data import Data

from graphnet.utilities.config import save_model_config
from graphnet.models.utils import calculate_distance_matrix
from graphnet.models import Model


class GraphBuilder(Model):  # pylint: disable=too-few-public-methods
    """Base class for graph building."""

    pass

TIME_PARAM_FOR_DIST = 1/10

class KNNGraphBuilderMulti(GraphBuilder):  # pylint: disable=too-few-public-methods
    """Builds graph from the k-nearest neighbours."""

    @save_model_config
    def __init__(
        self,
        nb_nearest_neighbours,
        columns,
    ):
        """Construct `KNNGraphBuilder`."""
        # Base class constructor
        super().__init__()

        # Member variable(s)
        assert len(nb_nearest_neighbours) == len(columns)
        self._nb_nearest_neighbours = nb_nearest_neighbours
        self._columns = columns

    def forward(self, data: Data) -> Data:
        """Forward pass."""
        # Constructs the adjacency matrix from the raw, DOM-level data and
        # returns this matrix
        
        if data.edge_index is not None:
            self.info(
                "WARNING: GraphBuilder received graph with pre-existing "
                "structure. Will overwrite."
            )
        edge_index_list = []
        x = data.x
        x[:,3] = x[:,3]*TIME_PARAM_FOR_DIST
        for idx in range(len(self._nb_nearest_neighbours)):
            nb_nearest_neighbour = self._nb_nearest_neighbours[idx]
            if type(nb_nearest_neighbour) == str:
                nb_nearest_neighbour_min, nb_nearest_neighbour_max = nb_nearest_neighbour.split('-')
                nb_nearest_neighbour = torch.randint(int(nb_nearest_neighbour_min), int(nb_nearest_neighbour_max), (1,))[0]
                #print('nb_nearest_neighbour', nb_nearest_neighbour)
            elif type(nb_nearest_neighbour) == list:
                nb_nearest_neighbour = random.choice(nb_nearest_neighbour)
                #print('nb_nearest_neighbour', nb_nearest_neighbour)
            edge_index = knn_graph(
                x[:, self._columns[idx]]/1000,
                nb_nearest_neighbour,
                data.batch,
            ).to(self.device)
            edge_index_list.append(edge_index)
        x[:,3] = x[:,3]/TIME_PARAM_FOR_DIST # 不要？

        data.edge_index = edge_index_list
        return data


# +
from pytorch_lightning.callbacks import EarlyStopping
from torch.optim.adam import Adam
from torch import Tensor
from graphnet.data.constants import FEATURES, TRUTH
from graphnet.models import StandardModel
from graphnet.models.detector.icecube import IceCubeKaggle
#from graphnet.models.gnn import DynEdge
from graphnet.models.graph_builders import KNNGraphBuilder
from graphnet.models.task.reconstruction import DirectionReconstructionWithKappa, ZenithReconstructionWithKappa, AzimuthReconstructionWithKappa
from graphnet.training.loss_functions import VonMisesFisher3DLoss, VonMisesFisher2DLoss, LossFunction, VonMisesFisherLoss
from graphnet.training.callbacks import ProgressBar, PiecewiseLinearLR
from graphnet.training.labels import Direction
from graphnet.training.utils import make_dataloader
from pytorch_lightning import Trainer
import pandas as pd
from typing import Any, Optional, Union, List, Dict
from graphnet.models.task import Task

class DistanceLoss2(LossFunction):

    def _forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        target = target.reshape(-1, 3)
        # Check(s)
        assert prediction.dim() == 2 and prediction.size()[1] == 4
        assert target.dim() == 2
        assert prediction.size()[0] == target.size()[0]
        
        eps = 1e-4
        prediction_length = torch.linalg.vector_norm(prediction[:, [0, 1, 2]], dim=1)
        #print('norm pre in dis',prediction_length[0])
        #print('prediction[0] in dis', prediction[0])
        prediction_length = torch.clamp(prediction_length, min=eps)
        prediction =  prediction[:, [0, 1, 2]]/prediction_length.unsqueeze(1)
        cosLoss = prediction[:, 0] * target[:, 0] + prediction[:, 1] * target[:, 1] + prediction[:, 2] * target[:, 2]    
        cosLoss = torch.clamp(cosLoss, min=-1+eps, max=1-eps)
        thetaLoss = torch.arccos(cosLoss)
        #thetaLoss = torch.clamp(thetaLoss, min=eps, max=np.pi-eps)   
        #print(thetaLoss)
        return thetaLoss
    
    

class IceCubeKaggle2(Detector):
    """`Detector` class for Kaggle Competition."""

    # Implementing abstract class attribute
    features = FEATURES.KAGGLE

    def _forward(self, data: Data) -> Data:
        """Ingest data, build graph, and preprocess features.

        Args:
            data: Input graph data.

        Returns:
            Connected and preprocessed graph data.
        """
        # Check(s)
        self._validate_features(data)

        # Preprocessing
        data.x[:, 0] /= 500.0  # x
        data.x[:, 1] /= 500.0  # y
        data.x[:, 2] /= 500.0  # z
        data.x[:, 3] = (data.x[:, 3] - 1.0e04) / (500.0*0.23)  # 0.23 is speed of light in ice
        data.x[:, 4] = torch.log10(data.x[:, 4]) / 3.0  # charge

        return data
    
class DirectionReconstructionWithKappa2(Task):
    """Reconstructs direction with kappa from the 3D-vMF distribution."""

    # Requires three features: untransformed points in (x,y,z)-space.
    default_target_labels = [
        "direction"
    ]  # contains dir_x, dir_y, dir_z see https://github.com/graphnet-team/graphnet/blob/95309556cfd46a4046bc4bd7609888aab649e295/src/graphnet/training/labels.py#L29
    default_prediction_labels = [
        "dir_x_pred",
        "dir_y_pred",
        "dir_z_pred",
        "direction_kappa",
    ]    
    nb_inputs = 3

    def _forward(self, x: Tensor) -> Tensor:
        # Transform outputs to angle and prepare prediction
        kappa = torch.linalg.vector_norm(x, dim=1)# + eps_like(x)
        kappa = torch.clamp(kappa, min=torch.finfo(x.dtype).eps)
        vec_x = x[:, 0] / kappa
        vec_y = x[:, 1] / kappa
        vec_z = x[:, 2] / kappa
        return torch.stack((vec_x, vec_y, vec_z, kappa), dim=1)

def build_model2(config: Dict[str,Any], train_dataloader: Any, train_dataset: Any) -> StandardModel2:
    """Builds GNN from config"""
    # Building model
    detector = IceCubeKaggle2(
        graph_builder=KNNGraphBuilderMulti(nb_nearest_neighbours=NB_NEAREST_NEIGHBOURS, columns=COLUMNS_NEAREST_NEIGHBOURS) #dynedge_layer_sizes と合わせて変更
    )
    gnn = DynEdge(
        nb_inputs=detector.nb_outputs,
        #global_pooling_schemes=["min", "max", "mean"],
        global_pooling_schemes=["max"],
        add_global_variables_after_pooling=True
    )

    if config["target"] == 'direction':
        task = DirectionReconstructionWithKappa2(
            hidden_size=gnn.nb_outputs,
            target_labels=config["target"],
            loss_function=VonMisesFisher3DLoss(),
        )
        task2 = DirectionReconstructionWithKappa2(
            hidden_size=gnn.nb_outputs,
            target_labels=config["target"],
            loss_function=DistanceLoss2(),
        )
        
        prediction_columns = [config["target"] + "_x", 
                              config["target"] + "_y", 
                              config["target"] + "_z", 
                              config["target"] + "_kappa" ]
        additional_attributes = ['zenith', 'azimuth', 'event_id']

    model = StandardModel2(
        detector=detector,
        gnn=gnn,
        tasks=[task2, task],
        dataset=train_dataset,
        max_epochs=config["fit"]["max_epochs"],
        optimizer_class=Adam,
        optimizer_kwargs={"lr": 1e-03, "eps": 1e-03},
        #optimizer_class=Lion,
        #optimizer_kwargs={"lr": 1e-04},
        scheduler_class=PiecewiseLinearLR,
        scheduler_kwargs={
            "milestones": [
                0,
                len(train_dataloader)*config["fit"]["max_epochs"]//(len(config['gpus'])*config['accumulate_grad_batches'][0]*2),
                len(train_dataloader)*config["fit"]["max_epochs"]//(len(config['gpus'])*config['accumulate_grad_batches'][0]),                
            ],
            "factors": [1, 1, 1e-03],
            "verbose": config["scheduler_verbose"],
        },
        scheduler_config={
            "interval": "step",
        },
    )
    model.prediction_columns = prediction_columns
    model.additional_attributes = additional_attributes
    
    return model

def load_pretrained_model(config: Dict[str,Any], state_dict_path: str = 'state_dict.pth') -> StandardModel2:
    train_dataloader, _, train_dataset, _ = make_dataloaders2(config = config)
    model = build_model2(config = config, 
                        train_dataloader = train_dataloader,
                        train_dataset = train_dataset,
                        )
    #model._inference_trainer = Trainer(config['fit'])
    model.load_state_dict(state_dict_path)
    model.prediction_columns = [config["target"] + "_x", 
                              config["target"] + "_y", 
                              config["target"] + "_z", 
                              config["target"] + "_kappa" ]
    model.additional_attributes = ['zenith', 'azimuth', 'event_id']
    return model

def make_dataloaders2(config: Dict[str, Any]) -> List[Any]:
    """Constructs training and validation dataloaders for training with early stopping."""
    train_dataloader, train_dataset = make_dataloader2(db = "dummy",
                                            selection = None,
                                            pulsemaps = config['pulsemap'],
                                            features = features,
                                            truth = truth,
                                            batch_ids = config['train_batch_ids'],
                                            batch_size = config['batch_size'],
                                            num_workers = config['num_workers'],
                                            shuffle = True,
                                            labels = {'direction': Direction()},
                                            index_column = config['index_column'],
                                            truth_table = config['truth_table'],
                                            max_len = config['train_len'],
                                            max_pulse = config['train_max_pulse'],
                                            min_pulse = config['train_min_pulse'],
                                            )
    
    validate_dataloader, validate_dataset = make_dataloader2(db = "dummy",
                                            selection = None,
                                            pulsemaps = config['pulsemap'],
                                            features = features,
                                            truth = truth,
                                            batch_ids = config['valid_batch_ids'],
                                            batch_size = config['batch_size'],
                                            num_workers = config['num_workers'],
                                            shuffle = False,
                                            labels = {'direction': Direction()},
                                            index_column = config['index_column'],
                                            truth_table = config['truth_table'],
                                            max_len = config['valid_len'],
                                            max_pulse = config['valid_max_pulse'],
                                            min_pulse = config['valid_min_pulse'],
                                          
                                            )
    return train_dataloader, validate_dataloader,  train_dataset, validate_dataset


def inference(model, config: Dict[str, Any]) -> pd.DataFrame:
    """Applies model to the database specified in config['inference_database_path'] and saves results to disk."""
    # Make Dataloader
    test_dataloader = make_dataloader(db = config['inference_database_path'],
                                            selection = None, # Entire database
                                            pulsemaps = config['pulsemap'],
                                            features = features,
                                            truth = truth,
                                            batch_size = config['batch_size'],
                                            num_workers = config['num_workers'],
                                            shuffle = False,
                                            labels = {'direction': Direction()},
                                            index_column = config['index_column'],
                                            truth_table = config['truth_table'],
                                            )
    
    # Get predictions
    results = model.predict_as_dataframe(
        gpus = config['gpus'],
        dataloader = test_dataloader,
        prediction_columns=model.prediction_columns,
        additional_attributes=model.additional_attributes,
    )
    return results


# -


# +

BATCH_DIR = '/remote/ceph/user/l/llorente/kaggle/train'
META_DIR = '/remote/ceph/user/l/llorente/kaggle/meta_splitted'
SAMPLE_TYPE = 'a'
FILTER_BY_KAPPA_THRE = 0.5


# +
import os
import socket

hostName = socket.gethostname()
hostName

DROPOUT=0.0
NB_NEAREST_NEIGHBOURS = [6]
COLUMNS_NEAREST_NEIGHBOURS = [slice(0,4)]
USE_G = True
ONLY_AUX_FALSE = False
SERIAL_CONNECTION = True
USE_PP = True

USE_TRANS_IN_LAST=0
DYNEDGE_LAYER_SIZE = [
                (
                    256,
                    256,
                ),
                (
                    256,
                    256,
                ),
                (
                    256,
                    256,
                ),
            ]

# Constants
features = FEATURES.KAGGLE
truth = TRUTH.KAGGLE
from pytorch_lightning import loggers as pl_loggers


#import ipynb_path
#os.environ['WANDB_NOTEBOOK_NAME'] = ipynb_path.get()
#runName = os.environ['WANDB_NOTEBOOK_NAME'].split('/')[-1].replace('.ipynb','').replace('.py','')# + '-' + hostName
runName="dummy"

    
# Configuration
FORCE_MAX_PULSE = 2000

project = "ens5"
TRAIN_BATCHS = list(range(1,12))

n_batch = 1 #len(TRAIN_BATCHS)               # TODO_ENS
n_round = 4

batch_size = 1000                # multi GPU
gpus = [3]                     # multi GPU
accumulate_grad_batches = {0: 2}

if len(gpus) > 1:
        distribution_strategy = 'ddp'
else:
    distribution_strategy = None



config = {
        "path": 'dummy',#dummy
        "inference_database_path": '',# dummy
        "pulsemap": 'pulse_table', #dummy
        "truth_table": 'meta_table', #dummy
        "features": features,
        "truth": truth,
        "index_column": 'event_id',
        "run_name_tag": 'my_example',
        "batch_size": batch_size,
        "num_workers": 2, #todo
        "target": 'direction',
        "early_stopping_patience": n_batch,
        "gpus": gpus,
        "fit": {
                "max_epochs": n_batch*n_round,
                "gpus": gpus,
                "distribution_strategy": distribution_strategy,
#                "check_val_every_n_epoch":10,
                "precision": 16,
                #"gradient_clip_val": 0.9,
#                "reload_dataloaders_every_n_epochs": 1,
                },

        "accumulate_grad_batches": accumulate_grad_batches,
        'runName': runName,
        'project': project,
        'scheduler_verbose': False,
        'train_batch_ids': list(range(1,n_batch+1)),
        'valid_batch_ids': [660],
        'test_selection': None,
        'base_dir': 'training',
        'train_len': 0, 
        'valid_len': 0, 
        'train_max_pulse': 250,
        'valid_max_pulse': 200,
        'train_min_pulse': 0,
        'valid_min_pulse': 0,
}


debug = False # bbb
if debug:
    runName = runName + '_debug'
    config["project"] = 'debug'
    config["num_workers"] = 0
    config["batch_size"] = 2
    config["train_len"] = 2
    config["valid_len"] = 2

# +
def angular_dist_score(az_true, zen_true, az_pred, zen_pred, return_arr=False):
    '''
    calculate the MAE of the angular distance between two directions.
    The two vectors are first converted to cartesian unit vectors,
    and then their scalar product is computed, which is equal to
    the cosine of the angle between the two vectors. The inverse
    cosine (arccos) thereof is then the angle between the two input vectors

    Parameters:
    -----------

    az_true : float (or array thereof)
        true azimuth value(s) in radian
    zen_true : float (or array thereof)
        true zenith value(s) in radian
    az_pred : float (or array thereof)
        predicted azimuth value(s) in radian
    zen_pred : float (or array thereof)
        predicted zenith value(s) in radian

    Returns:
    --------

    dist : float
        mean over the angular distance(s) in radian
    '''

    if not (np.all(np.isfinite(az_true)) and
            np.all(np.isfinite(zen_true)) and
            np.all(np.isfinite(az_pred)) and
            np.all(np.isfinite(zen_pred))):
        raise ValueError("All arguments must be finite")

    # pre-compute all sine and cosine values
    sa1 = np.sin(az_true)
    ca1 = np.cos(az_true)
    sz1 = np.sin(zen_true)
    cz1 = np.cos(zen_true)

    sa2 = np.sin(az_pred)
    ca2 = np.cos(az_pred)
    sz2 = np.sin(zen_pred)
    cz2 = np.cos(zen_pred)

    # scalar product of the two cartesian vectors (x = sz*ca, y = sz*sa, z = cz)
    scalar_prod = sz1*sz2*(ca1*ca2 + sa1*sa2) + (cz1*cz2)

    # scalar product of two unit vectors is always between -1 and 1, this is against nummerical instability
    # that might otherwise occure from the finite precision of the sine and cosine functions
    scalar_prod =  np.clip(scalar_prod, -1, 1)

    # convert back to an angle (in radian)
    if return_arr:
        return np.abs(np.arccos(scalar_prod))
    else:
        return np.average(np.abs(np.arccos(scalar_prod)))

def zeaz2xyz(az, ze):
    z = np.cos(ze)
    rz = np.sin(ze)
    x = rz*np.cos(az)
    y = rz*np.sin(az)
    return x, y, z

def xyz2zeaz(x,y,z):
    r = np.sqrt(x**2+y**2+z**2)
    #print('R', r)
    x = x/r
    y = y/r
    z = z/r
    ze = np.arccos(z)
    rz = np.sin(ze)
    az = np.arccos(x/rz)
    az = np.where(y < 0, np.pi*2-az, az)
    az = np.nan_to_num(az,0)
    return az, ze

def ens(az1, ze1, az2, ze2, w1, w2, trust='left', ens_thre=np.pi*0.5):
    dist = angular_dist_score(az1, ze1, az2, ze2, True)
    x1, y1, z1 = zeaz2xyz(az1, ze1)
    x2, y2, z2 = zeaz2xyz(az2, ze2)
    wx, wy, wz = (x1*w1+x2*w2)/(w1+w2),(y1*w1+y2*w2)/(w1+w2),(z1*w1+z2*w2)/(w1+w2)
    if trust=='left':
        trusted_x,trusted_y,trusted_z = x1,y1,z1
    else:
        trusted_x,trusted_y,trusted_z = x2,y2,z2
    ensx = np.where(dist < ens_thre, wx, trusted_x)
    ensy = np.where(dist < ens_thre, wy, trusted_y)
    ensz = np.where(dist < ens_thre, wz, trusted_z)

    return xyz2zeaz(ensx, ensy, ensz)
# -



# +
from torch.utils.data import Dataset, DataLoader

PRED_DIR = '/remote/ceph/user/l/llorente/prediction_models/tito_predictions'
META_DIR = '/remote/ceph/user/l/llorente/kaggle/meta_splitted'

class MyDatasetFilePreLoad(Dataset):
    def __init__(self, 
                 runNames, 
                 batch_ids,
                 tgt_cols = ['direction_x','direction_y','direction_z','direction_kappa', 'direction_x1','direction_y1','direction_z1','direction_kappa1'],
                 use_mid_fea = True,
                ):
        self.batch_ids = batch_ids
        self.runNames = runNames
        self.tgt_cols = tgt_cols
        self.use_mid_fea = use_mid_fea
        
        
        X_list = []
        Y_list = []
        event_ids_list = []
        idx1_list = []
        idx2_list = []
        
        for idx, this_batch_id in enumerate(batch_ids):
            meta = pd.read_parquet(f'{META_DIR}/meta_{this_batch_id}.parquet').reset_index(drop=True)
            meta['pulse_count'] = np.log1p(meta.last_pulse_index - meta.first_pulse_index + 1)
            
            result_list = []
            for runName in runNames:
                if 'pred_by_least_squares_baseline1183' == runName:
                    df = pd.read_csv(f'{PRED_DIR}/{runName}_{this_batch_id}.csv')
                    columns = ['azimuth','zenith']
                else:
                    print(f'reading {PRED_DIR}/{runName}_{this_batch_id}.pkl')
                    df = pd.read_pickle(f'{PRED_DIR}/{runName}_{this_batch_id}.pkl')
                    if 'direction_kappa' in df:
                        df['direction_kappa'] = np.log1p(df['direction_kappa'])
                    if 'direction_kappa1' in df:
                        df['direction_kappa1'] = np.log1p(df['direction_kappa1'])
                    columns = self.tgt_cols
                    if self.use_mid_fea:
                        columns = columns + [f'idx{i}' for i in range(128)]
                result_list.append(df[columns].reset_index(drop=True))
                
            X_list.append(pd.concat(result_list, axis=1).values)#.astype('float16'))
            Y_list.append(np.stack(zeaz2xyz(meta['azimuth'], meta['zenith'])).T)#.astype('float32'))
            event_ids_list.append(meta['event_id'].values)
            idx1_list.append(np.full(len(df),idx))
            idx2_list.append(np.arange(len(df)))
        self.idx1 = np.concatenate(idx1_list)
        del idx1_list
        gc.collect()
        self.idx2 = np.concatenate(idx2_list)
        del idx2_list
        gc.collect()
        self.event_ids = event_ids_list
        self.Y = Y_list
        self.X = X_list
        
    def reset_epoch(self) -> None:
        pass
    
    def __len__(self):
        total_len = 0
        for x in self.X:
            total_len += x.shape[0]
        return total_len
    
    def n_columns(self):
        return self.X[0].shape[1]

    def __getitem__(self, index):
        idx1 = self.idx1[index]
        idx2 = self.idx2[index]
        x = self.X[idx1][idx2]        
        y = self.Y[idx1][idx2]
        event_id = self.event_ids[idx1][idx2]      
        return x, y, event_id


TGT_COLS = ['direction_x','direction_y','direction_z','direction_kappa', 'direction_x1','direction_y1','direction_z1','direction_kappa1']  

runNames = [
    'model1',
    'model2',
    'model3',
    'model4',
    'model5',
    'model6',
]

#dataset_train = MyDatasetFilePreLoad(runNames, TRAIN_BATCHS)
dataset_train = MyDatasetFilePreLoad(runNames, [1])
dataset_valid = MyDatasetFilePreLoad(runNames, [55])

train_dataloader = DataLoader(dataset_train, batch_size=config['batch_size'], shuffle=True, num_workers=max(2,os.cpu_count()//2))
valid_dataloader = DataLoader(dataset_valid, batch_size=config['batch_size'], shuffle=False, num_workers=max(2,os.cpu_count()//2))


# +
"""Standard model class(es)."""

from typing import Any, Dict, List, Optional, Union

import torch
from torch import Tensor
from torch.nn import ModuleList
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch_geometric.data import Data

from graphnet.models.coarsening import Coarsening
from graphnet.utilities.config import save_model_config
from graphnet.models.detector.detector import Detector
from graphnet.models.gnn.gnn import GNN
from graphnet.models.model import Model
from graphnet.models.task import Task

"""Standard model class(es)."""



class StandardModel3(Model):
    """Main class for standard models in graphnet.

    This class chains together the different elements of a complete GNN-based
    model (detector read-in, GNN architecture, and task-specific read-outs).
    """

    @save_model_config
    def __init__(
        self,
        *,
        tasks: Union[Task, List[Task]],
        n_input_fea, 
        dataset,
        optimizer_class: type = Adam,
        optimizer_kwargs: Optional[Dict] = None,
        scheduler_class: Optional[type] = None,
        scheduler_kwargs: Optional[Dict] = None,
        scheduler_config: Optional[Dict] = None,
    ) -> None:
        """Construct `StandardModel`."""
        # Base class constructor
        super().__init__()

        # Check(s)
        if isinstance(tasks, Task):
            tasks = [tasks]
        assert isinstance(tasks, (list, tuple))
        assert all(isinstance(task, Task) for task in tasks)

        # Member variable(s)
        self._tasks = ModuleList(tasks)
        self._optimizer_class = optimizer_class
        self._optimizer_kwargs = optimizer_kwargs or dict()
        self._scheduler_class = scheduler_class
        self._scheduler_kwargs = scheduler_kwargs or dict()
        self._scheduler_config = scheduler_config or dict()
        self._n_input_fea = n_input_fea
        self._dataset = dataset
        
        mlp_layers = []
        layer_sizes = [n_input_fea, HIDDEN_SIZE, HIDDEN_SIZE, HIDDEN_SIZE] # todo1
        for nb_in, nb_out in zip(layer_sizes[:-1], layer_sizes[1:]):
            mlp_layers.append(torch.nn.Linear(nb_in, nb_out))
            mlp_layers.append(torch.nn.LeakyReLU())
            mlp_layers.append(torch.nn.Dropout(DROPOUT_PH2_MODEL))
        last_posting_layer_output_dim = nb_out

        self._mlp = torch.nn.Sequential(*mlp_layers)

            

    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure the model's optimizer(s)."""
        optimizer = self._optimizer_class(
            self.parameters(), **self._optimizer_kwargs
        )
        config = {
            "optimizer": optimizer,
        }
        if self._scheduler_class is not None:
            scheduler = self._scheduler_class(
                optimizer, **self._scheduler_kwargs
            )
            config.update(
                {
                    "lr_scheduler": {
                        "scheduler": scheduler,
                        **self._scheduler_config,
                    },
                }
            )
        return config

    def forward(self, x):
        x = x.float()
        x = self._mlp(x)
        x = [task(x) for task in self._tasks]
        return x

    def training_step(self, xye, idx) -> Tensor:
        """Perform training step."""
        x,y,event_ids = xye
        preds = self(x)
        batch = Data(x=x, direction=y)
        vlosses = self._tasks[1].compute_loss(preds[1], batch)
        vloss = torch.sum(vlosses)
        
        tlosses = self._tasks[0].compute_loss(preds[0], batch)
        tloss = torch.sum(tlosses)
        
        if self.current_epoch == 0:
            vloss_weight = 1
        else:
            current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
            vloss_weight = current_lr / 1e-03

        loss = vloss*vloss_weight + tloss
        #self.log_dict(
        #    {'trn_vloss_weight_step': vloss_weight},
        #    prog_bar=True,
        #    sync_dist=True,
        #    on_step=True,
        #    logger=True,
        #)
        return {"loss": loss, 'vloss': vloss, 'tloss': tloss, 'vloss_weight': vloss_weight}

    def validation_step(self, xye, idx) -> Tensor:
        """Perform validation step."""
        x,y,event_ids = xye
        preds = self(x)
        batch = Data(x=x, direction=y)
        vlosses = self._tasks[1].compute_loss(preds[1], batch)
        vloss = torch.sum(vlosses)
        
        tlosses = self._tasks[0].compute_loss(preds[0], batch)
        tloss = torch.sum(tlosses)
        
        current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        vloss_weight = current_lr / 1e-03

        loss = vloss*vloss_weight + tloss
        return {"loss": loss, 'vloss': vloss, 'tloss': tloss, 'vloss_weight': vloss_weight}

    def inference(self) -> None:
        """Activate inference mode."""
        for task in self._tasks:
            task.inference()

    def train(self, mode: bool = True) -> "Model":
        """Deactivate inference mode."""
        super().train(mode)
        if mode:
            for task in self._tasks:
                task.train_eval()
        return self

    def predict(
        self,
        dataloader: DataLoader,
        gpus: Optional[Union[List[int], int]] = None,
        distribution_strategy: Optional[str] = None,
    ) -> List[Tensor]:
        """Return predictions for `dataloader`."""
        self.inference()
        return super().predict(
            dataloader=dataloader,
            gpus=gpus,
            distribution_strategy=distribution_strategy,
        )
    
    def training_epoch_end(self, training_step_outputs):
        loss = torch.stack([x["loss"] for x in training_step_outputs]).mean()
        vloss = torch.stack([x["vloss"] for x in training_step_outputs]).mean()
        tloss = torch.stack([x["tloss"] for x in training_step_outputs]).mean()
        self.log_dict(
            {"trn_loss": loss, "trn_vloss": vloss, "trn_tloss": tloss},
            prog_bar=True,
            sync_dist=True,
        )
        print(f'epoch:{self.current_epoch}, train loss:{loss.item()}, tloss:{tloss.item()}, vloss:{vloss.item()}')
        self._dataset.reset_epoch()
        
    def validation_epoch_end(self, validation_step_outputs):
        loss = torch.stack([x["loss"] for x in validation_step_outputs]).mean()
        vloss = torch.stack([x["vloss"] for x in validation_step_outputs]).mean()
        tloss = torch.stack([x["tloss"] for x in validation_step_outputs]).mean()
        self.log_dict(
            {"val_loss": loss, "val_vloss": vloss, "val_tloss": tloss},
            prog_bar=True,
            sync_dist=True,
        )
        print(f'epoch:{self.current_epoch}, valid loss:{loss.item()}, tloss:{tloss.item()}, vloss:{vloss.item()}')

# -

HIDDEN_SIZE = 512
DROPOUT_PH2_MODEL = 0.0
N_INPUT_FEA = dataset_train.n_columns()
def build_model3(config, dataloader, dataset) -> StandardModel2:
    """Builds GNN from config"""
    # Building model

    if config["target"] == 'direction':
        task = DirectionReconstructionWithKappa2(
            hidden_size=HIDDEN_SIZE,
            target_labels=config["target"],
            loss_function=VonMisesFisher3DLoss(),
        )
        task2 = DirectionReconstructionWithKappa2(
            hidden_size=HIDDEN_SIZE,
            target_labels=config["target"],
            loss_function=DistanceLoss2(),
        )
        
        prediction_columns = [config["target"] + "_x", 
                              config["target"] + "_y", 
                              config["target"] + "_z", 
                              config["target"] + "_kappa" ]
        additional_attributes = ['zenith', 'azimuth', 'event_id']

    model = StandardModel3(
        tasks=[task2, task],
        n_input_fea=N_INPUT_FEA,
        dataset=dataset,
        optimizer_class=Adam,
        optimizer_kwargs={"lr": 1e-03, "eps": 1e-03},
        #optimizer_class=Lion,
        #optimizer_kwargs={"lr": 1e-04},
        scheduler_class=PiecewiseLinearLR,
        scheduler_kwargs={
            "milestones": [
                0,
                len(dataloader)*(config["fit"]["max_epochs"])//(len(config['gpus'])*config['accumulate_grad_batches'][0]*100),
                len(dataloader)*(config["fit"]["max_epochs"])//(len(config['gpus'])*config['accumulate_grad_batches'][0]*2),
                len(dataloader)*(config["fit"]["max_epochs"])//(len(config['gpus'])*config['accumulate_grad_batches'][0]), 
                #len(dataloader)*(config["fit"]["max_epochs"])//(1*config['accumulate_grad_batches'][0]*100),
                #len(dataloader)*(config["fit"]["max_epochs"])//(1*config['accumulate_grad_batches'][0]*2),
                #len(dataloader)*(config["fit"]["max_epochs"])//(1*config['accumulate_grad_batches'][0]),                
            ],
            "factors": [1e-03, 1, 1, 1e-04],
            "verbose": config["scheduler_verbose"],
        },
        scheduler_config={
            "interval": "step",
        },
    )
    model.prediction_columns = prediction_columns
    model.additional_attributes = additional_attributes
    
    return model
model = build_model3(config = config, dataloader = train_dataloader, dataset = dataset_train)

TRAIN = True
if TRAIN:
    from pytorch_lightning.callbacks import LearningRateMonitor
    from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
    from pytorch_lightning.callbacks import GradientAccumulationScheduler

    # Training model
    callbacks = [
        ModelCheckpoint(
            dirpath='/remote/ceph/user/l/llorente/prediction_models/stacking_tito/model_checkpoint_graphnet/',
            filename=runName+'-{epoch:02d}-{trn_tloss:.6f}',
            #every_n_epochs = 10,
            save_weights_only=False,
        ),
        ProgressBar(),
    ]

    if 'accumulate_grad_batches' in config and len(config['accumulate_grad_batches']) > 0:
        callbacks.append(GradientAccumulationScheduler(scheduling=config['accumulate_grad_batches']))

    #if debug == False:
    #    config["fit"]["logger"] = pl_loggers.WandbLogger(project=config["project"], name=runName)
    #    callbacks.append(LearningRateMonitor(logging_interval='epoch'))

    model.fit(
        train_dataloader,
        valid_dataloader,
        callbacks=callbacks,
        **config["fit"],
    )
    #model.save_state_dict(f'/remote/ceph/user/l/llorente/prediction_models/stacking_tito/{runName}-last.pth')
#CKPT = f'/remote/ceph/user/l/llorente/tito_solution/model_graphnet/stacking-6models-last.pth'
#
#state_dict =  torch.load(CKPT, torch.device('cpu'))
#if 'state_dict' in state_dict.keys():
#    state_dict = state_dict['state_dict']
#model.load_state_dict(state_dict)
#USE_ALL_FEA_IN_PRED=False
#validateMode=True
#
#event_ids = []
#zenith = []
#azimuth = []
#preds = []
#print('start predict')
#
#real_x = []
#real_y = []
#real_z = []
#with torch.no_grad():
#    model.eval()
#    model.to(f'cuda:{0}')
#    for batch in tqdm(valid_dataloader):
#        
#        pred = model(batch[0].to(f'cuda:{0}'))
#        #preds.append(pred[0])
#        if USE_ALL_FEA_IN_PRED:
#            preds.append(torch.cat(pred, axis=-1))
#        else:
#            preds.append(pred[0])
#        event_ids.append(batch[2])
#        if validateMode:
#            real_x.append(batch[1][:,0])
#            real_y.append(batch[1][:,1])
#            real_z.append(batch[1][:,2])
#preds = torch.cat(preds).to('cpu').detach().numpy()
##zenith = torch.cat(zenith).to('cpu').numpy()
##azimuth = torch.cat(azimuth).to('cpu').numpy()
#real_x = torch.cat(real_x).to('cpu').numpy()
#real_y = torch.cat(real_y).to('cpu').numpy()
#real_z = torch.cat(real_z).to('cpu').numpy()
##results = pd.DataFrame(preds, columns=model.prediction_columns)
#if USE_ALL_FEA_IN_PRED:
#    if preds.shape[1] == 128+8:
#        columns = ['direction_x','direction_y','direction_z','direction_kappa1','direction_x1','direction_y1','direction_z1','direction_kappa'] + [f'idx{i}' for i in range(128)]
#    else:
#        columns = ['direction_x','direction_y','direction_z','direction_kappa'] + [f'idx{i}' for i in range(128)]
#else:
#    columns=model.prediction_columns
#results = pd.DataFrame(preds, columns=columns)
#results['event_id'] = np.concatenate(event_ids)
#if validateMode:
#    #results['zenith'] = zenith#np.concatenate(zenith)
#    #results['azimuth'] = azimuth#np.concatenate(azimuth)
#    results['real_x'] = real_x
#    results['real_y'] = real_y
#    results['real_z'] = real_z
#    
#results.sort_values('event_id')
#results.to_csv(f'/remote/ceph/user/l/llorente/prediction_models/stacking_tito/model_checkpoint_graphnet/predictions.csv')


