"""Standard model class(es)."""

from typing import Any, Dict, List, Optional, Union

import torch
from torch import Tensor
from torch.nn import ModuleList
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch_geometric.data import Data
import pandas as pd

from graphnet.models.graphs import GraphDefinition
from graphnet.models.gnn.gnn import GNN
from graphnet.models.model import Model
from graphnet.models.task import Task


class StandardModelStacking(Model):
    """Main class for standard models in graphnet.

    This class chains together the different elements of a complete GNN-based
    model (detector read-in, GNN architecture, and task-specific read-outs).
    """

    def __init__(
        self,
        *,
        tasks: Union[Task, List[Task]],
        n_input_features: int = 3, 
        hidden_size: Optional[int] = 512,
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
        self.n_input_features = n_input_features
        self._dataset = dataset
        
        mlp_layers = []
        layer_sizes = [n_input_features, hidden_size, hidden_size, hidden_size] # todo1
        for nb_in, nb_out in zip(layer_sizes[:-1], layer_sizes[1:]):
            mlp_layers.append(torch.nn.Linear(nb_in, nb_out))
            mlp_layers.append(torch.nn.LeakyReLU())
            mlp_layers.append(torch.nn.Dropout(0.0))
            print("sizes of layers: ", nb_in, nb_out)

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
        x,y,event_nos = xye
        loss, loss_weight = self.shared_step(x, y, idx, istrain=True)
        self.log(
            "train_loss",
            loss,
            batch_size=self._get_batch_size(x),
            prog_bar=True,
            on_epoch=True,
            on_step=False,
            sync_dist=True,
        )
        return loss

    def validation_step(self, xye, idx) -> Tensor:
        """Perform validation step."""
        x,y,event_nos = xye
        loss, loss_weight = self.shared_step(x, y, idx, istrain=False)
        self.log(
            "val_loss",
            loss,
            batch_size=self._get_batch_size(x),
            prog_bar=True,
            on_epoch=True,
            on_step=False,
            sync_dist=True,
        )
        return loss

    def shared_step(self, x, y, batch_idx, istrain) -> Tensor:
        """Perform shared step.

        Applies the forward pass and the following loss calculation, shared
        between the training and validation step.
        """
        preds = self(x)
        batch = Data(x=x, direction=y)

        losses = self._tasks[0].compute_loss(preds[0], batch)
        loss = torch.sum(losses)

        if istrain and self.current_epoch == 0:
            loss_weight = 1
        else:
            current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
            loss_weight = current_lr / 1e-03
        loss = loss*loss_weight

        return loss, loss_weight
    
    def _get_batch_size(self, data: List[Data]) -> int:
        return len(data)
    
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
        distribution_strategy: Optional[str] = "auto",
    ) -> List[Tensor]:
        """Return predictions for `dataloader`."""
        self.inference()
        return super().predict(
            dataloader=dataloader,
            gpus=gpus,
            distribution_strategy=distribution_strategy,
        )