"""Standard model class(es)."""

from typing import Any, Dict, List, Optional, Union

import torch
from torch import Tensor
from torch.nn import ModuleList
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Data
import pandas as pd

from graphnet.models.coarsening import Coarsening
from graphnet.utilities.config import save_model_config
from graphnet.models.detector.detector import Detector
from graphnet.models.gnn.gnn import GNN
from graphnet.models.model import Model
from graphnet.models.task import Task


class StandardModelSoftTito(Model):
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
        coarsening: Optional[Coarsening] = None,
        optimizer_class: type = Adam,
        optimizer_kwargs: Optional[Dict] = None,
        scheduler_class: Optional[type] = None,
        scheduler_kwargs: Optional[Dict] = None,
        scheduler_config: Optional[Dict] = None,
    ) -> None:
        """Construct `StandardModel`."""
        # Base class constructor
        super().__init__(name=__name__, class_name=self.__class__.__name__)

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

    @property
    def target_labels(self) -> List[str]:
        """Return target label."""
        return [label for task in self._tasks for label in task._target_labels]

    @property
    def prediction_labels(self) -> List[str]:
        """Return prediction labels."""
        return [
            label for task in self._tasks for label in task._prediction_labels
        ]

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
        if self._coarsening:
            data = self._coarsening(data)

        x_list = []
        for i,d in enumerate(data):
            d = self._detector(d)
            x = self._gnn(d)
            x_list.append(x)
        x = torch.cat(x_list)
        
        preds = [task(x) for task in self._tasks]
        cols = ['direction','zenith','azimuth','event_id']
        for col in cols:
            data[0][col] = torch.cat([d[col] for d in data])
        return preds, data[0]

    def training_step(self, train_batch: Data, batch_idx: int) -> Tensor:
        """Perform training step."""
        #TODO1
        preds = self(train_batch)
        vlosses = self._tasks[1].compute_loss(preds[1], train_batch)
        vloss = torch.sum(vlosses)
        
        tlosses = self._tasks[0].compute_loss(preds[0], train_batch)
        tloss = torch.sum(tlosses)
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

    #def compute_loss(
    #    self, preds: Tensor, data: Data, verbose: bool = False
    #) -> Tensor:
    #    """Compute and sum losses across tasks."""
    #    losses = [
    #        task.compute_loss(pred, data)
    #        for task, pred in zip(self._tasks, preds)
    #    ]
    #    if verbose:
    #        self.info(f"{losses}")
    #    assert all(
    #        loss.dim() == 0 for loss in losses
    #    ), "Please reduce loss for each task separately"
    #    return torch.sum(torch.stack(losses))

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

    def predict_as_dataframe(
        self,
        dataloader: DataLoader,
        prediction_columns: Optional[List[str]] = None,
        *,
        node_level: bool = False,
        additional_attributes: Optional[List[str]] = None,
        index_column: str = "event_no",
        gpus: Optional[Union[List[int], int]] = None,
        distribution_strategy: Optional[str] = None,
    ) -> pd.DataFrame:
        """Return predictions for `dataloader` as a DataFrame.

        Include `additional_attributes` as additional columns in the output
        DataFrame.
        """
        if prediction_columns is None:
            prediction_columns = self.prediction_labels
        return super().predict_as_dataframe(
            dataloader=dataloader,
            prediction_columns=prediction_columns,
            node_level=node_level,
            additional_attributes=additional_attributes,
            index_column=index_column,
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
        coarsening: Optional[Coarsening] = None,
        dataset: Dataset = None,
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
        self._dataset = dataset
        self._detector = detector
        self._gnn = gnn
        self._tasks = ModuleList(tasks)
        self._coarsening = coarsening
        self._optimizer_class = optimizer_class
        self._optimizer_kwargs = optimizer_kwargs or dict()
        self._scheduler_class = scheduler_class
        self._scheduler_kwargs = scheduler_kwargs or dict()
        self._scheduler_config = scheduler_config or dict()
        
    def print_timer(self):
        sorted_dict = sorted(self.timer.items(), key=lambda x: x[1], reverse=True)
        print(sorted_dict)

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
        #cols = ['direction','zenith','azimuth','event_id']
        #for col in cols:
        #data['direction'] = torch.cat([d['direction'] for d in data])
        return preds

    def training_step(self, train_batch: Data, batch_idx: int) -> Tensor:
        """Perform training step."""
        #TODO1
        preds = self(train_batch)
        losses = self._tasks[0].compute_loss(preds[0], train_batch)
        loss = torch.sum(losses)
        
        return {"loss": loss}

    def validation_step(self, val_batch: Data, batch_idx: int) -> Tensor:
        """Perform validation step."""
        preds = self(val_batch)
        losses = self._tasks[0].compute_loss(preds[0], val_batch)
        loss = torch.sum(losses)
        
        return {"loss": loss}

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
        self.log_dict(
            {"trn_loss": loss},
            prog_bar=True,
            sync_dist=True,
        )
        print(f'epoch:{self.current_epoch}, train loss:{loss.item()}')
        #self._dataset.reset_epoch()
        
    def validation_epoch_end(self, validation_step_outputs):
        loss = torch.stack([x["loss"] for x in validation_step_outputs]).mean()
        self.log_dict(
            {"val_loss": loss},
            prog_bar=True,
            sync_dist=True,
        )
        print(f'epoch:{self.current_epoch}, valid loss:{loss.item()}')