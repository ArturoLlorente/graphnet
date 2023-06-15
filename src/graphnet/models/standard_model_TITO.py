from typing import Any, Dict, List, Optional, Union

import torch
from torch import Tensor
from torch.nn import ModuleList
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch_geometric.data import Data

from graphnet.data.dataset import Dataset
from graphnet.models.coarsening import Coarsening
from graphnet.utilities.config import save_model_config
from graphnet.models.detector.detector import Detector
from graphnet.models.gnn.gnn import GNN
from graphnet.models.model import Model
from graphnet.models.task import Task
from collections import defaultdict
import time


class StandardModelTito(Model):
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
        dataset: Dataset,
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
        self.timer = defaultdict(int)
        
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
        #data = self._detector(data)
        #x = self._gnn(data)
        x_list = []
        for i,d in enumerate(data):
            st = time.time()
            d = self._detector(d)
            #self.timer[f'detector_{i}'] += time.time()-st; st = time.time()
            x = self._gnn(d)
            #self.timer[f'gnn_{i}'] += time.time()-st; st = time.time()
            x_list.append(x)
        x = torch.cat(x_list)
        #preds = [task(x) for task in self._tasks]
        #self.timer[f'task'] += time.time()-st; st = time.time()
        preds = [self._tasks[0](x)]
        cols = ['direction','zenith','azimuth','event_id']
        for col in cols:
            data[0][col] = torch.cat([d[col] for d in data])
        #self.timer[f'fix_col'] += time.time()-st; st = time.time()
        #self.print_timer()
        return preds, data[0]

    def training_step(self, train_batch: Data, batch_idx: int) -> Tensor:
        """Perform training step."""
        #TODO1
        preds, train_batch = self(train_batch)
        losses = self._tasks[0].compute_loss(preds[0], train_batch)
        loss = torch.sum(losses)

        #x = self.current_epoch/self._max_epochs
        #x = 0.5 + x/2
        #y = 1-x
        #loss = vloss*y + tloss*x
        return {"loss": loss}

    def validation_step(self, val_batch: Data, batch_idx: int) -> Tensor:
        """Perform validation step."""
        preds, val_batch = self(val_batch)        
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
        self._dataset.reset_epoch()
        
    def validation_epoch_end(self, validation_step_outputs):
        loss = torch.stack([x["loss"] for x in validation_step_outputs]).mean()
        self.log_dict(
            {"val_loss": loss},
            prog_bar=True,
            sync_dist=True,
        )
        print(f'epoch:{self.current_epoch}, valid loss:{loss.item()}')