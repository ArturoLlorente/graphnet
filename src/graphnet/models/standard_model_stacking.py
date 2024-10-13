"""Standard model class(es)."""
from collections import OrderedDict
from typing import Dict, List, Optional, Union, Type

import numpy as np
import torch
from torch import Tensor
from torch.nn import ModuleList
from torch.optim import Adam
from torch.utils.data import DataLoader, SequentialSampler
from torch_geometric.data import Data
import pandas as pd
from graphnet.models.model import Model
from graphnet.models.task import StandardLearnedTask


class StandardModelStacking(Model):
    """Main class for standard models in graphnet.

    This class chains together the different elements of a complete GNN- based
    model (detector read-in, GNN backbone, and task-specific read-outs).
    """

    def __init__(
        self,
        *,
        tasks: Union[StandardLearnedTask, List[StandardLearnedTask]],
        n_input_features: int = 3, 
        hidden_size: Optional[int] = 512,
    ) -> None:
        """Construct `StandardModel`."""
        # Base class constructor
        super().__init__(name=__name__, class_name=self.__class__.__name__)

        # Check(s)
        if isinstance(tasks, StandardLearnedTask):
            tasks = [tasks]
        assert isinstance(tasks, (list, tuple))
        assert all(isinstance(task, StandardLearnedTask) for task in tasks)


        # Member variable(s)
        self._tasks = ModuleList(tasks)

        # Construct GNN        
        mlp_layers = []
        layer_sizes = [n_input_features, hidden_size, hidden_size, hidden_size] # todo1
        for nb_in, nb_out in zip(layer_sizes[:-1], layer_sizes[1:]):
            mlp_layers.append(torch.nn.Linear(nb_in, nb_out))
            mlp_layers.append(torch.nn.LeakyReLU())
            mlp_layers.append(torch.nn.Dropout(0.0))

        self._mlp = torch.nn.Sequential(*mlp_layers)
    def forward(
        self, data: Union[Data, List[Data]]
    ) -> List[Union[Tensor, Data]]:
        """Forward pass, chaining model components."""
        
        data = data.float()
        x = self._mlp(data)

        preds = [task(x) for task in self._tasks]
        return preds

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

    def compute_loss(
        self, preds: Tensor, data: List[Data], verbose: bool = False
    ) -> Tensor:
        """Compute and sum losses across tasks."""
        data_merged = {}
        target_labels_merged = list(set(self.target_labels))
        for label in target_labels_merged:
            data_merged[label] = torch.cat([d[label] for d in data], dim=0)
        for task in self._tasks:
            if task._loss_weight is not None:
                data_merged[task._loss_weight] = torch.cat(
                    [d[task._loss_weight] for d in data], dim=0
                )

        losses = [
            task.compute_loss(pred, data_merged)
            for task, pred in zip(self._tasks, preds)
        ]
        if verbose:
            self.info(f"{losses}")
        assert all(
            loss.dim() == 0 for loss in losses
        ), "Please reduce loss for each task separately"
        return torch.sum(torch.stack(losses))

    def inference(self) -> None:
        """Activate inference mode."""
        for task in self._tasks:
            task.inference()

    def predict(
        self,
        dataloader: DataLoader,
        gpus: Optional[Union[List[int], int]] = None,
        distribution_strategy: Optional[str] = "auto",
    ) -> List[Tensor]:
        """Return predictions for `dataloader`."""
        self.inference()
        self.train(mode=False)

        callbacks = self._create_default_callbacks(
            val_dataloader=None,
        )

        inference_trainer = self._construct_trainer(
            gpus=gpus,
            distribution_strategy=distribution_strategy,
            callbacks=callbacks,
        )

        predictions_list = inference_trainer.predict(self, dataloader)
        assert len(predictions_list), "Got no predictions"

        nb_outputs = len(predictions_list[0])
        predictions: List[Tensor] = [
            torch.cat([preds[ix] for preds in predictions_list], dim=0)
            for ix in range(nb_outputs)
        ]
        return predictions

    def predict_as_dataframe(
        self,
        dataloader: DataLoader,
        prediction_columns: Optional[List[str]] = None,
        *,
        additional_attributes: Optional[List[str]] = None,
        gpus: Optional[Union[List[int], int]] = None,
        distribution_strategy: Optional[str] = "auto",
    ) -> pd.DataFrame:
        """Return predictions for `dataloader` as a DataFrame.

        Include `additional_attributes` as additional columns in the output
        DataFrame.
        """
        if prediction_columns is None:
            prediction_columns = self.prediction_labels

        if additional_attributes is None:
            additional_attributes = []
        assert isinstance(additional_attributes, list)

        if (
            not isinstance(dataloader.sampler, SequentialSampler)
            and additional_attributes
        ):
            print(dataloader.sampler)
            raise UserWarning(
                "DataLoader has a `sampler` that is not `SequentialSampler`, "
                "indicating that shuffling is enabled. Using "
                "`predict_as_dataframe` with `additional_attributes` assumes "
                "that the sequence of batches in `dataloader` are "
                "deterministic. Either call this method a `dataloader` which "
                "doesn't resample batches; or do not request "
                "`additional_attributes`."
            )
        self.info(f"Column names for predictions are: \n {prediction_columns}")
        predictions_torch = self.predict(
            dataloader=dataloader,
            gpus=gpus,
            distribution_strategy=distribution_strategy,
        )
        predictions = (
            torch.cat(predictions_torch, dim=1).detach().cpu().numpy()
        )
        assert len(prediction_columns) == predictions.shape[1], (
            f"Number of provided column names ({len(prediction_columns)}) and "
            f"number of output columns ({predictions.shape[1]}) don't match."
        )

        # Check if predictions are on event- or pulse-level
        pulse_level_predictions = len(predictions) > len(dataloader.dataset)

        # Get additional attributes
        attributes: Dict[str, List[np.ndarray]] = OrderedDict(
            [(attr, []) for attr in additional_attributes]
        )
        for batch in dataloader:
            for attr in attributes:
                attribute = batch[attr]
                if isinstance(attribute, torch.Tensor):
                    attribute = attribute.detach().cpu().numpy()

                # Check if node level predictions
                # If true, additional attributes are repeated
                # to make dimensions fit
                if pulse_level_predictions:
                    if len(attribute) < np.sum(
                        batch.n_pulses.detach().cpu().numpy()
                    ):
                        attribute = np.repeat(
                            attribute, batch.n_pulses.detach().cpu().numpy()
                        )
                attributes[attr].extend(attribute)

        # Confirm that attributes match length of predictions
        skip_attributes = []
        for attr in attributes.keys():
            try:
                assert len(attributes[attr]) == len(predictions)
            except AssertionError:
                self.warning_once(
                    "Could not automatically adjust length"
                    f" of additional attribute '{attr}' to match length of"
                    f" predictions.This error can be caused by heavy"
                    " disagreement between number of examples in the"
                    " dataset vs. actual events in the dataloader, e.g. "
                    " heavy filtering of events in `collate_fn` passed to"
                    " `dataloader`. This can also be caused by requesting"
                    " pulse-level attributes for `Task`s that produce"
                    " event-level predictions. Attribute skipped."
                )
                skip_attributes.append(attr)

        # Remove bad attributes
        for attr in skip_attributes:
            attributes.pop(attr)
            additional_attributes.remove(attr)

        data = np.concatenate(
            [predictions]
            + [
                np.asarray(values)[:, np.newaxis]
                for values in attributes.values()
            ],
            axis=1,
        )

        results = pd.DataFrame(
            data, columns=prediction_columns + additional_attributes
        )
        return results