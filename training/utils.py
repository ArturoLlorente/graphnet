from typing import Dict, List, Optional, Union, Any, Callable
import os
import torch
from torch.utils.data import DataLoader

from graphnet.data.dataset import EnsembleDataset
from graphnet.data.dataset import SQLiteDataset, ParquetDataset
from graphnet.models.detector.detector import Detector

from graphnet.constants import ICECUBE_GEOMETRY_TABLE_DIR


def make_dataloaders(
    db: Union[List[str], str],
    train_selection: Optional[List[int]],
    val_selection: Optional[List[int]],
    backend: str,
    config: Dict[str, Any],
    train: Optional[bool] = True,
) -> DataLoader:

    """Construct `DataLoader` instance for training and validation."""
    # Check(s)
    if isinstance(config["pulsemap"], str):
        config["pulsemap"] = [config["pulsemap"]]
        
    dataset_kwags = {
        "graph_definition": config["graph_definition"],
        "pulsemaps": config["pulsemap"],
        "features": config["features"],
        "truth": config["truth"],
        "truth_table": config["truth_table"],
        "index_column": config["index_column"],
        }
        
    dataloader_kwargs = {
        "batch_size": config["batch_size"],
        "num_workers": config["num_workers"],
        "collate_fn": config["collate_fn"],
        "persistent_workers": config["persistent_workers"],
        "prefetch_factor": config["prefetch_factor"],
        #"multiprocessing_context": "spawn", # For parquet multiprocessing
    }
    if backend == "sqlite":
        train_datasets = []
        for db_idx, database in enumerate(db):
            train_datasets.append(
                SQLiteDataset(
                    path=database,
                    selection=train_selection[db_idx],
                    **dataset_kwags,
                )
            )

        if isinstance(config["labels"], dict):
            for label in config["labels"].keys():
                for train_dataset in train_datasets:
                    train_dataset.add_label(
                        key=label, fn=config["labels"][label]
                    )

        train_dataset = EnsembleDataset(train_datasets)
    elif backend == "parquet":
        assert isinstance(train_selection, list)
        train_dataset = ParquetDataset(
            path=db,
            selection=train_selection,
            **dataset_kwags,
        )
        if isinstance(config["labels"], dict):
            for label in config["labels"].keys():
                train_dataset.add_label(
                    key=label, fn=config["labels"][label]
                )

    training_dataloader = DataLoader(
        dataset=train_dataset,
        shuffle=train,
        **dataloader_kwargs,
    )

    if val_selection is not None:
        if backend == "sqlite":
            val_dataset = SQLiteDataset(
                path=db[-1],
                selection=val_selection,
                **dataset_kwags,
            )
        elif backend == "parquet":
            val_dataset = ParquetDataset(
                path=db,
                selection=val_selection,
                **dataset_kwags,
            )

        if isinstance(config["labels"], dict):
            for label in config["labels"].keys():
                val_dataset.add_label(
                    key=label, fn=config["labels"][label]
                )

        validation_dataloader = DataLoader(
            dataset=val_dataset,
            shuffle=False,
            **dataloader_kwargs,
        )
    else:
        validation_dataloader = None
        val_dataset = None
        
    return (
        training_dataloader,
        validation_dataloader,
        train_dataset,
        val_dataset,
    )


class IceCube86IceMix(Detector):
    """`Detector` class for IceCube-86."""

    geometry_table_path = os.path.join(
        ICECUBE_GEOMETRY_TABLE_DIR, "icecube86.parquet"
    )
    xyz = ["dom_x", "dom_y", "dom_z"]
    string_id_column = "string"
    sensor_id_column = "sensor_id"

    def feature_map(self) -> Dict[str, Callable]:
        """Map standardization functions to each dimension of input data."""
        feature_map = {
            "dom_x": self._dom_xyz,
            "dom_y": self._dom_xyz,
            "dom_z": self._dom_xyz,
            "dom_time": self._dom_time,
            "charge": self._charge,
            "rde": self._rde,
            "pmt_area": self._pmt_area,
            "hlc": self._identity,
        }
        return feature_map

    def _dom_xyz(self, x: torch.tensor) -> torch.tensor:
        return x / 500.0

    def _dom_time(self, x: torch.tensor) -> torch.tensor:
        return (x - 1.0e04) / 3.0e4

    def _charge(self, x: torch.tensor) -> torch.tensor:
        return torch.log10(x) / 3

    def _rde(self, x: torch.tensor) -> torch.tensor:
        return (x - 1) / 0.35

    def _pmt_area(self, x: torch.tensor) -> torch.tensor:
        return x / 0.05
    

def rename_state_dict_keys(model, checkpoint):
    """This function is used in the case it is 
    needed to load the original model weights."""
    ckpt_key_idx, passed_keys = 0, 0
    new_checkpoint = {} # new state dict
    model_keys = list(model.state_dict().keys())
    checkpoint_keys = list(checkpoint.keys())
    
    if (model_keys == checkpoint_keys): # If the keys are the same no need to rename
        new_checkpoint = checkpoint
    else:
        for model_k in model_keys: # Iterate over the model keys
            if model_k == '_tasks.0._affine.weight': # task weight key
                ckpt_value = checkpoint['proj_out.weight']
                new_checkpoint[model_k] = ckpt_value
                
            elif model_k == '_tasks.0._affine.bias': # task bias key
                ckpt_value = checkpoint['proj_out.bias']
                new_checkpoint[model_k] = ckpt_value
                
            else: # Other keys
                if checkpoint_keys[ckpt_key_idx] == 'proj_out.weight': # If checkpoint key is the task weight, skip weight and bias term. Assumes they are together.
                    passed_keys += 2
                ckpt_value = checkpoint[checkpoint_keys[ckpt_key_idx+passed_keys]]
                new_checkpoint[model_k] = ckpt_value
                ckpt_key_idx+=1  
    return new_checkpoint