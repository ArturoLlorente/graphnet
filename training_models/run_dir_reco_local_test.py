"""Example of training Model."""

import os
from typing import Any, Dict, List, Optional, Union, Callable
import numpy as np

from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer

from torch.optim.adam import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from graphnet.constants import EXAMPLE_DATA_DIR, EXAMPLE_OUTPUT_DIR
from graphnet.data.constants import FEATURES, TRUTH
from torch.utils.data import DataLoader
from graphnet.data.sqlite import SQLiteDataset
from graphnet.data.dataset import Dataset, EnsembleDataset



from graphnet.models import StandardModel
from graphnet.models.detector.prometheus import Prometheus
from graphnet.models.gnn import DynEdgeTITO
from graphnet.models.graph_builders import KNNGraphBuilder
from graphnet.models.task.reconstruction import (
    DirectionReconstructionWithKappa,
)
from graphnet.training.labels import Direction
from graphnet.training.callbacks import ProgressBar
from graphnet.training.loss_functions import VonMisesFisher3DLoss
from graphnet.training.utils import collate_fn
from graphnet.utilities.argparse import ArgumentParser
from graphnet.utilities.logging import Logger

# Constants
features = FEATURES.PROMETHEUS
truth = TRUTH.PROMETHEUS

def split_selection(selection):
    train, validate = np.split(selection, [int(.9*len(selection))])
    return train.tolist(), validate.tolist()

def main(
    path: str,
    pulsemap: str,
    target: str,
    truth_table: str,
    max_epochs: int,
    early_stopping_patience: int,
    batch_size: int,
    num_workers: int,
    wandb: bool = False,
    labels: Optional[Dict[str, Callable]] = None,
) -> None:
    """Run example."""

    # Configuration
    config: Dict[str, Any] = {
        "path": path,
        "pulsemap": pulsemap,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "target": target,
        "early_stopping_patience": early_stopping_patience,
        "fit": {
            "max_epochs": max_epochs,
        },
    }

    archive = os.path.join(EXAMPLE_OUTPUT_DIR, "train_tito_model")
    run_name = "dynedgeTITO_{}_example".format(config["target"])

    split_selections = {}
    for k in range(len(path)):
        train_selection, validation_selection = split_selection(list(range(0, 330)))
        split_selections[path[k]] = {'train': train_selection,
                                        'validation': validation_selection}
        print(path[k])
    train_datasets = []
    validate_datasets = []
    c = 1
        
    train_datasets = []
    for database in path:
        train_datasets.append(SQLiteDataset(
            path=database,
            pulsemaps=pulsemap,
            features=features,
            truth=truth,
            selection=split_selections[database]['train'],
            node_truth=None,
            truth_table=truth_table,
            node_truth_table=None,
            string_selection=None,
            loss_weight_table=None,
            loss_weight_column=None,
            index_column="event_no",
        ))
        validate_datasets.append(SQLiteDataset(
            path=database,
            pulsemaps=pulsemap,
            features=features,
            truth=truth,
            selection=split_selections[database]['validation'],
            node_truth=None,
            truth_table=truth_table,
            node_truth_table=None,
            string_selection=None,
            loss_weight_table=None,
            loss_weight_column=None,
            index_column="event_no",
        ))
        c += 1
        
    for label in labels.keys():
        for train_dataset in train_datasets:
            train_dataset.add_label(key=label, fn=labels[label])
        for val_dataset in validate_datasets:
            val_dataset.add_label(key=label, fn=labels[label])
            
    train_dataset = EnsembleDataset(train_datasets)
    val_dataset = EnsembleDataset(validate_datasets)
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        persistent_workers=True,
        prefetch_factor=2,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        persistent_workers=True,
        prefetch_factor=2,
    )
    # Building model
    detector = Prometheus(
        graph_builder=KNNGraphBuilder(nb_nearest_neighbours=6),
    )
    gnn = DynEdgeTITO(
        nb_inputs=detector.nb_outputs,
        global_pooling_schemes=["max"],
        layer_size_scale=3,  # 3x the default layer size [256, 256]
    )
    task = DirectionReconstructionWithKappa(
        hidden_size=gnn.nb_outputs,
        target_labels=target,
        loss_function=VonMisesFisher3DLoss(),
    )
    model = StandardModel(
        detector=detector,
        gnn=gnn,
        tasks=[task],
        optimizer_class=Adam,
        optimizer_kwargs={"lr": 1e-03, "eps": 1e-03},
        scheduler_class=ReduceLROnPlateau,
        scheduler_kwargs={
            "patience": config["early_stopping_patience"],
        },
        scheduler_config={
            "frequency": 1,
            "monitor": "val_loss",
        },
    )

    # Training model
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=config["early_stopping_patience"],
        ),
        ProgressBar(),
    ]

    trainer = Trainer(
        default_root_dir=archive + '/' + run_name,
        max_epochs=max_epochs,
        callbacks=callbacks,
        log_every_n_steps=1,
        #logger= wandb_logger if wandb else None,
        #strategy='ddp'
        #resume_from_checkpoint = 
    )
    
    try:
        trainer.fit(model, train_dataloader, val_dataloader)
    except KeyboardInterrupt:
        print("[ctrl+c] Exiting gracefully.")

    # Get predictions
    additional_attributes = [
        "injection_zenith",
        "injection_azimuth",
        "event_no",
        "total_energy"
    ]
    prediction_columns = [
        config["target"][0] + "_x_pred",
        config["target"][0] + "_y_pred",
        config["target"][0] + "_z_pred",
        config["target"][0] + "_kappa_pred",
    ]

    assert isinstance(additional_attributes, list)  # mypy

    results = model.predict_as_dataframe(
        val_dataloader,
        additional_attributes=additional_attributes,
        prediction_columns=prediction_columns,
    )

    # Save predictions and model to file
    db_name = path.split("/")[-1].split(".")[0]
    path = os.path.join(archive, db_name, run_name)
    #logger.info(f"Writing results to {path}")
    os.makedirs(path, exist_ok=True)

    results.to_csv(f"{path}/results.csv")
    model.save_state_dict(f"{path}/state_dict.pth")
    model.save(f"{path}/model.pth")


if __name__ == "__main__":

    path = [f"{EXAMPLE_DATA_DIR}/sqlite/prometheus/prometheus-events.db",
            f"{EXAMPLE_DATA_DIR}/sqlite/prometheus/prometheus-events.db"]
    pulsemap = "total"
    target = ["direction"]
    truth_table = "mc_truth"
    max_epochs = 5
    early_stopping_patience = 2
    batch_size = 16
    num_workers = 8
    wandb = False
    labels = {"direction": Direction(zenith_key="injection_zenith", azimuth_key="injection_azimuth")}
    
    main(
        path,
        pulsemap,
        target,    
        truth_table,
        max_epochs,
        early_stopping_patience,
        batch_size,
        num_workers,
        wandb,
        labels
    )