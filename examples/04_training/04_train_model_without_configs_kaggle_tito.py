"""Example of training Model."""

import os
from typing import Any, Dict, List, Optional

from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
import torch
from torch.optim.adam import Adam

from graphnet.constants import EXAMPLE_DATA_DIR, EXAMPLE_OUTPUT_DIR
from graphnet.data.constants import FEATURES, TRUTH
from graphnet.models.standard_model1_tito import StandardModel2
from graphnet.models.detector.prometheus import Prometheus
from graphnet.models.gnn import DynEdge
from graphnet.models.gnn.dynedge_tito_kaggle1 import DynEdgeTITO
from graphnet.models.graph_builders import KNNGraphBuilder
from graphnet.models.task.reconstruction import (
    EnergyReconstruction,
    DirectionReconstructionWithKappa,
)
from graphnet.training.callbacks import ProgressBar, PiecewiseLinearLR
from graphnet.training.loss_functions import LogCoshLoss, DistanceLoss2
from graphnet.training.labels import Direction
from graphnet.training.utils import (
    make_train_validation_dataloader,
    collate_fn_tito,
    collate_fn,
    make_dataloaders2,
    inference,
)
from graphnet.utilities.argparse import ArgumentParser
from graphnet.utilities.logging import Logger

# Constants
features = FEATURES.KAGGLE
truth = TRUTH.KAGGLE

print(features, type(features))
print(truth, type(truth))

TRAIN_MODE = False
DROPOUT = 0.0
NB_NEAREST_NEIGHBOURS = [6]
COLUMNS_NEAREST_NEIGHBOURS = [slice(0, 4)]
USE_G = True
ONLY_AUX_FALSE = False
SERIAL_CONNECTION = True
USE_PP = True
USE_TRANS_IN_LAST = 0
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


def main(
    path: str,
    pulsemap: str,
    target: str,
    truth_table: str,
    gpus: Optional[List[int]],
    max_epochs: int,
    early_stopping_patience: int,
    batch_size: int,
    num_workers: int,
    wandb: bool = False,
) -> None:
    """Run example."""
    # Construct Logger
    logger = Logger()

    # Initialise Weights & Biases (W&B) run
    if wandb:
        # Make sure W&B output directory exists
        wandb_dir = "./wandb/"
        os.makedirs(wandb_dir, exist_ok=True)
        wandb_logger = WandbLogger(
            project="example-script",
            entity="graphnet-team",
            save_dir=wandb_dir,
            log_model=True,
        )

    logger.info(f"features: {features}")
    logger.info(f"truth: {truth}")

    # Configuration
    config: Dict[str, Any] = {
        "path": path,
        "pulsemap": pulsemap,
        "batch_size": batch_size,
        "train_batch_ids": list(range(1, early_stopping_patience + 1)),
        "valid_batch_ids": [660],  # only suport one batch
        "num_workers": num_workers,
        "target": target,
        "early_stopping_patience": early_stopping_patience,
        "features": features,
        "truth": truth,
        "truth_table": "meta_table",  # dummy
        "direction": Direction(),
        "train_len": 0,  # not using anymore
        "valid_len": 0,
        "train_max_pulse": 300,
        "valid_max_pulse": 200,
        "train_min_pulse": 0,
        "valid_min_pulse": 0,
        "index_column": "event_no",
        "fit": {
            "gpus": gpus,
            "max_epochs": max_epochs,
        },
    }

    archive = os.path.join(EXAMPLE_OUTPUT_DIR, "train_model_without_configs")
    run_name = "dynedge_{}_example".format(config["target"])
    if wandb:
        # Log configuration to W&B
        wandb_logger.experiment.config.update(config)

    (
        train_dataloader,
        validate_dataloader,
        train_dataset,
        validate_dataset,
    ) = make_dataloaders2(
        config=config,
    )

    # Building model
    detector = Prometheus(
        graph_builder=KNNGraphBuilder(nb_nearest_neighbours=8),
    )
    gnn = DynEdgeTITO(
        nb_inputs=detector.nb_outputs,
        dynedge_layer_sizes=DYNEDGE_LAYER_SIZE,
        global_pooling_schemes=["max"],
        add_global_variables_after_pooling=True,
    )
    task = DirectionReconstructionWithKappa(
        hidden_size=gnn.nb_outputs,
        target_labels=config["target"],
        loss_function=DistanceLoss2(),
    )
    model = StandardModel2(
        detector=detector,
        gnn=gnn,
        tasks=[task],
        dataset=train_dataset,
        max_epochs=config["fit"]["max_epochs"],
        optimizer_class=Adam,
        optimizer_kwargs={"lr": 1e-03, "eps": 1e-03},
        scheduler_class=PiecewiseLinearLR,
        scheduler_kwargs={
            "milestones": [
                0,
                len(train_dataset) / 2,
                len(train_dataset) * config["fit"]["max_epochs"],
            ],
            "factors": [1e-03, 1, 1e-03],
        },
        scheduler_config={
            "interval": "step",
        },
        use_all_fea_in_pred=False,
    )

    print(model)

    # Training model
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=config["early_stopping_patience"],
        ),
        ProgressBar(),
    ]

    model.fit(
        train_dataloader,
        validate_dataloader,
        callbacks=callbacks,
        logger=wandb_logger if wandb else None,
        **config["fit"],
    )

    # Get predictions
    additional_attributes = model.target_labels
    assert isinstance(additional_attributes, list)  # mypy

    results = model.predict_as_dataframe(
        validate_dataloader,
        additional_attributes=additional_attributes + ["event_no"],
        prediction_columns=model.prediction_columns,
    )

    # Save predictions and model to file
    db_name = path.split("/")[-1].split(".")[0]
    path = os.path.join(archive, db_name, run_name)
    logger.info(f"Writing results to {path}")
    os.makedirs(path, exist_ok=True)

    results.to_csv(f"{path}/results.csv")
    model.save_state_dict(f"{path}/state_dict.pth")
    model.save(f"{path}/model.pth")


if __name__ == "__main__":

    # Parse command-line arguments
    parser = ArgumentParser(
        description="""
Train GNN model without the use of config files.
"""
    )

    parser.add_argument(
        "--path",
        help="Path to dataset file (default: %(default)s)",
        default=f"{EXAMPLE_DATA_DIR}/sqlite/prometheus/prometheus-events.db",
    )

    parser.add_argument(
        "--pulsemap",
        help="Name of pulsemap to use (default: %(default)s)",
        default="total",
    )

    parser.add_argument(
        "--target",
        help=(
            "Name of feature to use as regression target (default: "
            "%(default)s)"
        ),
        default="total_energy",
    )

    parser.add_argument(
        "--truth-table",
        help="Name of truth table to be used (default: %(default)s)",
        default="mc_truth",
    )

    parser.with_standard_arguments(
        "gpus",
        ("max-epochs", 5),
        "early-stopping-patience",
        ("batch-size", 16),
        ("num-workers", 8),
    )

    parser.add_argument(
        "--wandb",
        action="store_true",
        help="If True, Weights & Biases are used to track the experiment.",
    )

    args = parser.parse_args()

    main(
        args.path,
        args.pulsemap,
        args.target,
        args.truth_table,
        args.gpus,
        args.max_epochs,
        args.early_stopping_patience,
        args.batch_size,
        args.num_workers,
        args.wandb,
    )
