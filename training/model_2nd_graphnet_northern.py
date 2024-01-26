import os
import sys
import pandas as pd
from tqdm.auto import tqdm
import torch
import pickle
from torch.optim import AdamW, Adam
from torch.optim.lr_scheduler import OneCycleLR

from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    GradientAccumulationScheduler,
)

from pytorch_lightning.loggers import WandbLogger

from graphnet.data.dataset import EnsembleDataset
from graphnet.data.dataset import SQLiteDataset
from graphnet.data.constants import FEATURES, TRUTH
from graphnet.models import StandardModel, StandardAverageModel
from graphnet.models.graphs import KNNGraph
from graphnet.models.graphs.nodes import NodesAsPulses
from graphnet.models.task.reconstruction import DirectionReconstructionWithKappa
from graphnet.training.labels import Direction, Direction_flipped
from graphnet.training.loss_functions import VonMisesFisher3DLoss
from graphnet.training.callbacks import ProgressBar
from graphnet.training.utils import make_dataloader
from graphnet.training.utils import collate_fn, collator_sequence_buckleting

from typing import Dict, List, Optional, Union, Any
from torch.utils.data import DataLoader

from graphnet.models.detector.detector import Detector
from graphnet.models.detector.icecube import IceCube86


#sys.path.insert(0, "/remote/ceph/user/l/llorente/")
from graphnet.models.gnn import DeepIceModel, EncoderWithDirectionReconstruction

MODELS = {'model1': DeepIceModel(dim=768, dim_base=192, depth=12, head_size=32),
        'model2': DeepIceModel(dim=768, dim_base=192, depth=12, head_size=64),
        'model3': DeepIceModel(dim=768, dim_base=192, depth=12, head_size=32, n_rel=4),
        'model4': EncoderWithDirectionReconstruction(dim=384, dim_base=128, depth=8, head_size=32, knn_features=3),
        'model5': EncoderWithDirectionReconstruction(dim=768, dim_base=192, depth=12, head_size=64, knn_features=4)}

MODEL_PATH = {'model1': '/remote/ceph/user/l/llorente/icecube_2nd_place/ice-cube-final-models/baselineV3_BE_globalrel_d32_0_6ema.pth',
            'model2': '/remote/ceph/user/l/llorente/icecube_2nd_place/ice-cube-final-models/baselineV3_BE_globalrel_d64_0_3emaFT_2.pth',
            'model3': '/remote/ceph/user/l/llorente/icecube_2nd_place/ice-cube-final-models/VFTV3_4RELFT_7.pth',
            'model4': '/remote/ceph/user/l/llorente/icecube_2nd_place/ice-cube-final-models/V22FT6_1.pth',
            'model5': '/remote/ceph/user/l/llorente/icecube_2nd_place/ice-cube-final-models/V23FT5_6.pth'}

MODEL_WEIGHTS = {'model1': 0.08254897,
                'model2': 0.15350807,
                'model3': 0.19367443,
                'model4': 0.23597202,
                'model5': 0.3342965}

def make_dataloaders(
    db: Union[List[str], str],
    train_selection: Optional[List[int]],
    val_selection: Optional[List[int]],
    config: Dict[str, Any],
) -> DataLoader:

    """Construct `DataLoader` instance."""
    # Check(s)
    if isinstance(config["pulsemap"], str):
        config["pulsemap"] = [config["pulsemap"]]
    assert isinstance(train_selection, list)
    if isinstance(db, list):
        assert len(train_selection) == len(db)
        train_datasets = []
        pbar = tqdm(total=len(db))
        for db_idx, database in enumerate(db):

            train_datasets.append(
                SQLiteDataset(
                    path=database,
                    graph_definition=config["graph_definition"],
                    pulsemaps=config["pulsemap"],
                    features=config["features"],
                    truth=config["truth"],
                    selection=train_selection[db_idx],
                    node_truth=config["node_truth"],
                    truth_table=config["truth_table"],
                    node_truth_table=config["node_truth_table"],
                    string_selection=config["string_selection"],
                    loss_weight_table=config["loss_weight_table"],
                    loss_weight_column=config["loss_weight_column"],
                    index_column=config["index_column"],
                )
            )
            pbar.update(1)

        if isinstance(config["labels"], dict):
            for label in config["labels"].keys():
                for train_dataset in train_datasets:
                    train_dataset.add_label(
                        key=label, fn=config["labels"][label]
                    )

        train_dataset = EnsembleDataset(train_datasets)

        training_dataloader = DataLoader(
            dataset=train_dataset,
            batch_size=config["batch_size"],
            shuffle=True,
            num_workers=config["num_workers"],
            collate_fn=config["collate_fn"],
            persistent_workers=config["persistent_workers"],
            prefetch_factor=2,
            #drop_last=True,
        )

        if val_selection is not None:
            val_dataset = SQLiteDataset(
                path=db[-1],
                graph_definition=config["graph_definition"],
                pulsemaps=config["pulsemap"],
                features=config["features"],
                truth=config["truth"],
                selection=val_selection,
                node_truth=config["node_truth"],
                truth_table=config["truth_table"],
                node_truth_table=config["node_truth_table"],
                string_selection=config["string_selection"],
                loss_weight_table=config["loss_weight_table"],
                loss_weight_column=config["loss_weight_column"],
                index_column=config["index_column"],
            )

            if isinstance(config["labels"], dict):
                for label in config["labels"].keys():
                    val_dataset.add_label(
                        key=label, fn=config["labels"][label]
                    )

            validation_dataloader = DataLoader(
                dataset=val_dataset,
                batch_size=config["batch_size"],
                shuffle=False,
                num_workers=config["num_workers"],
                collate_fn=config["collate_fn"],
                persistent_workers=config["persistent_workers"],
                prefetch_factor=2,
                #drop_last=True,
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


def build_model(
    gnn_model,
    config: Dict[str, Any],
):

    gnn = gnn_model

    if gnn_model._get_name() == "EncoderWithDirectionReconstructionV22":
        task = DirectionReconstructionWithKappa(
            hidden_size=384,
            target_labels="direction",
            loss_function=VonMisesFisher3DLoss(),
        )
    else:
        task = DirectionReconstructionWithKappa(
            hidden_size=768,
            target_labels="direction",
            loss_function=VonMisesFisher3DLoss(),
        )

    model_kwargs = {
        "graph_definition": config["graph_definition"],
        "gnn": gnn,
        "tasks": [task],
        "optimizer_class": config["optimizer_class"],
        "optimizer_kwargs": config["optimizer_kwargs"],
        "scheduler_class": config["scheduler_class"],
        "scheduler_kwargs": config["scheduler_kwargs"],
        "scheduler_config": config["scheduler_config"],
    }
    
    if INFERENCE:
        model = StandardModel(**model_kwargs)
    else:
        if isinstance(config["swa_starting_epoch"], int):
            model_kwargs["swa_starting_epoch"] = config["swa_starting_epoch"]
        if config["ema_decay"]:
            model_kwargs["ema_decay"] = config["ema_decay"]
        
        if config["swa_starting_epoch"] or config["ema_decay"]:
            model = StandardAverageModel(**model_kwargs)
        else:
            model = StandardModel(**model_kwargs)
        

    model.prediction_columns = config["prediction_columns"]
    model.additional_attributes = config["additional_attributes"]

    return model


def inference(
    model_name: str,
    device: int,
    checkpoint_path: str,
    test_min_pulses: int,
    test_max_pulses: int,
    batch_size: int,
    test_path: str = None,
    test_selection_file: str = None,
    config: Dict[str, Any] = None,
):

    test_selection = (
        test_selection_file.loc[
            (test_selection_file["n_pulses"] <= test_max_pulses)
            & (test_selection_file["n_pulses"] > test_min_pulses),
            :,][config["index_column"]].ravel().tolist())
    
    test_dataloader = make_dataloader(
        db=test_path,
        selection=test_selection,
        graph_definition=config["graph_definition"],
        pulsemaps=config["pulsemap"],
        num_workers=config["num_workers"],
        features=config["features"],
        shuffle=False,
        truth=config["truth"],
        batch_size=batch_size,
        truth_table=config["truth_table"],
        index_column=config["index_column"],
        labels=config["labels"],
    )

    cuda_device = f"cuda:{device[0]}" if len(device)>0 else "cpu"
    models, weights = [], []
    if model_name == "all":
        for i in range(1,6):
            model = build_model(gnn_model=MODELS[f"model{i}"], config=config)
            model.eval()
            model.inference()
            checkpoint_path = MODEL_PATH[f"model{i}"]
            checkpoint = torch.load(checkpoint_path, torch.device("cpu"))
            if "state_dict" in checkpoint:
                checkpoint = checkpoint["state_dict"]
            new_checkpoint = {('_tasks.0._affine.weight' if k == 'proj_out.weight' else '_tasks.0._affine.bias' if k == 'proj_out.bias' else '_gnn.' + k): v for k, v in checkpoint.items()}

            model.load_state_dict(new_checkpoint)
            model.to(cuda_device)
            models.append(model)
            weights.append(MODEL_WEIGHTS[f"model{i}"])
    else:
        model = build_model(gnn_model=MODELS[model_name], config=config)
        model.eval()
        model.inference()
        checkpoint = torch.load(checkpoint_path, torch.device("cpu"))

        if "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]
        new_checkpoint = {('_tasks.0._affine.weight' if k == 'proj_out.weight' else '_tasks.0._affine.bias' if k == 'proj_out.bias' else '_gnn.' + k): v for k, v in checkpoint.items()} #kaggle weights
        #new_checkpoint = {('_tasks.0._affine.weight' if k == 'proj_out.weight' else '_tasks.0._affine.bias' if k == 'proj_out.bias' else k): v for k, v in checkpoint.items()}
        model.load_state_dict(new_checkpoint)
        model.to(cuda_device)
        models.append(model)
        weights.append(1)
        
    weights = torch.FloatTensor(weights)
    weights /= weights.sum()
    
    event_nos, zenith, azimuth, preds = [], [], [], []
    print("start predict")
    
    with torch.no_grad():
        for batch in tqdm(test_dataloader):

            #x = dict_to(x, device)
            pred = (torch.stack([torch.nan_to_num(model(batch.to(cuda_device))[0]).clip(-1000, 1000) for model in models], -1).cpu() * weights).sum(-1)

            #pred = model(batch.to(cuda_device))
            #preds.append(torch.nan_to_num(pred[0]).clip(-1000,1000))
            preds.append(pred)
            event_nos.append(batch.event_no)
            zenith.append(batch.zenith)
            azimuth.append(batch.azimuth)
    preds = torch.cat(preds).to("cpu").detach().numpy()
    columns = [
        "direction_x",
        "direction_y",
        "direction_z",
        "direction_kappa",
    ]

    results = pd.DataFrame(preds, columns=columns)
    results[config["index_column"]] = (
        torch.cat(event_nos).to("cpu").detach().numpy()
    )

    results["zenith"] = torch.cat(zenith).to("cpu").numpy()
    results["azimuth"] = torch.cat(azimuth).to("cpu").numpy()

    return results


# Main function call
if __name__ == "__main__":

    config = {
        "archive": "/remote/ceph/user/l/llorente/icemix_northern_retrain",
        "target": "direction",
        "weight_column_name": None,
        "weight_table_name": None,
        "batch_size": 85,
        "num_workers": 32,
        "pulsemap": "InIceDSTPulses",
        "truth_table": "truth",
        "index_column": "event_no",
        "labels": {"direction": Direction_flipped()},
        "global_pooling_schemes": ["max"],
        "num_database_files": 8,
        "node_truth_table": None,
        "node_truth": None,
        "string_selection": None,
        "loss_weight_table": None,
        "loss_weight_column": None,
        "persistent_workers": True,
        "detector": IceCube86(),
        "node_definition": NodesAsPulses(),
        "nb_nearest_neighbours": 6,
        "features": ["dom_x", "dom_y", "dom_z", "dom_time", "charge", "hlc", "rde"],
        "truth": ["energy", "energy_track", "position_x", "position_y", "position_z", "azimuth", "zenith", "pid", "elasticity", "interaction_type", "interaction_time"],
        "columns_nearest_neighbours": [0, 1, 2],
        "collate_fn": collator_sequence_buckleting([0.8]),
        "prediction_columns": ["dir_x_pred", "dir_y_pred", "dir_z_pred", "dir_kappa_pred"],
        "fit": {"max_epochs": 1, "gpus": [2], "precision": '16-mixed'},
        "optimizer_class": AdamW,
        "optimizer_kwargs": {"lr": 2e-5, "weight_decay": 0.05, "eps": 1e-7},
        "scheduler_class": OneCycleLR,
        "scheduler_kwargs": {"max_lr": 2e-5, "pct_start": 0.01, "anneal_strategy": 'cos', "div_factor": 25, "final_div_factor": 25},
        "scheduler_config": {"frequency": 1, "monitor": "val_loss", "interval": "step"},
        "wandb": False,
        "ema_decay": 0.9998,
        "swa_starting_epoch": 0,
        "ckpt_path": False
    }
    config["additional_attributes"] = [ "zenith", "azimuth", config["index_column"], "energy"]
    INFERENCE = True
    model_name = "model5"

    config['retrain_from_checkpoint'] = MODEL_PATH[model_name]

    if config["swa_starting_epoch"] is not None:
        config["fit"]["distribution_strategy"] = 'ddp_find_unused_parameters_true'
    else:
        config["fit"]["distribution_strategy"] = 'ddp'

    run_name = (
        f"{model_name}_retrain_IceMix_batch{config['batch_size']}_optimizer_AdamW_LR{config['scheduler_kwargs']['max_lr']}_annealStrat_{config['scheduler_kwargs']['anneal_strategy']}_"
        f"ema_decay_{config['ema_decay']}_new_dataset_definition_26_01"
    )

    # Configurations
    torch.multiprocessing.set_sharing_strategy("file_system")

    db_dir = "/mnt/scratch/rasmus_orsoe/databases/dev_northern_tracks_muon_labels_v3/"
    sel_dir = "/remote/ceph/user/l/llorente/northern_track_selection/"
    all_databases, all_selections = [], []
    test_idx = 5
    for idx in range(1, config["num_database_files"] + 1):
        if idx == test_idx:
            test_database = (
                db_dir + f"dev_northern_tracks_muon_labels_v3_part_{idx}.db"
            )
            test_selection = pd.read_csv(sel_dir + f"part_{idx}.csv")
        else:
            all_databases.append(
                db_dir + f"dev_northern_tracks_muon_labels_v3_part_{idx}.db"
            )
            all_selections.append(pd.read_csv(sel_dir + f"part_{idx}.csv"))          
    # get_list_of_databases:
    train_selections = []
    for selection in all_selections:
        train_selections.append(selection.loc[selection["n_pulses"],:][config["index_column"]].ravel().tolist())
    train_selections[-1] = train_selections[-1][:int(len(train_selections[-1]) * 0.9)]
    
    val_selection = (selection.loc[(selection["n_pulses"]), :][config["index_column"]].ravel().tolist())
    val_selection = val_selection[int(len(val_selection) * 0.9) :]


    config["graph_definition"] = KNNGraph(
        detector=config["detector"],
        node_definition=config["node_definition"],
        nb_nearest_neighbours=config["nb_nearest_neighbours"],
        input_feature_names=config["features"],
        columns=config["columns_nearest_neighbours"],
    )

    if INFERENCE:
        config["scheduler_kwargs"] = None
    else:
        (
            training_dataloader,
            validation_dataloader,
            train_dataset,
            val_dataset,
        ) = make_dataloaders(
            db=all_databases,
            train_selection=train_selections,
            val_selection=val_selection,
            config=config,
        )

        if config["scheduler_kwargs"]:
            config["scheduler_kwargs"]["steps_per_epoch"] = len(training_dataloader)
            config["scheduler_kwargs"]["epochs"] = config["fit"]["max_epochs"]

        callbacks = [
            ProgressBar(),
            GradientAccumulationScheduler(scheduling={0: 4096//config["batch_size"]}),
        ]
        if validation_dataloader is not None:
            callbacks.append(
                ModelCheckpoint(
                    dirpath=config["archive"] + "/model_checkpoint_graphnet/",
                    filename=run_name + "-{epoch:02d}-{val_loss:.6f}-{train_loss:.6f}",
                    monitor="val_loss",
                    save_top_k=30,
                    every_n_epochs=1,
                    save_weights_only=False,
                )
            )
    
    if not INFERENCE:
        model = build_model(gnn_model=MODELS[model_name], config=config)
        if config["ckpt_path"]:
            config["fit"]["ckpt_path"] = config["ckpt_path"]

        if config["retrain_from_checkpoint"]:
            checkpoint_path = config["retrain_from_checkpoint"]
            print("Loading weights from ...", checkpoint_path)
            checkpoint = torch.load(checkpoint_path, torch.device("cpu"))
            if "state_dict" in checkpoint:
                checkpoint = checkpoint["state_dict"]
            new_checkpoint = {('_tasks.0._affine.weight' if k == 'proj_out.weight' else '_tasks.0._affine.bias' if k == 'proj_out.bias' else '_gnn.' + k): v for k, v in checkpoint.items()}
            model.load_state_dict(new_checkpoint, strict=False)
            del checkpoint, new_checkpoint
        if validation_dataloader is not None:
            config["fit"]["val_dataloader"] = validation_dataloader

        model.fit(
            training_dataloader,
            callbacks=callbacks,
            **config["fit"],
        )
        
        #save config into a file
        with open(os.path.join(config["archive"], f"{run_name}_config.pkl"), "wb") as f:
            pickle.dump(config, f)
            
        model.save(os.path.join(config["archive"], f"{run_name}.pth"))
        model.save_state_dict(
            os.path.join(config["archive"], f"{run_name}_state_dict.pth")
        )
        print(f"Model saved to {config['archive']}/{run_name}.pth")
    else:

        all_res = []
        checkpoint_path = MODEL_PATH["model5"]#'/remote/ceph/user/l/llorente/icemix_northern_retrain/model5_retrain_IceMix__batch85_optimizer_AdamW_LR2e-05_annealStrat_cos_ema_decay_0.9998__23_01_state_dict.pth'
        run_name_pred = f"pred_icemix_model5_baseline_full"
        
        factor = 1
        pulse_breakpoints = [0, 100, 200, 300, 500, 1000, 1500, 3000]  # 10000]
        batch_sizes_per_pulse = [1800, 750, 350, 150, 35,30,30]#[1800, 175, 40, 11, 4]  # 5, 2]
        config["num_workers"] = 16
        
        #test_selection = test_selection[:int(len(test_selection)*0.01)]

        for min_pulse, max_pulse in zip(
            pulse_breakpoints[:-1], pulse_breakpoints[1:]
        ):
            print(
                f"predicting {min_pulse} to {max_pulse} pulses with batch size {int(factor*batch_sizes_per_pulse[pulse_breakpoints.index(max_pulse)-1])}"
            )
            pred_checkpoint_path = f"/remote/ceph/user/l/llorente/IceMix_solution_northern/backup_inference/{run_name_pred}_{min_pulse}to{max_pulse}pulses.pkl"
            if os.path.exists(pred_checkpoint_path):
                results = pickle.load(open(pred_checkpoint_path, "rb"))
            else:
                results = inference(
                    model_name = model_name,
                    device=config["fit"]["gpus"],
                    test_min_pulses=min_pulse,
                    test_max_pulses=max_pulse,
                    batch_size=max(int(
                        factor * batch_sizes_per_pulse[pulse_breakpoints.index(max_pulse) - 1]
                    ), 1),
                    checkpoint_path=checkpoint_path,
                    test_path=test_database,
                    test_selection_file=test_selection,
                    config=config,
                )
                pickle.dump(results, open(pred_checkpoint_path, "wb"))

            all_res.append(results)
            del results
            torch.cuda.empty_cache()

        results = pd.concat(all_res).sort_values(config["index_column"])
        
        path_to_save = "/remote/ceph/user/l/llorente/IceMix_solution_northern"
        results.to_csv(f"{path_to_save}/{run_name_pred}.csv")
        print(f"predicted and saved in {path_to_save}/{run_name_pred}.csv")
