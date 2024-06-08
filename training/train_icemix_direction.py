import os
import pandas as pd
from tqdm.auto import tqdm
import torch
import pickle
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from typing import Dict, Any

from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    GradientAccumulationScheduler,
)

from graphnet.models import StandardModel
from graphnet.models.graphs import KNNGraph
from graphnet.models.graphs.nodes import IceMixNodes
from graphnet.models.gnn import DeepIce
from graphnet.models.task.reconstruction import DirectionReconstructionWithKappa
from graphnet.training.labels import Direction, Direction_flipped
from graphnet.training.loss_functions import VonMisesFisher3DLoss
from graphnet.training.callbacks import ProgressBar
from graphnet.training.utils import make_dataloader
from graphnet.training.utils import collate_fn, collator_sequence_buckleting

from utils import make_dataloaders, IceCube86IceMix, rename_state_dict_keys

from graphnet.constants import GRAPHNET_ROOT_DIR
PRETRAINED_MODEL_DIR = os.path.join(GRAPHNET_ROOT_DIR, "src", "graphnet", "models", "pretrained", "icecube", "kaggle", "icemix", "neutrino_direction")

# 5 models 
MODELS = {'B_d32': DeepIce(hidden_dim=768, seq_length=192, depth=12, head_size=32, n_features=7),
          'B_d64': DeepIce(hidden_dim=768, seq_length=192, depth=12, head_size=64, n_features=7),
          'B_d32_4rel': DeepIce(hidden_dim=768, seq_length=192, depth=12, head_size=32, n_features=7, n_rel=4),
          'S+DynEdge_d32': DeepIce(hidden_dim=384, seq_length=128, depth=8, head_size=32, include_dynedge=True, scaled_emb=True, n_features=7, n_rel=1),
          'B+DynEdge_d64': DeepIce(hidden_dim=768, seq_length=192, depth=12, head_size=64, include_dynedge=True, scaled_emb=True, n_features=7, n_rel=4)}

MODEL_PATH = {'B_d32': PRETRAINED_MODEL_DIR+'/B_d32/baselineV3_BE_globalrel_d32_0_6ema.pth',
              'B_d64': PRETRAINED_MODEL_DIR+'/B_d64/baselineV3_BE_globalrel_d64_0_3emaFT_2.pth',
              'B_d32_4rel': PRETRAINED_MODEL_DIR+'/B_d32_4rel/VFTV3_4RELFT_7.pth',
              'S+DynEdge_d32': PRETRAINED_MODEL_DIR+'/S+DynEdge_d32/V22FT6_1.pth',
              'B+DynEdge_d64': PRETRAINED_MODEL_DIR+'/B+DynEdge_d64/V23FT5_6.pth',
              }

MODEL_WEIGHTS = {'B_d32': 0.08254897,
                 'B_d64': 0.15350807,
                 'B_d32_4rel': 0.19367443,
                 'B+DynEdge_d64': 0.23597202,
                 'S+DynEdge_d32': 0.3342965,
                 }

def build_model(
    backbone,
    config: Dict[str, Any],
):
    
    task = DirectionReconstructionWithKappa(
        hidden_size=backbone.nb_outputs,
        target_labels="direction",
        loss_function=VonMisesFisher3DLoss(),
    )

    model_kwargs = {
        "graph_definition": config["graph_definition"],
        "backbone": backbone,
        "tasks": [task],
        "optimizer_class": config["optimizer_class"],
        "optimizer_kwargs": config["optimizer_kwargs"],
        "scheduler_class": config["scheduler_class"],
        "scheduler_kwargs": config["scheduler_kwargs"],
        "scheduler_config": config["scheduler_config"],
    }
    
    model = StandardModel(**model_kwargs)

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
        
    test_selection = list([(
        sel.loc[(sel["n_pulses"] <= test_max_pulses) & (sel["n_pulses"] > test_min_pulses),
                :,][config["index_column"]].to_numpy()) for sel in test_selection_file])
    
    if isinstance(test_path, str):
        test_dataloader = make_dataloader(
            db=test_path,
            selection=test_selection[0],
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
    else:
        config["batch_size"] = batch_size
        config["collate_fn"] = collate_fn
        (
            test_dataloader,
            _,
            test_dataset,
            _,
        ) = make_dataloaders(
            db=test_path,
            train_selection=test_selection,
            val_selection=None,
            config=config,
            train=False,
            backend="sqlite",
        )

    cuda_device = f"cuda:{device[0]}" if len(device)>0 else "cpu"
    models, weights = [], []
    if model_name == "all":
        for model_i, model_architecture in MODELS.items():
            model = build_model(backbone=model_architecture, config=config)
            model.eval()
            model.inference()
            checkpoint_path = MODEL_PATH[model_i]
            checkpoint = torch.load(checkpoint_path, torch.device("cpu"))
            if "state_dict" in checkpoint:
                checkpoint = checkpoint["state_dict"]
                
            new_checkpoint = rename_state_dict_keys(model, checkpoint) 
            model.load_state_dict(new_checkpoint)
            model.to(cuda_device)
            models.append(model)
            weights.append(MODEL_WEIGHTS[model_i])
    else:
        model = build_model(backbone=MODELS[model_name], config=config)
        model.eval()
        model.inference()
        checkpoint = torch.load(checkpoint_path, torch.device("cpu"))

        if "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]
        
        new_checkpoint = rename_state_dict_keys(model, checkpoint)
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
            batch_cuda = batch.to(cuda_device)
            pred = (torch.stack([torch.nan_to_num(model(batch_cuda)[0]).clip(-1000, 1000) for model in models], -1).cpu() * weights).sum(-1)
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
     # Added rde to the standard. Change the ordering. Important the ordering when using SinusoidalPosEmb
    icemix_features = ["dom_x", "dom_y", "dom_z", "dom_time", "charge", "hlc", "rde"]

    config = {
        "archive": "/scratch/users/allorana/icemix_cascades_retrain",
        "target": "direction",
        "batch_size": 12, # 12 for cascades, 16 for tracks
        "num_workers": 8,
        "pulsemap": "InIceDSTPulses",
        "truth_table": "truth",
        "index_column": "event_no",
        "labels": {"direction": Direction_flipped()},
        "persistent_workers": True,
        "detector": IceCube86IceMix(),
        "node_definition": IceMixNodes(input_feature_names=icemix_features,
                                       max_pulses=3000,
                                       ),
        "nb_nearest_neighbours": 8,
        "features": icemix_features,
        "truth": ["energy", "energy_track", "position_x", "position_y", "position_z", "azimuth", "zenith", "pid", "elasticity", "interaction_type"],
        "columns_nearest_neighbours": [0, 1, 2], # [0, 1, 2, 3] for model5
        "collate_fn": collator_sequence_buckleting([0.5,0.9]),
        "fit": {"max_epochs": 1, "gpus": [0], "precision": '16-mixed'},
        "optimizer_class": AdamW,
        "optimizer_kwargs": {"lr": 0.35e-5, "weight_decay": 0.05, "eps": 1e-7}, 
        "scheduler_class": OneCycleLR, 
        #"scheduler_kwargs": {"max_lr": 0.5e-5, "pct_start": 0.01, "anneal_strategy": 'cos', "div_factor": 15, "final_div_factor": 15}, #cascades 5e
        "scheduler_kwargs": {"max_lr": 0.35e-5, "pct_start": 0.01, "anneal_strategy": 'cos', "div_factor": 10, "final_div_factor": 10}, #cascades 6e, 7e, 8e
        #"scheduler_kwargs": {"max_lr": 1e-5, "pct_start": 0.01, "anneal_strategy": 'cos', "div_factor": 25, "final_div_factor": 25}, #track 1e
        #"scheduler_kwargs": {"max_lr": 0.5e-5, "pct_start": 0.01, "anneal_strategy": 'cos', "div_factor": 10, "final_div_factor": 10}, #track 2e
        "scheduler_config": {"frequency": 1, "monitor": "val_loss", "interval": "step"},
        "ckpt_path": False, # Continue training from a checkpoint
        "prefetch_factor": 2,
        "db_backend": "sqlite", # "sqlite" or "parquet"
        "event_type": "cascade", # "track" or "cascade
    }
    INFERENCE = True
    model_name = ["B_d32", "B_d64", "B_d32_4rel", "B+DynEdge_d64", "S+DynEdge_d32"]
    model_name = model_name[1]
    if not INFERENCE:
        config["retrain_from_checkpoint"] = config['archive'] + '/retrain_B_d64_7e_cascade_state_dict.pth'
    

    run_name = (
	f"retrain_{model_name}_8e_{config['event_type']}"
    )

    torch.multiprocessing.set_sharing_strategy("file_system")
    if INFERENCE and config["num_workers"] > 0:
        torch.multiprocessing.set_start_method("spawn") # When using num_workers>1, otherwise it will crash after one epoch. Not needed for OneCycleLR, since it is only 1 epoch
    

    if config["event_type"] == "cascade":
        all_databases, test_databases = [],[]
        test_selections, train_selections = [],[]
        if config["db_backend"] == "sqlite":
            # total files 34. 29 for training, 1 for validation and 4 for testing. No validation used for OneCycleLR
            sqlite_db_dir = '/scratch/users/allorana/merged_sqlite_1505/DNNCascadeL4_NuGen' 
            if not INFERENCE:
                for i in range(30): 
                    all_databases.append(f'{sqlite_db_dir}/DNNCascadeL4_NuGen_part{i}.db')
                    train_selections.append(None)
                val_selection = None
            else:
                for i in range(30,34):
                    all_databases.append(f'{sqlite_db_dir}/DNNCascadeL4_NuGen_part{i}.db')  
                    test_selections.append(pd.read_csv(f'{sqlite_db_dir}/selection_files/part{i}_n_pulses.csv'))

        elif config["db_backend"] == "parquet":
            all_selections = list(range(1000))
            all_databases = '/scratch/users/allorana/parquet_small/merged'
            train_val_selection = all_selections[:int(len(all_selections) * 0.9)]
            train_selections = train_val_selection[:-1]
            val_selection = train_val_selection[-1:]
        
            test_selection = all_selections[int(len(all_selections) * 0.9):]
    elif config["event_type"] == "track":
        train_selections = []
        sqlite_db_dir = '/scratch/users/allorana/northern_sqlite/old_files'
        if config["db_backend"] == 'sqlite':
            if not INFERENCE:
                all_databases = []
                for i in range(6):
                    all_databases.append(f'{sqlite_db_dir}/dev_northern_tracks_muon_labels_v3_part_{i+1}.db') 
                    train_selections.append(None)
                train_selections_last = pd.read_csv(f'{sqlite_db_dir}/selection_files/part_6.csv')
                train_selections_last = train_selections_last.loc[(train_selections_last["n_pulses"]), :][config["index_column"]].to_numpy()
                train_selections[-1] = train_selections_last[:int(len(train_selections_last) * 0.5)]
                val_selection = None
            else:
                all_databases = f'{sqlite_db_dir}/dev_northern_tracks_muon_labels_v3_part_6.db'
                test_selections = pd.read_csv(f'{sqlite_db_dir}/selection_files/part_6.csv')
                #test_selections = test_selections.loc[(test_selections["n_pulses"]), :][config["index_column"]].to_numpy()
                test_selections = [test_selections[:int(len(test_selections) * 0.5)]]
        elif config["db_backend"] == 'parquet':
            assert False, "parquet backend not implemented for tracks"
        else:
            assert False, "backend not recognized"                
    else:
        assert False, "event_type not recognized"
    
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
            backend=config["db_backend"],
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
        model = build_model(backbone=MODELS[model_name], config=config)
        if config["ckpt_path"]:
            config["fit"]["ckpt_path"] = config["ckpt_path"]

        if config["retrain_from_checkpoint"]:
            checkpoint_path = config["retrain_from_checkpoint"]
            print("Loading weights from ...", checkpoint_path)
            checkpoint = torch.load(checkpoint_path, torch.device("cpu"))
            if "state_dict" in checkpoint:
                checkpoint = checkpoint["state_dict"]
            new_checkpoint = rename_state_dict_keys(model, checkpoint)
            model.load_state_dict(new_checkpoint, strict=False)
            del checkpoint, new_checkpoint
        
        if validation_dataloader is not None:
            config["fit"]["val_dataloader"] = validation_dataloader

        model.fit(
            training_dataloader,
            callbacks=callbacks,
            **config["fit"],
        )
            
        model.save(os.path.join(config["archive"], f"{run_name}.pth"))
        model.save_state_dict(
            os.path.join(config["archive"], f"{run_name}_state_dict.pth")
        )
        model.save_config(os.path.join(config["archive"], f"{run_name}_config.yml"))
    else:
        for model_i in [model_name]:#["B_d32", "B_d64", "B_d32_4rel", "B+DynEdge_d64", "S+DynEdge_d32"]: # Possibility to loop over all models
            all_results = []
            checkpoint_path = config['archive'] + '/retrain_B_d32_4rel_2e_track_state_dict.pth'
            run_name_pred = f"{model_i}_2e_{config['event_type']}"
            test_databases = all_databases
            factor = 0.6 # Factor for the batch size. This was 0.6 is set for H100 100GB
            pulse_breakpoints =     [0, 100,   200, 300, 500,  1000, 2000, 3000, 10000000]
            batch_sizes_per_pulse = [4800, 2800,  700, 350,  90,   20,   15,   15]
        
            for min_pulse, max_pulse in zip(
                pulse_breakpoints[:-1], pulse_breakpoints[1:]
            ):
                print(
                    f"predicting {model_i} {min_pulse} to {max_pulse} pulses with batch size {int(factor*batch_sizes_per_pulse[pulse_breakpoints.index(max_pulse)-1])}"
                )
                pred_checkpoint_path = f"/scratch/users/allorana/prediction_cascades_icemix/{run_name_pred}_{min_pulse}to{max_pulse}pulses.pkl"
                if os.path.exists(pred_checkpoint_path):
                    results = pickle.load(open(pred_checkpoint_path, "rb")) # Skip if already predicted on this breakpoint
                else:
                    results = inference(
                        model_name = model_i,
                        device=config["fit"]["gpus"],
                        test_min_pulses=min_pulse,
                        test_max_pulses=max_pulse,
                        batch_size=max(int(
                            factor * batch_sizes_per_pulse[pulse_breakpoints.index(max_pulse) - 1]
                        ), 1),
                        checkpoint_path=checkpoint_path,
                        test_path=test_databases,
                        test_selection_file=test_selections,
                        config=config,
                    )
                    pickle.dump(results, open(pred_checkpoint_path, "wb")) # Save breakpoint results

                all_results.append(results)
                del results
                torch.cuda.empty_cache()

            results = pd.concat(all_results).sort_values(config["index_column"])
            
            path_to_save = f"/scratch/users/allorana/prediction_cascades_icemix" # Save the results in a folder
            results.to_csv(f"{path_to_save}/{run_name_pred}.csv")
            print(f"predicted and saved in {path_to_save}/{run_name_pred}.csv")
            del results
