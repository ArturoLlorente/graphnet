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


from graphnet.data.constants import FEATURES, TRUTH
from graphnet.models import StandardModel
from graphnet.models.graphs import KNNGraph
from graphnet.models.graphs.nodes import IceMixNodes
from graphnet.models.task.reconstruction import DirectionReconstructionWithKappa
from graphnet.training.labels import Direction, Direction_flipped
from graphnet.training.loss_functions import VonMisesFisher3DLoss
from graphnet.training.callbacks import ProgressBar
from graphnet.training.utils import make_dataloader
from graphnet.training.utils import collate_fn, collator_sequence_buckleting

from utils import make_dataloaders

from typing import Dict, Any, Callable
from graphnet.models.gnn import DeepIce

from graphnet.models.detector.detector import Detector
from graphnet.constants import ICECUBE_GEOMETRY_TABLE_DIR


class IceCube86(Detector):
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

MODELS = {'model1': DeepIce(hidden_dim=768, seq_length=192, depth=12, head_size=32, n_features=7),
        'model2': DeepIce(hidden_dim=768, seq_length=192, depth=12, head_size=64, n_features=7),
        'model3': DeepIce(hidden_dim=768, seq_length=192, depth=12, head_size=32, n_rel=4, n_features=7),
        'model4': DeepIce(hidden_dim=384, seq_length=128, depth=8, head_size=32, include_dynedge=True, scaled_emb=True, n_features=7, n_rel=1,     
                        #dynedge_args = {'nb_inputs': 9,
                        #                'nb_neighbours': 8,
                        #                'post_processing_layer_sizes': [336, 192],
                        #                'dynedge_layer_sizes': [(128, 256),(336, 256),(336, 256),(336, 256)],
                        #                'global_pooling_schemes': None,
                        #                'activation_layer': "gelu",
                        #                'add_norm_layer': True,
                        #                'skip_readout': True}
                        ),
        'model5': DeepIce(hidden_dim=768, seq_length=192, depth=12, head_size=64, include_dynedge=True, scaled_emb=True, n_features=7, n_rel=4)}

MODEL_PATH = {'model1': '/scratch/users/allorana/models_icemix/baselineV3_BE_globalrel_d32_0_6ema.pth',
              'model2': '/scratch/users/allorana/models_icemix/baselineV3_BE_globalrel_d64_0_3emaFT_2.pth',
              'model3': '/scratch/users/allorana/models_icemix/VFTV3_4RELFT_7.pth',
              'model4': '/scratch/users/allorana/models_icemix/V22FT6_1.pth',
              'model5': '/scratch/users/allorana/models_icemix/V23FT5_6.pth'}

MODEL_WEIGHTS = {'model1': 0.08254897,
                'model2': 0.15350807,
                'model3': 0.19367443,
                'model4': 0.23597202,
                'model5': 0.3342965}



def rename_state_dict_keys(model, checkpoint):
    i, passed_keys = 0, 0
    new_checkpoint = {}
    model_keys = list(model.state_dict().keys())
    checkpoint_keys = list(checkpoint.keys())
    
    if (model_keys == checkpoint_keys):
        new_checkpoint = checkpoint
    else:
        for m_k in model_keys:
            if m_k == '_tasks.0._affine.weight':
                v = checkpoint['proj_out.weight']
                new_checkpoint[m_k] = v
                
            elif m_k == '_tasks.0._affine.bias':
                v = checkpoint['proj_out.bias']
                new_checkpoint[m_k] = v
                
            else: 
                if checkpoint_keys[i] == 'proj_out.weight':#or checkpoint_keys[i] == 'proj_out.bias':  
                    passed_keys += 2
                v = checkpoint[checkpoint_keys[i+passed_keys]]
                new_checkpoint[m_k] = v
                i+=1  
    return new_checkpoint

def build_model(
    gnn_model,
    config: Dict[str, Any],
):

    backbone = gnn_model
    
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
        for i in range(1,6):
            config["model_name"] = f"model{i}"
            model = build_model(gnn_model=MODELS[f"model{i}"], config=config)
            model.eval()
            model.inference()
            checkpoint_path = MODEL_PATH[f"model{i}"]
            checkpoint = torch.load(checkpoint_path, torch.device("cpu"))
            if "state_dict" in checkpoint:
                checkpoint = checkpoint["state_dict"]
                
            new_checkpoint = rename_state_dict_keys(model, checkpoint) 
            model.load_state_dict(new_checkpoint)
            model.to(cuda_device)
            models.append(model)
            weights.append(MODEL_WEIGHTS[f"model{i}"])
    else:
        model = build_model(gnn_model=MODELS[model_name], config=config)
        #print(model)
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
            pred = (torch.stack([torch.nan_to_num(model(batch.to(cuda_device))[0]).clip(-1000, 1000) for model in models], -1).cpu() * weights).sum(-1)
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
    
    used_features = ["dom_x", "dom_y", "dom_z", "dom_time", "charge", "hlc", "rde"] # Added rde to the standard , change the ordering. Important the ordering when using SinusoidalPositionalEmbedding

    config = {
        "archive": "/scratch/users/allorana/icemix_cascades_retrain",
        "target": "direction",
        "batch_size": 12,
        "num_workers": 8,
        "num_database_files": 29+1,# total files 34. 29 for training, 1 for validation and 4 for testing
        "pulsemap": "InIceDSTPulses",
        "truth_table": "truth",
        "index_column": "event_no",
        "labels": {"direction": Direction_flipped()},
        "global_pooling_schemes": ["max"],
        "persistent_workers": True,
        "detector": IceCube86(),
        "node_definition": IceMixNodes(input_feature_names=used_features,
                                       max_pulses=3000,
                                       ),
        "nb_nearest_neighbours": 8,
        "features": used_features,
        "truth": ["energy", "energy_track", "position_x", "position_y", "position_z", "azimuth", "zenith", "pid", "elasticity", "interaction_type", "interaction_time"],
        "columns_nearest_neighbours": [0, 1, 2],
        "collate_fn": collator_sequence_buckleting([0.5,0.9]),
        "fit": {"max_epochs": 1, "gpus": [0], "precision": '16-mixed'},
        "optimizer_class": AdamW,
        "optimizer_kwargs": {"lr": 1e-5, "weight_decay": 0.05, "eps": 1e-7},
        "scheduler_class": OneCycleLR,
        "scheduler_kwargs": {"max_lr": 1e-5, "pct_start": 0.01, "anneal_strategy": 'cos', "div_factor": 25, "final_div_factor": 25},
        "scheduler_config": {"frequency": 1, "monitor": "val_loss", "interval": "step"},
        "wandb": False,
        "ema_decay": None,#0.9998,
        "swa_starting_epoch": None,#0
        "ckpt_path": False,
        "prefetch_factor": 2,
        "db_backend": "sqlite",
        "event_type": "track",
    }
    config["additional_attributes"] = [ "zenith", "azimuth", config["index_column"], "energy"]
    INFERENCE = True
    model_name = "model2"
    
    #torch.multiprocessing.set_start_method("spawn")
    
    if model_name == "model5":
        config["columns_nearest_neighbours"] = [0, 1, 2, 3]

    config['retrain_from_checkpoint'] = '/scratch/users/allorana/icemix_cascades_retrain/retrain_model2_3e_state_dict.pth'

    #if config["swa_starting_epoch"] is not None:
    #config["fit"]["distribution_strategy"] = 'ddp'
    #else:
    #    config["fit"]["distribution_strategy"] = 'ddp'

    run_name = (
#        f"{model_name}_cascades_retrain_IceMix_batch{config['batch_size']}_optimizer_AdamW_LR{config['scheduler_kwargs']['max_lr']}_annealStrat_{config['scheduler_kwargs']['anneal_strategy']}_"
#        f"ema_decay_{config['ema_decay']}_1epoch_14_05"
	f"retrain_{model_name}_4e_{config['event_type']}"
    )

    # Configurations
    torch.multiprocessing.set_sharing_strategy("file_system")
    

    if config["event_type"] == "cascade":
        all_databases, test_databases = [],[]
        test_selections, train_selections = [],[]
        if config["db_backend"] == "sqlite":
            sqlite_db_dir = '/scratch/users/allorana/merged_sqlite_1505'
            if not INFERENCE:
                for i in range(30):
                    all_databases.append(f'{sqlite_db_dir}/part{i}/merged/merged.db')
                    train_selections.append(None)
                val_selection = None
            else:
                for i in range(30,34):
                    all_databases.append(f'{sqlite_db_dir}/part{i}/merged/merged.db')  
                    test_selections.append(pd.read_csv(f'{sqlite_db_dir}/selection_files/part{i}_n_pulses.csv'))

        elif config["db_backend"] == "parquet":
            all_selections = list(range(1000))
            all_databases = '/scratch/users/allorana/parquet_really_small/merged'
            train_val_selection = all_selections[:int(len(all_selections) * 0.9)]
            train_selections = train_val_selection[:-1]
            val_selection = train_val_selection[-1:]
        
            test_selection = all_selections[int(len(all_selections) * 0.9):]
    elif config["event_type"] == "track":
        train_selections = []
        sqlite_db_dir = '/scratch/users/allorana/northern_sqlite'
        if config["db_backend"] == 'sqlite':
            if not INFERENCE:
                all_databases = []
                for i in range(3):
                    all_databases.append(f'{sqlite_db_dir}/dev_northern_tracks_full_part_{i+1}.db') 
                    train_selections.append(None)
                train_selections[-1] = pd.read_csv(f'{sqlite_db_dir}/selection_files/part3_n_pulses.csv')
                train_selections = train_selections.loc[(train_selections["n_pulses"]), :][config["index_column"]].to_numpy()
                train_selections = train_selections[:int(len(train_selections) * 0.6)]
                val_selection = None
            else:
                all_databases = f'{sqlite_db_dir}/dev_northern_tracks_full_part_4.db'
                test_selections = pd.read_csv(f'{sqlite_db_dir}/selection_files/part4_n_pulses.csv')
                #test_selections = test_selections.loc[(test_selections["n_pulses"]), :][config["index_column"]].to_numpy()
                test_selections = [test_selections[int(len(test_selections) * 0.6):]]
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
        model = build_model(gnn_model=MODELS[model_name], config=config)
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
        
        #save config into a file
        with open(os.path.join(config["archive"], f"{run_name}_config.pkl"), "wb") as f:
            pickle.dump(config, f)
            
        model.save(os.path.join(config["archive"], f"{run_name}.pth"))
        model.save_state_dict(
            os.path.join(config["archive"], f"{run_name}_state_dict.pth")
        )
        #print(f"Model saved to {config['archive']}/{run_name}.pth")
    else:
        #torch.multiprocessing.set_start_method("spawn")

        for model_i in ['model2']:#['model3', 'model4']:
            all_res = []
            #checkpoint_path = '/scratch/users/allorana/icemix_cascades_retrain/retrain_model2_2e_state_dict.pth'
            checkpoint_path = MODEL_PATH[model_i]
            run_name_pred = f"{model_i}_baseline_{config['event_type']}"
            test_databases = all_databases
            factor = 0.6
            pulse_breakpoints =     [0, 100,   200, 300, 500,  1000, 2000, 3000, 10000000]
            batch_sizes_per_pulse = [4800, 2800,  700, 350,  90,   20,   10,   15]
            config["num_workers"] = 6
            config["fit"]["gpus"] = [1]
        
            for min_pulse, max_pulse in zip(
                pulse_breakpoints[:-1], pulse_breakpoints[1:]
            ):
                print(
                    f"predicting {model_i} {min_pulse} to {max_pulse} pulses with batch size {int(factor*batch_sizes_per_pulse[pulse_breakpoints.index(max_pulse)-1])}"
                )
                pred_checkpoint_path = f"/scratch/users/allorana/prediction_cascades_icemix/{run_name_pred}_{min_pulse}to{max_pulse}pulses.pkl"
                if os.path.exists(pred_checkpoint_path):
                    results = pickle.load(open(pred_checkpoint_path, "rb"))
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
                    pickle.dump(results, open(pred_checkpoint_path, "wb"))

                all_res.append(results)
                del results
                torch.cuda.empty_cache()

            results = pd.concat(all_res).sort_values(config["index_column"])
            
            path_to_save = f"/scratch/users/allorana/prediction_cascades_icemix"
            results.to_csv(f"{path_to_save}/{run_name_pred}.csv")
            print(f"predicted and saved in {path_to_save}/{run_name_pred}.csv")
            del results
