import os
import sys
import pandas as pd
from tqdm.auto import tqdm
import torch
import pickle
from typing import Dict, Any, Callable

from torch.optim import Adam

from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    GradientAccumulationScheduler,
    LearningRateMonitor,
    EarlyStopping,
)
from pytorch_lightning.loggers import WandbLogger


from graphnet.data.constants import FEATURES, TRUTH
from graphnet.models import StandardModel, StandardModelPred
from graphnet.models.graphs import KNNGraph
from graphnet.models.graphs.nodes import NodesAsPulses
from graphnet.models.task.reconstruction import DirectionReconstructionWithKappa
from graphnet.training.labels import Direction, Direction_flipped
from graphnet.training.loss_functions import VonMisesFisher3DLoss, LossFunction
from graphnet.training.callbacks import ProgressBar, PiecewiseLinearLR
from graphnet.training.utils import make_dataloader
from graphnet.training.utils import collate_fn, collator_sequence_buckleting
from graphnet.models.detector import Detector
from graphnet.models.task import Task
from graphnet.models.gnn import DynEdgeTITO
from graphnet.constants import ICECUBE_GEOMETRY_TABLE_DIR


from utils import make_dataloaders, rename_state_dict_keys


GNN_KWARGS = {"model1": {"dyntrans_layer_sizes":  [(256, 256), (256, 256), (256, 256), (256, 256)], "global_pooling_schemes": ["max"] , "use_global_features": True, "use_post_processing_layers": True},
                "model2": {"dyntrans_layer_sizes":  [(256, 256), (256, 256), (256, 256), (256, 256)], "global_pooling_schemes": ["max"] , "use_global_features": True, "use_post_processing_layers": True},
                "model3": {"dyntrans_layer_sizes":  [(256, 256), (256, 256), (256, 256)], "global_pooling_schemes": ["max"] , "use_global_features": False, "use_post_processing_layers": False},
                "model4": {"dyntrans_layer_sizes":  [(256, 256), (256, 256), (256, 256)], "global_pooling_schemes": ["max"] , "use_global_features": True, "use_post_processing_layers": True},
                "model5": {"dyntrans_layer_sizes":  [(256, 256), (256, 256), (256, 256), (256, 256)], "global_pooling_schemes": ["max"] , "use_global_features": True, "use_post_processing_layers": True},
                "model6": {"dyntrans_layer_sizes":  [(256, 256), (256, 256), (256, 256), (256, 256)], "global_pooling_schemes": ["max"] , "use_global_features": True, "use_post_processing_layers": True}}

columns_nearest_neighbours_all = {
    "model1": [0, 1, 2],
    "model2": [0, 1, 2],
    "model3": [0, 1, 2],
    "model4": [0, 1, 2, 3],
    "model5": [0, 1, 2],
    "model6": [0, 1, 2, 3],
}

model_dir = '/scratch/users/allorana/models_tito/'
MODEL_WEIGHTS = {"model1": model_dir + 'model1-last.pth',
              "model2": model_dir + 'model2-last.pth',
              "model3": model_dir + 'model3-last.pth',
              "model4": model_dir + 'model4-last.pth',
              "model5": model_dir + 'model5-last.pth',
              "model6": model_dir + 'model6-last.pth'}

class IceCube86TITO(Detector):
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
            "hlc": self._hlc,
        }
        return feature_map

    def _dom_xyz(self, x: torch.Tensor) -> torch.Tensor:
        return x / 500.0

    def _dom_time(self, x: torch.Tensor) -> torch.Tensor:
        return (x - 1.0e04) / (500.0 * 0.23)

    def _charge(self, x: torch.Tensor) -> torch.Tensor:
        return torch.log10(x) / 3.0

    def _rde(self, x: torch.Tensor) -> torch.Tensor:
        return (x - 1.25) / 0.25

    def _pmt_area(self, x: torch.Tensor) -> torch.Tensor:
        return x / 0.05

    def _hlc(self, x: torch.Tensor) -> torch.Tensor:
        return torch.where(
            torch.eq(x, 0), torch.ones_like(x), torch.ones_like(x) * 0
        )

class DirectionReconstructionWithKappaTITO(Task):
    default_target_labels = ["direction"]
    default_prediction_labels = [
        "dir_x_pred",
        "dir_y_pred",
        "dir_z_pred",
        "dir_kappa_pred",
    ]
    nb_inputs = 3

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        kappa = torch.linalg.vector_norm(x, dim=1)
        kappa = torch.clamp(kappa, min=torch.finfo(x.dtype).eps)
        vec_x = x[:, 0] / kappa
        vec_y = x[:, 1] / kappa
        vec_z = x[:, 2] / kappa
        return torch.stack((vec_x, vec_y, vec_z, kappa), dim=1)
    
class DistanceLoss2(LossFunction):
    def _forward(
        self, prediction: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        target = target.reshape(-1, 3)
        assert prediction.dim() == 2 and prediction.size()[1] == 4
        assert target.dim() == 2
        assert prediction.size()[0] == target.size()[0]
        eps = 1e-4
        prediction_length = torch.linalg.vector_norm(
            prediction[:, [0, 1, 2]], dim=1
        )
        prediction_length = torch.clamp(prediction_length, min=eps)
        prediction = prediction[:, [0, 1, 2]] / prediction_length.unsqueeze(1)
        cosLoss = (
            prediction[:, 0] * target[:, 0]
            + prediction[:, 1] * target[:, 1]
            + prediction[:, 2] * target[:, 2]
        )
        cosLoss = torch.clamp(cosLoss, min=-1 + eps, max=1 - eps)
        thetaLoss = torch.arccos(cosLoss)
        return thetaLoss
    
    
def build_model(
    model_name: Dict[str, Any],
    config: Dict[str, Any],
):

    backbone = DynEdgeTITO(
            nb_inputs=config['graph_definition'].nb_outputs,
            dyntrans_layer_sizes=GNN_KWARGS[model_name]['dyntrans_layer_sizes'],
            global_pooling_schemes=GNN_KWARGS[model_name]['global_pooling_schemes'],
            use_global_features=GNN_KWARGS[model_name]['use_global_features'],
            use_post_processing_layers=GNN_KWARGS[model_name]['use_post_processing_layers'],
            )   
     
    task = DirectionReconstructionWithKappa(
            hidden_size=backbone.nb_outputs,
            target_labels="direction",
            loss_function=VonMisesFisher3DLoss(),
        )
    task2 = DirectionReconstructionWithKappa(
        hidden_size=backbone.nb_outputs,
        target_labels="direction",
        loss_function=VonMisesFisher3DLoss(),
    )

    model_kwargs = {
        "graph_definition": config["graph_definition"],
        "backbone": backbone,
        "tasks": [task,task2],
        "optimizer_class": config["optimizer_class"],
        "optimizer_kwargs": config["optimizer_kwargs"],
        "scheduler_class": config["scheduler_class"],
        "scheduler_kwargs": config["scheduler_kwargs"],
        "scheduler_config": config["scheduler_config"],
    }
    if USE_ALL_FEATURES_IN_PREDICTION:
        model = StandardModelPred(**model_kwargs)
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
    
    
    if isinstance(test_selection_file, pd.DataFrame):
        test_selection_file = [test_selection_file]
          
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
        AssertionError("Not implemented")
    else:
        model = build_model(model_name=model_name, config=config)
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
    
    event_nos ,zenith ,azimuth ,preds = [], [], [], []
    print('start predict')
    validateMode=True
    with torch.no_grad():
        model.to(f'cuda:{device[0]}')
        for batch in tqdm(test_dataloader):
            pred = model(batch.to(f'cuda:{device[0]}'))
            
            if USE_ALL_FEATURES_IN_PREDICTION:
                preds.append(torch.cat(pred, axis=-1))
            else:
                preds.append(pred[0])
            event_nos.append(batch.event_no)
            if validateMode:
                zenith.append(batch.zenith)
                azimuth.append(batch.azimuth)
    preds = torch.cat(preds).to("cpu").detach().numpy()
    
    
    if USE_ALL_FEATURES_IN_PREDICTION:
        if preds.shape[1] == 128+8:
            columns = ['direction_x','direction_y','direction_z','direction_kappa',
                       'direction_x1','direction_y1','direction_z1','direction_kappa1'] + [f'idx{i}' for i in range(128)]
        else:
            columns = ['direction_x','direction_y','direction_z','direction_kappa'] + [f'idx{i}' for i in range(128)]
    else:
        columns = ['direction_x','direction_y','direction_z','direction_kappa']


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
        "archive": "/scratch/users/allorana/tito_cascades_retrain",
        "target": "direction",
        "batch_size": 28,
        "num_workers": 4,
        "pulsemap": "InIceDSTPulses",
        "truth_table": "truth",
        "index_column": "event_no",
        "labels": {"direction": Direction()},
        "accumulate_grad_batches": {0: 2},
        "persistent_workers": True,
        "detector": IceCube86TITO(),
        "node_definition": NodesAsPulses(),
        "nb_nearest_neighbours": 6,
        "features": ["dom_x", "dom_y", "dom_z", "dom_time", "charge", "hlc"],
        "truth": TRUTH.ICECUBE86,
        "columns_nearest_neighbours": [0, 1, 2],
        "collate_fn": collator_sequence_buckleting([0.8]),
        "prediction_columns": ["dir_x_pred", "dir_y_pred", "dir_z_pred", "dir_kappa_pred"],
        "fit": {"max_epochs": 100, "gpus": [1], "precision": '16-mixed'},
        "optimizer_class": Adam,
        "optimizer_kwargs": {"lr": 1e-5, "weight_decay": 0.05, "eps": 1e-7},
        "scheduler_class": PiecewiseLinearLR,
        "scheduler_config": {"interval": "step"},
        "ckpt_path": False,
        "prefetch_factor": 2,
        "db_backend": "sqlite", # "sqlite" or "parquet"
        "event_type": "track", # "track" or "cascade
        }
    config["additional_attributes"] = [ "zenith", "azimuth", config["index_column"], "energy"]
    INFERENCE = True
    model_name = "model1"
    USE_ALL_FEATURES_IN_PREDICTION = True

    config['retrain_from_checkpoint'] = MODEL_WEIGHTS[model_name]

    if len(config["fit"]["gpus"]) > 1:
        config["fit"]["distribution_strategy"] = 'ddp'

    run_name = (
        #f"{model_name}_retrain_IceMix_batch{config['batch_size']}_optimizer_Adam_LR{config['scheduler_kwargs']['max_lr']}_annealStrat_{config['scheduler_kwargs']['anneal_strategy']}_"
        #f"ema_decay_{config['ema_decay']}_1epoch_11_02"
        f"test_tito_{model_name}"
    )

    # Configurations
    torch.multiprocessing.set_sharing_strategy("file_system")
    if INFERENCE:
        torch.multiprocessing.set_start_method("spawn")


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
            config=config,
        )

        config['scheduler_kwargs'] = {
                        "milestones": [
                            0,
                            len(training_dataloader)//(len(config['fit']['gpus'])*config['accumulate_grad_batches'][0]*30),
                            len(training_dataloader)*config['fit']['max_epochs']//(len(config['fit']['gpus'])*config['accumulate_grad_batches'][0]*2),
                            len(training_dataloader)*config['fit']['max_epochs']//(len(config['fit']['gpus'])*config['accumulate_grad_batches'][0]),                
                        ],
                        "factors": [1e-03, 1, 1, 1e-03],
                        "verbose": False,
        }

        callbacks = [
            ProgressBar(),
            GradientAccumulationScheduler(scheduling=config['accumulate_grad_batches']),
            LearningRateMonitor(logging_interval='step'),
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
        model = build_model(model_name=model_name, config=config)
        if config["ckpt_path"]:
            config["fit"]["ckpt_path"] = config["ckpt_path"]

        if config["retrain_from_checkpoint"]:
            checkpoint_path = config["retrain_from_checkpoint"]
            print("Loading weights from ...", checkpoint_path)
            checkpoint = torch.load(checkpoint_path, torch.device("cpu"))
            if "state_dict" in checkpoint:
                checkpoint = checkpoint["state_dict"]
            #new_checkpoint = {('_tasks.0._affine.weight' if k == 'proj_out.weight' else '_tasks.0._affine.bias' if k == 'proj_out.bias' else '_gnn.' + k): v for k, v in checkpoint.items()}
            model.load_state_dict(checkpoint, strict=False)
            del checkpoint
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
        for model_name in MODEL_WEIGHTS.keys():
            all_res = []
            checkpoint_path = MODEL_WEIGHTS[model_name]
            run_name_pred = f"{model_name}_baseline_{config['event_type']}"
            test_databases = all_databases

            factor = 0.6
            pulse_breakpoints = [0, 100, 200, 300, 500, 1000, 5000, 10000]
            batch_sizes_per_pulse = [4800, 2800, 700, 400, 240, 120, 60, 30]

        
            for min_pulse, max_pulse in zip(
                pulse_breakpoints[:-1], pulse_breakpoints[1:]
            ):
                print(
                    f"predicting {min_pulse} to {max_pulse} pulses with batch size {int(factor*batch_sizes_per_pulse[pulse_breakpoints.index(max_pulse)-1])}"
                )
                pred_checkpoint_path = f"/scratch/users/allorana/prediction_cascades_tito/{run_name_pred}_{min_pulse}to{max_pulse}pulses.pkl"
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
                        test_path=test_databases,
                        test_selection_file=test_selections,
                        config=config,
                    )
                    pickle.dump(results, open(pred_checkpoint_path, "wb"))

                all_res.append(results)
                del results
                torch.cuda.empty_cache()

            results = pd.concat(all_res).sort_values(config["index_column"])
            
            path_to_save = f"/scratch/users/allorana/prediction_cascades_tito"
            results.to_csv(f"{path_to_save}/{run_name_pred}.csv")
            print(f"predicted and saved in {path_to_save}/{run_name_pred}.csv")
