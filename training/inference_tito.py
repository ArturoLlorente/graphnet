import os
import sys
import pandas as pd
from tqdm.auto import tqdm
import torch
import pickle
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    GradientAccumulationScheduler,
)

from pytorch_lightning.loggers import WandbLogger


from graphnet.data.constants import FEATURES, TRUTH
from graphnet.models import StandardModel
from graphnet.models.graphs import KNNGraph
from graphnet.models.graphs.nodes import NodesAsPulses
from graphnet.models.task.reconstruction import DirectionReconstructionWithKappa
from graphnet.training.labels import Direction, Direction_flipped
from graphnet.training.loss_functions import VonMisesFisher3DLoss, LossFunction
from graphnet.training.callbacks import ProgressBar
from graphnet.training.utils import make_dataloader
from graphnet.training.utils import collate_fn, collator_sequence_buckleting

from utils import make_dataloaders

from typing import Dict, Any, Callable
from graphnet.models.detector import Detector
from graphnet.models.gnn import DynEdgeTITO

MODEL_KWARGS = {"model1": {"dyntrans_layer_sizes":  [(256, 256), (256, 256), (256, 256), (256, 256)], "global_pooling_schemes": ["max"] , "use_global_features": True, "use_post_processing_layers": True},
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




class IceCube86TITO(Detector):
    """`Detector` class for IceCube-86."""

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
        return torch.log10(x)

    def _rde(self, x: torch.Tensor) -> torch.Tensor:
        return (x - 1.25) / 0.25

    def _pmt_area(self, x: torch.Tensor) -> torch.Tensor:
        return x / 0.05

    def _hlc(self, x: torch.Tensor) -> torch.Tensor:
        return torch.where(
            torch.eq(x, 0), torch.ones_like(x), torch.ones_like(x) * 0
        )

def rename_state_dict_keys(model, checkpoint):
    return checkpoint

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

    test_selection = (
        test_selection_file.loc[
            (test_selection_file["n_pulses"] <= test_max_pulses)
            & (test_selection_file["n_pulses"] > test_min_pulses),
            :,][config["index_column"]].to_numpy())
    
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
        model = build_model(gnn_model=  [model_name], config=config)
        #print(model)
        model.eval()
        model.inference()
        checkpoint_path = MODEL_PATH[model_name]
        checkpoint = torch.load(checkpoint_path, torch.device("cpu"))

        if "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]
        #temp_checkpoint = {('_tasks.0._affine.weight' if k == 'proj_out.weight' else '_tasks.0._affine.bias' if k == 'proj_out.bias' else 'backbone.' + k): v for k, v in checkpoint.items()}
        
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

    config = {
        "archive": "/remote/ceph/user/l/llorente/tito_cascade_retrain",
        "target": "direction",
        "batch_size": 28,
        "num_workers": 32,
        "pulsemap": "InIceDSTPulses",
        "truth_table": "truth",
        "index_column": "event_no",
        "labels": {"direction": Direction_flipped()},
        "num_database_files": 8,
        "persistent_workers": True,
        "detector": IceCube86(),
        "node_definition": IceMixNodes(),
        "nb_nearest_neighbours": 6,
        "features": ["dom_x", "dom_y", "dom_z", "dom_time", "charge", "hlc", "rde"],
        "truth": ["energy", "energy_track", "position_x", "position_y", "position_z", "azimuth", "zenith", "pid", "elasticity", "interaction_type", "interaction_time"],
        "columns_nearest_neighbours": [0, 1, 2],
        "collate_fn": collator_sequence_buckleting([0.8]),
        "prediction_columns": ["dir_y_pred", "dir_x_pred", "dir_z_pred", "dir_kappa_pred"],
        "fit": {"max_epochs": 1, "gpus": [0,1,2,3], "precision": '16-mixed'},
        "optimizer_class": AdamW,
        "optimizer_kwargs": {"lr": 1e-5, "weight_decay": 0.05, "eps": 1e-7},
        "scheduler_class": ReduceLROnPlateau,
        "scheduler_kwargs": {"max_lr": 1e-5, "pct_start": 0.01, "anneal_strategy": 'cos', "div_factor": 25, "final_div_factor": 25},
        "scheduler_config": {"frequency": 1, "monitor": "val_loss", "interval": "step"},
        "ckpt_path": False,
        #"model_name": 'model2',
    }
    config["additional_attributes"] = [ "zenith", "azimuth", config["index_column"], "energy"]
    INFERENCE = True
    model_name = "all"

    #config['retrain_from_checkpoint'] = MODEL_PATH[model_name]

    if config["swa_starting_epoch"] is not None:
        config["fit"]["distribution_strategy"] = 'ddp_find_unused_parameters_true'
    else:
        config["fit"]["distribution_strategy"] = 'ddp'

    run_name = (
        f"{model_name}_retrain_IceMix_batch{config['batch_size']}_optimizer_AdamW_LR{config['scheduler_kwargs']['max_lr']}_annealStrat_{config['scheduler_kwargs']['anneal_strategy']}_"
        f"ema_decay_{config['ema_decay']}_1epoch_11_02"
    )

    # Configurations
    torch.multiprocessing.set_sharing_strategy("file_system")
    #torch.multiprocessing.set_start_method("spawn")

    if os.path.isdir("/mnt/scratch/rasmus_orsoe/"):
        db_dir = "/mnt/scratch/rasmus_orsoe/databases/dev_northern_tracks_muon_labels_v3/"
        print("Using /mnt directory")
    else:
        db_dir = "/remote/ceph/user/l/llorente/dev_northern_tracks_muon_labels_v3/"
        print("Using /remote directory")
    
    #sel_dir = "/remote/ceph/user/l/llorente/northern_track_selection/"
    #all_databases, all_selections = [], []
    #test_idx = 5
    #for idx in range(1, config["num_database_files"] + 1):
    #    if idx == test_idx:
    #        test_database = (
    #            db_dir + f"dev_northern_tracks_muon_labels_v3_part_{idx}.db"
    #        )
    #        test_selection = pd.read_csv(sel_dir + f"part_{idx}.csv")
    #    else:
    #        all_databases.append(
    #            db_dir + f"dev_northern_tracks_muon_labels_v3_part_{idx}.db"
    #        )
    #        all_selections.append(pd.read_csv(sel_dir + f"part_{idx}.csv"))         
#
    #train_selections = []
    #for selection in all_selections:
    #    train_selections.append(selection.loc[selection["n_pulses"],:][config["index_column"]].ravel().tolist())
    #train_selections[-1] = train_selections[-1][:int(len(train_selections[-1]) * 0.9)]
    #
    #val_selection = (selection.loc[(selection["n_pulses"]), :][config["index_column"]].ravel().tolist())
    #val_selection = val_selection[int(len(val_selection) * 0.9) :]

    
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
        checkpoint_path = None#"/remote/ceph/user/l/llorente/icemix_northern_retrain/model5_retrain_IceMix_batch32_optimizer_AdamW_LR2e-05_annealStrat_cos_ema_decay_0.9998_2epoch_30_01_state_dict.pth"
        run_name_pred = f"test_all_reco"
        test_database = '/scratch/users/allorana/cascades_21537.db'
        
        factor = 0.8
        pulse_breakpoints = [0, 100, 200, 300]#, 500, 10000000]
        batch_sizes_per_pulse = [3600, 2800, 700]#, 400, 240]
        config["num_workers"] = 32
        
        test_selection = pd.read_csv('/scratch/users/allorana/cascades_21537_selection.csv')

        for min_pulse, max_pulse in zip(
            pulse_breakpoints[:-1], pulse_breakpoints[1:]
        ):
            print(
                f"predicting {min_pulse} to {max_pulse} pulses with batch size {int(factor*batch_sizes_per_pulse[pulse_breakpoints.index(max_pulse)-1])}"
            )
            pred_checkpoint_path = f"/scratch/users/allorana/prediction_cascades_icemix/{run_name_pred}_{min_pulse}to{max_pulse}pulses.pkl"
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
        
        path_to_save = f"/scratch/users/allorana/prediction_cascades_icemix"
        results.to_csv(f"{path_to_save}/{run_name_pred}.csv")
        print(f"predicted and saved in {path_to_save}/{run_name_pred}.csv")
