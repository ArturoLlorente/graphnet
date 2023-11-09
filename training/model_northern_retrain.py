import os
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import sys

import torch
from torch.optim.adam import Adam

from pytorch_lightning.callbacks import ModelCheckpoint, GradientAccumulationScheduler, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger 
from pytorch_lightning.callbacks import EarlyStopping

from graphnet.data.dataset import EnsembleDataset
from graphnet.data.dataset import SQLiteDataset
from graphnet.data.constants import FEATURES, TRUTH

from graphnet.models import StandardModel, StandardModelPred
from graphnet.models.graphs import KNNGraph
from graphnet.models.gnn import DynEdgeTITO
from graphnet.models.graphs.nodes import NodesAsPulses
from graphnet.models.task.reconstruction import DirectionReconstructionWithKappa

from graphnet.training.labels import Direction
from graphnet.training.loss_functions import VonMisesFisher3DLoss, LossFunction
from graphnet.training.callbacks import ProgressBar, PiecewiseLinearLR
from graphnet.training.utils import make_dataloader
from graphnet.training.utils import collate_fn, collator_sequence_buckleting

from typing import Dict, List, Optional, Union, Callable, Any
from torch.utils.data import DataLoader


from graphnet.models.detector.detector import Detector
from graphnet.models.task import Task

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
        return (x - 1.0e04) / (500.0*0.23)

    def _charge(self, x: torch.Tensor) -> torch.Tensor:
        return torch.log10(x)

    def _rde(self, x: torch.Tensor) -> torch.Tensor:
        return (x - 1.25) / 0.25

    def _pmt_area(self, x: torch.Tensor) -> torch.Tensor:
        return x / 0.05
    
    def _hlc(self, x: torch.Tensor) -> torch.Tensor:
        return torch.where(torch.eq(x, 0), torch.ones_like(x), torch.ones_like(x)*0)
class DirectionReconstructionWithKappaTITO(Task):
    default_target_labels = ["direction"]
    default_prediction_labels = ["dir_x_pred","dir_y_pred","dir_z_pred","dir_kappa_pred"]
    nb_inputs = 3
    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        kappa = torch.linalg.vector_norm(x, dim=1)
        kappa = torch.clamp(kappa, min=torch.finfo(x.dtype).eps)
        vec_x = x[:, 0] / kappa
        vec_y = x[:, 1] / kappa
        vec_z = x[:, 2] / kappa
        return torch.stack((vec_x, vec_y, vec_z, kappa), dim=1)
class DistanceLoss2(LossFunction):
    def _forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target.reshape(-1, 3)
        assert prediction.dim() == 2 and prediction.size()[1] == 4
        assert target.dim() == 2
        assert prediction.size()[0] == target.size()[0]
        eps = 1e-4
        prediction_length = torch.linalg.vector_norm(prediction[:, [0, 1, 2]], dim=1)
        prediction_length = torch.clamp(prediction_length, min=eps)
        prediction =  prediction[:, [0, 1, 2]]/prediction_length.unsqueeze(1)
        cosLoss = prediction[:, 0] * target[:, 0] + prediction[:, 1] * target[:, 1] + prediction[:, 2] * target[:, 2]    
        cosLoss = torch.clamp(cosLoss, min=-1+eps, max=1-eps)
        thetaLoss = torch.arccos(cosLoss)
        return thetaLoss
    
from torch_geometric.data import Data, Batch

class DynamicCollator:
    def __init__(self, max_vram: int, max_n_pulses: int):
        self.max_vram = max_vram
        self.max_n_pulses = max_n_pulses

    def __call__(self, graphs: List[Data]) -> List[Batch]:
        # Filter out graphs with n_pulses <= 1 and sort by n_pulses
        graphs = [g for g in graphs if g.n_pulses > 1]
        graphs.sort(key=lambda x: x.n_pulses)
        
        batch_list = []
        current_batch = []
        current_batch_size = 0

        for graph in graphs:
            # Calculate the expected memory usage for the current graph
            graph_memory = self.calculate_graph_memory_usage(graph)
            
            # If adding the current graph to the batch exceeds the VRAM limit or
            # the batch size exceeds your desired limit based on n_pulses, create a new batch.
            if current_batch_size + graph_memory > self.max_vram or len(current_batch) >= self.max_batch_size_for_n_pulses(graph.n_pulses):
                if len(current_batch) > 0:
                    batch = Batch.from_data_list(current_batch)
                    batch_list.append(batch)
                current_batch = []
                current_batch_size = 0
                
            current_batch.append(graph)
            current_batch_size += graph_memory

        # Process the last batch
        if len(current_batch) > 0:
            batch = Batch.from_data_list(current_batch)
            batch_list.append(batch)

        return batch_list
class DynamicCollator:
    def __init__(self, max_vram: int):
        self.max_vram = max_vram

    def __call__(self, graphs: List[Data]) -> List[Batch]:
        # Filter out graphs with n_pulses <= 1
        graphs = [g for g in graphs if g.n_pulses > 1]
        graphs.sort(key=lambda x: x.n_pulses)
        batch_list = []
        current_batch = []

        for graph in graphs:
            # Calculate VRAM usage for the current batch
            current_batch.append(graph)
            current_batch_size = len(current_batch) * graph.num_nodes  # You can adjust this based on your actual VRAM constraints
            print("current batch size: ", current_batch_size)

            if current_batch_size > self.max_vram:
                if len(current_batch) > 0:
                    batch = Batch.from_data_list(current_batch)
                    batch_list.append(batch)
                current_batch = [graph]
        
        # Process the last batch
        if len(current_batch) > 0:
            batch = Batch.from_data_list(current_batch)
            batch_list.append(batch)

        return batch_list


use_global_features_all = {'model1': True, 'model2': True, 'model3': False, 'model4': True, 'model5': True, 'model6': True}
use_post_processing_layers_all = {'model1': True, 'model2': True, 'model3': False, 'model4': True, 'model5': True, 'model6': True}
dyntrans_layer_sizes_all = {'model1': [(256, 256), (256, 256), (256, 256), (256, 256)],
                        'model2': [(256, 256), (256, 256), (256, 256), (256, 256)],
                        'model3': [(256, 256), (256, 256), (256, 256)],
                        'model4': [(256, 256), (256, 256), (256, 256)],
                        'model5': [(256, 256), (256, 256), (256, 256), (256, 256)],
                        'model6': [(256, 256), (256, 256), (256, 256), (256, 256)]}
columns_nearest_neighbours_all = {'model1': [0, 1, 2], 'model2': [0, 1, 2], 'model3': [0, 1, 2], 'model4': [0, 1, 2, 3], 'model5': [0, 1, 2], 'model6': [0, 1, 2, 3]}
    
def make_dataloaders(
    db: Union[List[str], str],
    train_selection: Optional[List[int]],
    val_selection: Optional[List[int]],
    config: Dict[str, Any],
) -> DataLoader:
    
    """Construct `DataLoader` instance."""
    # Check(s)
    if isinstance(config['pulsemap'], str):
        config['pulsemap'] = [config['pulsemap']]
    assert isinstance(train_selection, list)
    if isinstance(db, list):
        assert len(train_selection) == len(db)
        train_datasets = []
        pbar = tqdm(total=len(db))
        for db_idx, database in enumerate(db):

            train_datasets.append(SQLiteDataset(
                path=database,
                graph_definition=config['graph_definition'],
                pulsemaps=config['pulsemap'],
                features=config['features'],
                truth=config['truth'],
                selection=train_selection[db_idx],
                node_truth=config['node_truth'],
                truth_table=config['truth_table'],
                node_truth_table=config['node_truth_table'],
                string_selection=config['string_selection'],
                loss_weight_table=config['loss_weight_table'],
                loss_weight_column=config['loss_weight_column'],
                index_column=config['index_column'],
            ))
            pbar.update(1)

        if isinstance(config['labels'], dict):
            for label in config['labels'].keys():
                for train_dataset in train_datasets:
                    train_dataset.add_label(key=label, fn=config['labels'][label])
                    
        train_dataset = EnsembleDataset(train_datasets)

        training_dataloader = DataLoader(
            dataset=train_dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=config['num_workers'],
            collate_fn=config['collate_fn'],
            persistent_workers=config['persistent_workers'],
            prefetch_factor=2
        )

        if val_selection is not None:
            val_dataset = SQLiteDataset(
                            path = db[-1],
                            graph_definition=config['graph_definition'],
                            pulsemaps=config['pulsemap'],
                            features=config['features'],
                            truth=config['truth'],
                            selection=val_selection,
                            node_truth=config['node_truth'],
                            truth_table=config['truth_table'],
                            node_truth_table=config['node_truth_table'],
                            string_selection=config['string_selection'],
                            loss_weight_table=config['loss_weight_table'],
                            loss_weight_column=config['loss_weight_column'],
                            index_column=config['index_column'],
            )
            
            if isinstance(config['labels'], dict):
                for label in config['labels'].keys():
                    val_dataset.add_label(key=label, fn=config['labels'][label])
            
            validation_dataloader = DataLoader(
                dataset=val_dataset,
                batch_size=config['batch_size'],
                shuffle=False,
                num_workers=config['num_workers'],
                collate_fn=config['collate_fn'],
                persistent_workers=config['persistent_workers'],
                prefetch_factor=2,
            )
        else:
            validation_dataloader = None
            val_dataset = None
        
                    
    return training_dataloader, validation_dataloader, train_dataset, val_dataset


def build_model(
    config: Dict[str, Any],
    ):

    gnn = DynEdgeTITO(
            nb_inputs=config['graph_definition'].nb_outputs,
            dyntrans_layer_sizes=config['dyntrans_layer_sizes'],
            global_pooling_schemes=config['global_pooling_schemes'],
            use_global_features=config['use_global_features'],
            use_post_processing_layers=config['use_post_processing_layers'],
            )
    
    task = DirectionReconstructionWithKappaTITO(
            hidden_size=gnn.nb_outputs,
            target_labels="direction",
            loss_function=DistanceLoss2(),
        )
    task2 = DirectionReconstructionWithKappa(
        hidden_size=gnn.nb_outputs,
        target_labels="direction",
        loss_function=VonMisesFisher3DLoss(),
    )

    if not INFERENCE:
        model = StandardModel(
                graph_definition=config['graph_definition'],
                gnn=gnn,
                tasks=[task, task2],
                optimizer_class=Adam,
                optimizer_kwargs={'lr': 1e-03, 'eps': 1e-03},
                scheduler_class= config['scheduler_class'],
                scheduler_kwargs=config['scheduler_kwargs'],
                scheduler_config={'interval': 'step'},
            )
    else:
        model = StandardModelPred(
                graph_definition=config['graph_definition'],
                gnn=gnn,
                tasks=[task, task2],
                optimizer_class=Adam,
                optimizer_kwargs={'lr': 1e-03, 'eps': 1e-03},
                scheduler_class= config['scheduler_class'],
                scheduler_kwargs=config['scheduler_kwargs'],
                scheduler_config={'interval': 'step'},
            )

    model.prediction_columns = config['prediction_columns']
    model.additional_attributes = config['additional_attributes']
    
    return model


def inference(device: int,
              checkpoint_path: str,
              test_min_pulses: int,
              test_max_pulses: int,
              batch_size: int,
              use_all_features_in_prediction: bool = True,
              test_path: str = None,
              test_selection_file: str = None,
              config: Dict[str, Any] = None,
            ):

    test_selection = test_selection_file.loc[(test_selection_file['n_pulses']<=test_max_pulses) & 
                                             (test_selection_file['n_pulses']>test_min_pulses),:][config['index_column']].ravel().tolist()

    test_dataloader =  make_dataloader(db = test_path,
                                        selection = test_selection,
                                        graph_definition = config['graph_definition'],
                                        pulsemaps = config['pulsemap'],
                                        num_workers = config['num_workers'],
                                        features = config['features'],
                                        shuffle = False,
                                        truth = config['truth'],
                                        batch_size = batch_size,
                                        truth_table = config['truth_table'],
                                        index_column=config['index_column'],
                                        labels = config['labels'],
                                        )
    

    model = build_model(config)
    model.eval()
    model.inference()
    checkpoint = torch.load(checkpoint_path, torch.device('cpu'))
    
    if 'state_dict' in checkpoint:
        checkpoint = checkpoint['state_dict']
    model.load_state_dict(checkpoint)


    event_nos ,zenith ,azimuth ,preds = [], [], [], []
    print('start predict')
    validateMode=True
    with torch.no_grad():
        model.to(f'cuda:{device[0]}')
        for batch in tqdm(test_dataloader):
            pred = model(batch.to(f'cuda:{device[0]}'))
            
            if use_all_features_in_prediction:
                preds.append(torch.cat(pred, axis=-1))
            else:
                preds.append(pred[0])
            event_nos.append(batch.event_no)
            if validateMode:
                zenith.append(batch.zenith)
                azimuth.append(batch.azimuth)
    preds = torch.cat(preds).to('cpu').detach().numpy()


    if use_all_features_in_prediction:
        if preds.shape[1] == 128+8:
            columns = ['direction_x','direction_y','direction_z','direction_kappa',
                       'direction_x1','direction_y1','direction_z1','direction_kappa1'] + [f'idx{i}' for i in range(128)]
        else:
            columns = ['direction_x','direction_y','direction_z','direction_kappa'] + [f'idx{i}' for i in range(128)]
    else:
        columns = ['direction_x','direction_y','direction_z','direction_kappa']
        
    results = pd.DataFrame(preds, columns=columns)
    results[config['index_column']] = torch.cat(event_nos).to('cpu').detach().numpy()
    if validateMode:
        results['zenith'] = torch.cat(zenith).to('cpu').numpy()
        results['azimuth'] = torch.cat(azimuth).to('cpu').numpy()

    return results
# Main function call
if __name__ == "__main__":

    config = {
        "archive": "/remote/ceph/user/l/llorente/train_DynEdgeTITO_northern_Oct23",
        "target": 'direction',
        "weight_column_name": None,
        "weight_table_name": None,
        "batch_size": 2,
        "early_stopping_patience": 20,
        "num_workers": 64,
        "pulsemap": 'InIceDSTPulses',
        "truth_table": 'truth',
        "index_column": 'event_no',
        "labels": {'direction': Direction()},
        "global_pooling_schemes": ["max"],
        "accumulate_grad_batches": {0: 1},
        "train_max_pulses": [12990, 13000],
        "val_max_pulses": 500,
        "num_database_files": 1,
        "node_truth_table": None,
        "node_truth": None,
        "string_selection": None,
        "loss_weight_table": None,
        "loss_weight_column": None,
        "persistent_workers": True,
        "detector": IceCube86TITO(),
        "node_definition": NodesAsPulses(),
        "nb_nearest_neighbours": 6,
        "features": ['dom_x', 'dom_y', 'dom_z', 'dom_time', 'charge', 'hlc'],
        "truth": TRUTH.ICECUBE86,
        "columns_nearest_neighbours": [0, 1, 2],
        "collate_fn": collator_sequence_buckleting([1/2]),
        "prediction_columns": ['dir_x_pred', 'dir_y_pred', 'dir_z_pred', 'dir_kappa_pred'],
        "fit": {
            "max_epochs": 20,
            "gpus": [2,3],
            "check_val_every_n_epoch": 1,
            "precision": '16-mixed',
        },
        "scheduler_class": PiecewiseLinearLR,
        "wandb": False,
        "resume_training_path": None,
    }
    
    MODEL = 'model5'
    INFERENCE = False


    #ICECUBE86 = ["dom_x","dom_y","dom_z","dom_time","charge","hlc"]
    #KAGGLE = ["x", "y", "z", "time", "charge", "auxiliary"]

    config["use_global_features"] = use_global_features_all[MODEL]
    config["use_post_processing_layers"] = use_post_processing_layers_all[MODEL]
    config["dyntrans_layer_sizes"] = dyntrans_layer_sizes_all[MODEL]
    config["columns_nearest_neighbours"] = columns_nearest_neighbours_all[MODEL]
    config['additional_attributes'] = ['zenith', 'azimuth', config['index_column'], 'energy']
    if len(config['fit']['gpus']) > 1:
        config['fit']['distribution_strategy'] = 'ddp'

    #run_name = (f"{MODEL}_NEWTEST_dynedgeTITO_directionReco_{config['fit']['max_epochs']}e_trainMaxPulses{config['train_max_pulses'][0]}_"
    #            f"trainMinPulses{config['train_max_pulses'][1]}_valMaxPulses{config['val_max_pulses']}_layerSize{len(config['dyntrans_layer_sizes'])}"
    #            f"_useGG{config['use_global_features']}_usePP{config['use_post_processing_layers']}_batch{config['batch_size']}"
    #            f"_numDatabaseFiles{config['num_database_files']}")
    run_name = "dummy"

    # Configurations
    torch.multiprocessing.set_sharing_strategy('file_system')
    #torch.multiprocessing.set_start_method('spawn', force=True)
    torch.set_float32_matmul_precision('high')
    
    db_dir = '/mnt/scratch/rasmus_orsoe/databases/dev_northern_tracks_muon_labels_v3/'
    all_databases, all_selections = [], []
    #test_idx = 5
    #for idx in range(1,9):
    #    if idx == test_idx: 
    #        test_database = db_dir + f'dev_northern_tracks_muon_labels_v3_part_{idx}.db'
    #        test_selection = pd.read_csv(f'/remote/ceph/user/l/llorente/northern_track_selection/part_{idx}.csv')
    #    else:
    #        all_databases.append(db_dir + f'dev_northern_tracks_muon_labels_v3_part_{idx}.db')
    #        all_selections.append(pd.read_csv(f'/remote/ceph/user/l/llorente/northern_track_selection/part_{idx}.csv'))
    ## get_list_of_databases:
    #train_selections = []
    #for selection in all_selections:
    #    train_selections.append(selection.loc[selection['n_pulses'] < config['train_max_pulses'],:][config['index_column']].ravel().tolist())
    #train_selections[-1] = train_selections[-1][:int(len(train_selections[-1])*0.9)]
    #val_selection = selection.loc[(selection['n_pulses'] < config['val_max_pulses']),:][config['index_column']].ravel().tolist()
    #val_selection = val_selection[int(len(val_selection)*0.9):]
    
    all_databases.append(db_dir + f'dev_northern_tracks_muon_labels_v3_part_1.db')
    all_selections = pd.read_csv(f'/remote/ceph/user/l/llorente/northern_track_selection/part_1.csv')
    train_selections = [all_selections.loc[
                                (all_selections['n_pulses'] >= config['train_max_pulses'][0])&
                                (all_selections['n_pulses'] < config['train_max_pulses'][1]),:
                                ][config['index_column']].ravel().tolist()]
    val_selection = None
    test_database = None
    test_selection = None
    
    config["graph_definition"] = KNNGraph(detector=config["detector"],
                                          node_definition=config["node_definition"],
                                          nb_nearest_neighbours=config["nb_nearest_neighbours"],
                                          input_feature_names=config["features"],
                                          columns=config["columns_nearest_neighbours"],
                                        )



    
    if INFERENCE:
        config['scheduler_kwargs'] = None
    else:
        (training_dataloader, validation_dataloader,train_dataset, val_dataset) = make_dataloaders(db=all_databases, 
                                                                                                   train_selection=train_selections,
                                                                                                   val_selection=val_selection,
                                                                                                   config=config)


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
            #EarlyStopping(monitor='val_loss', patience=config['early_stopping_patience']),
            ProgressBar(),
            GradientAccumulationScheduler(scheduling=config['accumulate_grad_batches']),
            #LearningRateMonitor(logging_interval='step'),
        ]
        if validation_dataloader is not None:
            callbacks.append(ModelCheckpoint(
                dirpath=config['archive']+'/model_checkpoint_graphnet/',
                filename=run_name+'-{epoch:02d}-{val_loss:.6f}',
                monitor= 'val_loss',
                save_top_k = 30,
                every_n_epochs = 1,
                save_weights_only=False))

    model = build_model(config)
    
    if not INFERENCE:
        
        config['retrain_from_checkpoint'] = f'/remote/ceph/user/l/llorente/tito_solution/model_graphnet/{MODEL}-last.pth'
        
        if config['resume_training_path']:
            config['fit']['resume_training_path'] = config['resume_training_path']
        if config['retrain_from_checkpoint']:
            checkpoint_path = config['retrain_from_checkpoint']           
            checkpoint = torch.load(checkpoint_path, torch.device('cpu'))
            if 'state_dict' in checkpoint:
                checkpoint = checkpoint['state_dict']
            model.load_state_dict(checkpoint)
            del checkpoint
        if validation_dataloader is not None:
            config['validation_dataloader'] = validation_dataloader
            
            
        model.fit(
            training_dataloader,
            callbacks=callbacks,
            **config['fit'],
        )
        #model.save(os.path.join(config['archive'], f"{run_name}.pth"))
        #model.save_state_dict(os.path.join(config['archive'], f"{run_name}_state_dict.pth"))
        #print(f"Model saved to {config['archive']}/{run_name}.pth")
    else:

        all_res = []
        checkpoint_path = f'/remote/ceph/user/l/llorente/tito_solution/model_graphnet/{MODEL}-last.pth'

        factor = 1
        pulse_breakpoints = [0, 500, 1000, 1500, 2000, 3000]#, 10000]
        batch_sizes_per_pulse = [1750, 175, 40, 11, 4]#5, 2]
        config['num_workers'] = 8
        
        for min_pulse, max_pulse in zip(pulse_breakpoints[:-1], pulse_breakpoints[1:]):
            print(f'predicting {min_pulse} to {max_pulse} pulses with batch size {int(factor*batch_sizes_per_pulse[pulse_breakpoints.index(max_pulse)-1])}')
            results = inference(device=config['fit']['gpus'], 
                                test_min_pulses=min_pulse, 
                                test_max_pulses=max_pulse,
                                batch_size=int(factor*batch_sizes_per_pulse[pulse_breakpoints.index(max_pulse)-1]), 
                                checkpoint_path=checkpoint_path, 
                                use_all_features_in_prediction=True,
                                test_path=test_database,
                                test_selection_file=test_selection,
                                config=config)
            all_res.append(results)
            del results
            torch.cuda.empty_cache()

        results = pd.concat(all_res).sort_values(config['index_column'])

        run_name_pred = f'{MODEL}_northern_load_tito'
        path_to_save = '/remote/ceph/user/l/llorente/train_DynEdgeTITO_northern_Oct23/prediction_models'
        results.to_csv(f"{path_to_save}/{run_name_pred}.csv")
        print(f'predicted and saved in {path_to_save}/{run_name_pred}_graphnet_2.csv')