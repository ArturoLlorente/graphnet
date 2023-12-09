import os
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import gc

import torch
from torch.optim.adam import Adam

from pytorch_lightning.callbacks import ModelCheckpoint, GradientAccumulationScheduler, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger 
from pytorch_lightning.callbacks import EarlyStopping

from graphnet.data.dataset import EnsembleDataset
from graphnet.data.dataset import SQLiteDataset
from graphnet.data.constants import FEATURES, TRUTH

from graphnet.models import StandardModel, StandardModelPred
from graphnet.models.detector.icecube import IceCube86
from graphnet.models.graphs import KNNGraph
from graphnet.models.gnn import DynEdgeTITO
from graphnet.models.graphs.nodes import NodesAsPulses
from graphnet.models.task.reconstruction import DirectionReconstructionWithKappa

from graphnet.training.labels import Direction
from graphnet.training.loss_functions import VonMisesFisher3DLoss
from graphnet.training.callbacks import ProgressBar, PiecewiseLinearLR
from graphnet.training.utils import make_dataloader
from graphnet.training.utils import collator_sequence_buckleting
from graphnet.utilities.logging import Logger

from typing import Dict, List, Optional, Union, Callable, Tuple
from torch.utils.data import DataLoader

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
    graph_definition: Optional[KNNGraph],
    pulsemaps: Union[str, List[str]],
    features: List[str],
    truth: List[str],
    *,
    batch_size: int = 256,
    train_selection: Optional[List[int]],
    val_selection: Optional[List[int]],
    num_workers: int = 30,
    persistent_workers: bool = True,
    node_truth: List[str] = None,
    truth_table: str = 'truth',
    node_truth_table: Optional[str] = None,
    string_selection: List[int] = None,
    loss_weight_table: Optional[str] = None,
    loss_weight_column: Optional[str] = None,
    index_column: str = 'event_no',
    labels: Optional[Dict[str, Callable]] = None,
    collate_fn: Optional[Callable] = collator_sequence_buckleting(),
) -> DataLoader:
    
    """Construct `DataLoader` instance."""
    # Check(s)
    if isinstance(pulsemaps, str):
        pulsemaps = [pulsemaps]
    if isinstance(db, list):
        assert len(train_selection) == len(db)
        train_datasets = []
        pbar = tqdm(total=len(db))
        for db_idx, database in enumerate(db):

            train_datasets.append(SQLiteDataset(
                path=database,
                graph_definition=graph_definition,
                pulsemaps=pulsemaps,
                features=features,
                truth=truth,
                selection=train_selection[db_idx],
                node_truth=node_truth,
                truth_table=truth_table,
                node_truth_table=node_truth_table,
                string_selection=string_selection,
                loss_weight_table=loss_weight_table,
                loss_weight_column=loss_weight_column,
                index_column=index_column,
            ))
            pbar.update(1)

        if isinstance(labels, dict):
            for label in labels.keys():
                for train_dataset in train_datasets:
                    train_dataset.add_label(key=label, fn=labels[label])
                    
        train_dataset = EnsembleDataset(train_datasets)

        training_dataloader = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=collate_fn,
            persistent_workers=persistent_workers,
            prefetch_factor=2
        )

        val_dataset = SQLiteDataset(
                        path = db[-1],
                        graph_definition=graph_definition,
                        pulsemaps=pulsemaps,
                        features=features,
                        truth=truth,
                        selection=val_selection,
                        node_truth=node_truth,
                        truth_table=truth_table,
                        node_truth_table=node_truth_table,
                        string_selection=string_selection,
                        loss_weight_table=loss_weight_table,
                        loss_weight_column=loss_weight_column,
                        index_column=index_column,
        )
        
        if isinstance(labels, dict):
            for label in labels.keys():
                val_dataset.add_label(key=label, fn=labels[label])
        
        validation_dataloader = DataLoader(
            dataset=val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn,
            persistent_workers=persistent_workers,
            prefetch_factor=2,
        )
        
                    
    return training_dataloader, validation_dataloader, train_dataset, val_dataset


def build_model(
    *,
    graph_definition: Optional[KNNGraph],
    dyntrans_layer_sizes: Optional[List[Tuple[int, ...]]] = [(256, 256), (256, 256), (256, 256)],
    global_pooling_schemes: Optional[List[str]] = ["max"],
    use_global_features: bool = True,
    use_post_processing_layers: bool = True,
    scheduler_class: Optional[type] = None,
    scheduler_kwargs: Optional[dict] = None,
    ):

    gnn = DynEdgeTITO(
            nb_inputs=graph_definition.nb_outputs,
            dyntrans_layer_sizes=dyntrans_layer_sizes,
            global_pooling_schemes=global_pooling_schemes,
            use_global_features=use_global_features,
            use_post_processing_layers=use_post_processing_layers,
            )
    
    task = DirectionReconstructionWithKappa(
            hidden_size=gnn.nb_outputs,
            target_labels="direction",
            loss_function=VonMisesFisher3DLoss(),
        )
        
    prediction_columns =['dir_x_pred', 'dir_y_pred', 'dir_z_pred', 'dir_kappa_pred']
    additional_attributes=['zenith', 'azimuth', 'event_no', 'energy']

    scheduler_config={
            "interval": "step",
        }

    if INFERENCE:
        model = StandardModelPred(
                graph_definition=graph_definition,
                gnn=gnn,
                tasks=[task],
                optimizer_class=Adam,
                optimizer_kwargs={'lr': 1e-03, 'eps': 1e-03},
                scheduler_class= scheduler_class,
                scheduler_kwargs=scheduler_kwargs,
                scheduler_config=scheduler_config,
            )
    else:
        model = StandardModel(
                graph_definition=graph_definition,
                gnn=gnn,
                tasks=[task],
                optimizer_class=Adam,
                optimizer_kwargs={'lr': 1e-03, 'eps': 1e-03},
                scheduler_class= scheduler_class,
                scheduler_kwargs=scheduler_kwargs,
                scheduler_config=scheduler_config,
            )
    model.prediction_columns = prediction_columns
    model.additional_attributes = additional_attributes
    
    return model


def inference(device: int,
                checkpoint_path: str,
                test_min_pulses: int,
                test_max_pulses: int,
                batch_size: int,
                use_all_features_in_prediction: bool = True,
                graph_definition: Optional[KNNGraph] = None,
                dyntrans_layer_sizes: Optional[List[Tuple[int, ...]]] = [(256, 256), (256, 256), (256, 256)],
                global_pooling_schemes: Optional[List[str]] = ["max"],
                use_global_features: bool = True,
                use_post_processing_layers: bool = True,
                scheduler_class: Optional[type] = None,
                scheduler_kwargs: Optional[dict] = None,
                ):

    test_path = '/mnt/scratch/rasmus_orsoe/databases/dev_northern_tracks_muon_labels_v3/dev_northern_tracks_muon_labels_v3_part_5.db'
    test_selection_file = pd.read_csv('/remote/ceph/user/l/llorente/northern_track_selection/part_5.csv')
    test_selection = test_selection_file.loc[(test_selection_file['n_pulses']<=test_max_pulses) & (test_selection_file['n_pulses']>test_min_pulses),:]['event_no'].ravel().tolist()

    test_dataloader =  make_dataloader(db = test_path,
                                        selection = test_selection,
                                        graph_definition = graph_definition,
                                        pulsemaps = 'InIceDSTPulses',
                                        num_workers = 8,
                                        features = FEATURES.ICECUBE86,
                                        shuffle = False,
                                        truth = TRUTH.ICECUBE86,
                                        batch_size = batch_size,
                                        truth_table = 'truth',
                                        index_column='event_no',
                                        labels = {'direction': Direction()},  
                                        )
    

    model = build_model(graph_definition=graph_definition,
                        dyntrans_layer_sizes=dyntrans_layer_sizes,
                        global_pooling_schemes=global_pooling_schemes,
                        use_global_features=use_global_features,
                        use_post_processing_layers=use_post_processing_layers,
                        scheduler_class=scheduler_class,
                        scheduler_kwargs=scheduler_kwargs,
                        )
    model.eval()
    model.inference()
    checkpoint = torch.load(checkpoint_path, torch.device('cpu'))
    
    if 'state_dict' in checkpoint:
        checkpoint = checkpoint['state_dict']
    model.load_state_dict(checkpoint)


    event_nos = []
    zenith = []
    azimuth = []
    preds = []
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
            columns = ['direction_x','direction_y','direction_z','direction_kappa1','direction_x1','direction_y1','direction_z1','direction_kappa'] + [f'idx{i}' for i in range(128)]
        else:
            columns = ['direction_x','direction_y','direction_z','direction_kappa'] + [f'idx{i}' for i in range(128)]
    else:
        columns = ['direction_x','direction_y','direction_z','direction_kappa']
        
    results = pd.DataFrame(preds, columns=columns)
    results['event_no'] = torch.cat(event_nos).to('cpu').detach().numpy()
    if validateMode:
        results['zenith'] = torch.cat(zenith).to('cpu').numpy()
        results['azimuth'] = torch.cat(azimuth).to('cpu').numpy()

    return results
# Main function call
if __name__ == "__main__":
    
    target = ['direction']
    archive = "/remote/ceph/user/l/llorente/train_DynEdgeTITO_northern_Oct23"
    weight_column_name = None 
    weight_table_name =  None
    batch_size = 350
    n_epochs = 50
    device = [2]
    num_workers = 16
    pulsemap = 'InIceDSTPulses'
    node_truth_table = None
    node_truth = None
    truth_table = 'truth'
    index_column = 'event_no'
    labels = {'direction': Direction()}
    global_pooling_schemes = ["max"]
    accumulate_grad_batches = {0: 2}
    num_database_files = 4
    train_max_pulses = 1000
    val_max_pulses = 1000
    scheduler_class = PiecewiseLinearLR
    wandb = False
    INFERENCE = True
    ## Diferent models

    #resume_training_path = False
    #resume_training_path = '/remote/ceph/user/l/llorente/train_DynEdgeTITO_northern_Oct23/model_checkpoint_graphnet/model3_NEWTEST_dynedgeTITO_directionReco_50e_trainMaxPulses1000_valMaxPulses1000_layerSize3_useGGFalse_usePPFalse_batch350_numDatabaseFiles4_accGradBatch2-epoch=19-val_loss=-2.650039.ckpt'
    MODEL = 'model1'
    use_global_features = use_global_features_all[MODEL]
    use_post_processing_layers = use_post_processing_layers_all[MODEL]
    dyntrans_layer_sizes = dyntrans_layer_sizes_all[MODEL]
    columns_nearest_neighbours = columns_nearest_neighbours_all[MODEL]


    run_name = (f"{MODEL}_NEWTEST_dynedgeTITO_directionReco_{n_epochs}e_trainMaxPulses{train_max_pulses}_valMaxPulses{val_max_pulses}"
                f"_layerSize{len(dyntrans_layer_sizes)}_useGG{use_global_features}_usePP{use_post_processing_layers}_batch{batch_size}"
                f"_numDatabaseFiles{num_database_files}_accGradBatch{accumulate_grad_batches[0]}")
    #run_name = "dummy"

    # Configurations
    torch.multiprocessing.set_sharing_strategy('file_system')
    torch.multiprocessing.set_start_method('spawn', force=True)

    # Constants
    features = FEATURES.ICECUBE86
    truth = TRUTH.ICECUBE86
    
    all_databases = ['/mnt/scratch/rasmus_orsoe/databases/dev_northern_tracks_muon_labels_v3/dev_northern_tracks_muon_labels_v3_part_1.db',
                    '/mnt/scratch/rasmus_orsoe/databases/dev_northern_tracks_muon_labels_v3/dev_northern_tracks_muon_labels_v3_part_2.db',
                    '/mnt/scratch/rasmus_orsoe/databases/dev_northern_tracks_muon_labels_v3/dev_northern_tracks_muon_labels_v3_part_3.db',
                    '/mnt/scratch/rasmus_orsoe/databases/dev_northern_tracks_muon_labels_v3/dev_northern_tracks_muon_labels_v3_part_4.db']
    # get selections:
    all_selections = [pd.read_csv('/home/iwsatlas1/oersoe/phd/northern_tracks/energy_reconstruction/selections/dev_northern_tracks_muon_labels_v3_part_1_regression_selection.csv'),
                    pd.read_csv('/home/iwsatlas1/oersoe/phd/northern_tracks/energy_reconstruction/selections/dev_northern_tracks_muon_labels_v3_part_2_regression_selection.csv'),
                    pd.read_csv('/home/iwsatlas1/oersoe/phd/northern_tracks/energy_reconstruction/selections/dev_northern_tracks_muon_labels_v3_part_3_regression_selection.csv'),
                    pd.read_csv('/home/iwsatlas1/oersoe/phd/northern_tracks/energy_reconstruction/selections/dev_northern_tracks_muon_labels_v3_part_4_regression_selection.csv')]
    
    all_databases = all_databases[:num_database_files]
    print(f'using {len(all_databases)} databases')
    all_selections = all_selections[:num_database_files]

    #all_databases = ['/remote/ceph/user/l/llorente/northeren_tracks_ensembled/northern_tracks_part5.db']
    #all_selections = [pd.read_csv('/home/iwsatlas1/oersoe/phd/northern_tracks/energy_reconstruction/selections/dev_northern_tracks_muon_labels_v3_part_5_regression_selection.csv')]
    # get_list_of_databases:
    train_selections = []
    for selection in all_selections:
        train_selections.append(selection.loc[selection['n_pulses'] < train_max_pulses,:][index_column].ravel().tolist())
    train_selections[-1] = train_selections[-1][:int(len(train_selections[-1])*0.9)]
    val_selection = train_selections[-1][int(len(train_selections[-1])*0.9):]

    print("Loading databases")

    graph_definition = KNNGraph(
        detector=IceCube86(),
        node_definition=NodesAsPulses(),
        nb_nearest_neighbours=6,
        node_feature_names=features,
        columns=columns_nearest_neighbours,
    )



    
    if INFERENCE:
        scheduler_kwargs=None
    else:
        (training_dataloader, 
        validation_dataloader,
        train_dataset, 
        val_dataset) = make_dataloaders(db=all_databases,
                                            graph_definition=graph_definition,
                                            pulsemaps=pulsemap,
                                            features=features,
                                            truth=truth,
                                            batch_size=batch_size,
                                            train_selection=train_selections,
                                            val_selection=val_selection,
                                            num_workers=num_workers,
                                            persistent_workers=True,
                                            node_truth=node_truth,
                                            truth_table=truth_table,
                                            node_truth_table=node_truth_table,
                                            string_selection=None,
                                            loss_weight_table=None,
                                            loss_weight_column=None,
                                            index_column=index_column,
                                            labels=labels,
                                            collate_fn=collator_sequence_buckleting(),
                                            )
        scheduler_kwargs={
            "milestones": [
                0,
                len(training_dataloader)//(len(device)*accumulate_grad_batches[0]*30),
                len(training_dataloader)*n_epochs//(len(device)*accumulate_grad_batches[0]*2),
                len(training_dataloader)*n_epochs//(len(device)*accumulate_grad_batches[0]),                
            ],
            "factors": [1e-03, 1, 1, 1e-03],
            "verbose": False,
        }

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=15),
            ModelCheckpoint(
                dirpath=archive+'/model_checkpoint_graphnet/',
                filename=run_name+'-{epoch:02d}-{val_loss:.6f}',
                monitor= 'val_loss',
                save_top_k = 30,
                every_n_epochs = 1,
                save_weights_only=False,
            ),
            ProgressBar(),
            GradientAccumulationScheduler(scheduling=accumulate_grad_batches),
            #LearningRateMonitor(logging_interval='step'),
        ]


    print("Starting training")
    model = build_model(graph_definition=graph_definition,
                        dyntrans_layer_sizes=dyntrans_layer_sizes,
                        global_pooling_schemes=global_pooling_schemes,
                        use_global_features=use_global_features,
                        use_post_processing_layers=use_post_processing_layers,
                        scheduler_class=scheduler_class,
                        scheduler_kwargs=scheduler_kwargs,
                        )
    distribution_strategy = 'ddp'
        
    if not INFERENCE:
        fit_params = {"max_epochs": n_epochs,
                      "gpus": device,
                      "distribution_strategy": distribution_strategy,
                      "check_val_every_n_epoch": 1,
                      "precision": 32,
                      }
        
        if resume_training_path:
            fit_params['ckpt_path'] = resume_training_path
            
        model.fit(
            training_dataloader,
            validation_dataloader,
            callbacks=callbacks,
            **fit_params,
        )
        model.save(os.path.join(archive, f"{run_name}.pth"))
        model.save_state_dict(os.path.join(archive, f"{run_name}_state_dict.pth"))
        print(f"Model saved to {archive}/{run_name}.pth")
    else:
        
        all_res = []
        #checkpoint_path = (os.path.join(archive, f"{run_name}_state_dict.pth"))
        checkpoint_path = '/remote/ceph/user/l/llorente/train_DynEdgeTITO_northern_Oct23/model1_NEWTEST_dynedgeTITO_directionReco_50e_trainMaxPulses1000_valMaxPulses1000_layerSize4_useGGTrue_usePPTrue_batch256_numDatabaseFiles4_accGradBatch2_state_dict.pth'

        factor = 1
        pulse_breakpoints = [0, 500, 1000, 1500, 2000, 3000]*factor
        batch_sizes_per_pulse = [1750, 150, 40, 11, 4]
        
        for min_pulse, max_pulse in zip(pulse_breakpoints[:-1], pulse_breakpoints[1:]):
            print(f'predicting {min_pulse} to {max_pulse} pulses with batch size {batch_sizes_per_pulse[pulse_breakpoints.index(max_pulse)-1]}')
            results = inference(device=device, 
                                test_min_pulses=min_pulse, 
                                test_max_pulses=max_pulse,
                                batch_size=batch_sizes_per_pulse[pulse_breakpoints.index(max_pulse)-1], 
                                checkpoint_path=checkpoint_path, 
                                use_all_features_in_prediction=True,
                                graph_definition=graph_definition,
                                dyntrans_layer_sizes=dyntrans_layer_sizes,
                                global_pooling_schemes=global_pooling_schemes,
                                use_global_features=use_global_features,
                                use_post_processing_layers=use_post_processing_layers,
                                scheduler_class=scheduler_class,
                                scheduler_kwargs=scheduler_kwargs)
            all_res.append(results)
            del results
            torch.cuda.empty_cache()

        results = pd.concat(all_res).sort_values(config['index_column'])
        run_name_pred = f'{MODEL}_retrain_northern'
        path_to_save = '/remote/ceph/user/l/llorente/tito_northern_retrain'
        results.to_csv(f"{path_to_save}/{run_name_pred}.csv")
        print(f'predicted and saved in {path_to_save}/{run_name_pred}.csv')