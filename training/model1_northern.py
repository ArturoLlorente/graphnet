import os
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import torch
from torch.optim.adam import Adam

from pytorch_lightning.callbacks import ModelCheckpoint, GradientAccumulationScheduler, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger 

from graphnet.data.dataset import EnsembleDataset
from graphnet.data.dataset import SQLiteDataset
from graphnet.data.constants import FEATURES, TRUTH

from graphnet.models import StandardModel
from graphnet.models.detector.icecube import IceCubeKaggle
from graphnet.models.graphs import KNNGraph
from graphnet.models.gnn import DynEdgeTITO
from graphnet.models.graphs.nodes import NodesAsPulses
from graphnet.models.task.reconstruction import DirectionReconstructionWithKappa

from graphnet.training.labels import Direction
from graphnet.training.loss_functions import VonMisesFisher3DLoss
from graphnet.training.callbacks import ProgressBar, PiecewiseLinearLR
from graphnet.training.utils import make_dataloader
from graphnet.training.utils import collate_fn, collator_sequence_buckleting
from graphnet.utilities.logging import Logger

from typing import Dict, List, Optional, Union, Callable, Tuple
from torch.utils.data import DataLoader

def split_selection(selection):
    """produces a 90% , 10% split for training and validation sets.

    Args:
        selection (pandas.DataFrame): A dataframe containing your selection

    Returns:
        train: indices for training. numpy.ndarray
        validate: indices for validation. numpy.ndarray
    """
    train, validate = np.split(selection, [int(.9*len(selection))])
    return train.tolist(), validate.tolist()
    
def make_dataloaders(
    db_train: Union[List[str], str],
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
    truth_table: str = 'meta_table',
    node_truth_table: Optional[str] = None,
    string_selection: List[int] = None,
    loss_weight_table: Optional[str] = None,
    loss_weight_column: Optional[str] = None,
    index_column: str = 'event_id',
    labels: Optional[Dict[str, Callable]] = None,
    collate_fn: Optional[Callable] = collate_fn,
) -> DataLoader:
    
    """Construct `DataLoader` instance."""
    # Check(s)
    if isinstance(pulsemaps, str):
        pulsemaps = [pulsemaps]
    if isinstance(db_train, list):
        assert len(train_selection) == len(db_train)
        train_datasets = []
        pbar = tqdm(total=len(db_train))
        for db_idx, database in enumerate(db_train):

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
                        path = db_train[-1],
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
    additional_attributes=['zenith', 'azimuth', 'event_id', 'energy']

    scheduler_config={
            "interval": "step",
        }


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
                test_min_pulses: int,
                test_max_pulses: int,
                batch_size: int,
                use_all_features_in_prediction: bool = True,
            ):
    test_path = '/mnt/scratch/rasmus_orsoe/databases/dev_northern_tracks_muon_labels_v3/dev_northern_tracks_muon_labels_v3_part_5.db'
    test_selection_file = pd.read_csv('/home/iwsatlas1/oersoe/phd/northern_tracks/energy_reconstruction/selections/dev_northern_tracks_muon_labels_v3_part_5_regression_selection.csv')
    test_selection = test_selection_file.loc[(test_selection_file['n_pulses']<test_max_pulses) & (test_selection_file['n_pulses']>test_min_pulses),:]['event_id'].ravel().tolist()

    test_dataloader =  make_dataloader(db = test_path,
                                        selection = test_selection,
                                        pulsemaps = 'pulse_table',
                                        num_workers = 64,
                                        features = FEATURES.KAGGLE,
                                        shuffle = False,
                                        truth = TRUTH.KAGGLE,
                                        batch_size = batch_size,
                                        truth_table = 'meta_table',
                                        index_column='event_id',
                                        labels = {'direction': Direction()},  
                                        )
    
    model = build_model(dyntrans_layer_sizes=dyntrans_layer_sizes,
                        global_pooling_schemes=global_pooling_schemes,
                        use_global_features=use_global_features,
                        use_post_processing_layers=use_post_processing_layers,
                        scheduler_class=scheduler_class,
                        scheduler_kwargs=scheduler_kwargs,
                        )
    
    model.eval()
    model.inference()
    checkpoint_path = f'/remote/ceph/user/l/llorente/tito_solution/northeren_tracks_Oct3/model1-1e.pth'
    checkpoint = torch.load(checkpoint_path, torch.device('cpu'))
    
    if 'state_dict' in checkpoint:
        checkpoint = checkpoint['state_dict']
    model.load_state_dict(checkpoint)


    event_ids = []
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
            event_ids.append(batch.event_id)
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
    results['event_id'] = torch.cat(event_ids).to('cpu').detach().numpy()
    if validateMode:
        results['zenith'] = torch.cat(zenith).to('cpu').numpy()
        results['azimuth'] = torch.cat(azimuth).to('cpu').numpy()

    return results
# Main function call
if __name__ == "__main__":
    
    target = ['direction']
    archive = "/remote/ceph/user/l/llorente/training_1e_test/graphnet_model"
    weight_column_name = None 
    weight_table_name =  None
    batch_size = 100
    n_round = 1
    #CUDA_DEVICE = 0
    #device = [CUDA_DEVICE] if CUDA_DEVICE is not None else None
    device = [0]
    num_workers = 64
    pulsemap = 'pulse_table'
    node_truth_table = None
    node_truth = None
    truth_table = 'meta_table'
    index_column = 'event_id'
    labels = {'direction': Direction()}
    global_pooling_schemes = ["max"]
    dyntrans_layer_sizes = [(256, 256),
                            (256, 256),
                            (256, 256),
                            (256, 256)]
    accumulate_grad_batches = {0: 1}
    num_database_files = 2
    use_global_features = True
    use_post_processing_layers = True
    train_max_pulses = 300
    val_max_pulses = 300
    scheduler_class = PiecewiseLinearLR

    # Configurations
    torch.multiprocessing.set_sharing_strategy('file_system')
    #torch.multiprocessing.set_start_method('spawn', force=True)

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
    # get_list_of_databases:
    train_selections = []
    for selection in all_selections:
        train_selections.append(selection.loc[selection['n_pulses'] < train_max_pulses,:]['event_id'].ravel().tolist())
    train_selections[-1] = train_selections[-1][:int(len(train_selections[-1])*0.9)]
    val_selection = train_selections[-1][int(len(train_selections[-1])*0.9):]

    print("Loading databases")

    graph_definition = KNNGraph(
        detector=IceCubeKaggle(),
        node_definition=NodesAsPulses(),
        nb_nearest_neighbours=6,
        node_feature_names=features,
    )

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

    
    run_name = f"dynedgeTITO__direction_reco_{n_round}e_train_max_pulses{train_max_pulses}_val_max_pulses{val_max_pulses}_layersize{len(dyntrans_layer_sizes)}_use_GG_{use_global_features}_use_PP_{use_post_processing_layers}_batch_{batch_size}_nround{n_round}_testing_db"
    
    if device is not None:
        scheduler_kwargs={
            "milestones": [
                0,
                10  * len(training_dataloader)//(len(device)*accumulate_grad_batches[0]*n_round),
                len(training_dataloader)//(len(device)*accumulate_grad_batches[0]*(20/10)),
                len(training_dataloader)//(len(device)*accumulate_grad_batches[0]),                
            ],
            "factors": [1e-03, 1, 1, 1e-03],
            "verbose": False,
        }
    else:
        scheduler_kwargs={
            "milestones": [
                0,
                10  * len(training_dataloader)//(accumulate_grad_batches[0]*num_database_files*55*n_round),
                len(training_dataloader)//(accumulate_grad_batches[0]*(20/10)),
                len(training_dataloader)//(accumulate_grad_batches[0]),                
            ],
            "factors": [1e-03, 1, 1, 1e-03],
            "verbose": False,
        }

    print("Starting training")
    model = build_model(graph_definition=graph_definition,
                        dyntrans_layer_sizes=dyntrans_layer_sizes,
                        global_pooling_schemes=global_pooling_schemes,
                        use_global_features=use_global_features,
                        use_post_processing_layers=use_post_processing_layers,
                        scheduler_class=scheduler_class,
                        scheduler_kwargs=scheduler_kwargs,
                        )
    

    # Training model
    callbacks = [
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

    distribution_strategy = 'ddp'

    model.fit(
        training_dataloader,
        validation_dataloader,
        callbacks=callbacks,
        max_epochs=n_round,
        gpus=device,
        distribution_strategy=distribution_strategy,
        check_val_every_n_epoch=1,
        precision=32,
        #logger=WandbLogger(project='train_6batch_1e_graphnet', name=run_name),
        #reload_dataloaders_every_n_epochs=1,
    )
        
    #model.save(os.path.join(archive, f"{run_name}.pth"))
    #model.save_state_dict(os.path.join(archive, f"{run_name}_state_dict.pth"))
    #print(f"Model saved to {archive}/{run_name}.pth")
        
        
        
        
    ##results0 = inference(device=device, test_min_pulses=0, test_max_pulses=500, batch_size=1000)
    ##results1 = inference(device=device, test_min_pulses=500, test_max_pulses=1000, batch_size=350)
    ##results2 = inference(device=device, test_min_pulses=1000, test_max_pulses=1500, batch_size=150)
    ##results3 = inference(device=device, test_min_pulses=1500, test_max_pulses=2000, batch_size=50)
    ##results4 = inference(device=device, test_min_pulses=2000, test_max_pulses=3000, batch_size=20)
    ##    
    ##results = pd.concat([results0, results1, results2, results3, results4]).sort_values('event_id')
    ##
    ##run_name = 'model1_batch1-55'       
    ##results.to_csv(f"/remote/ceph/user/l/llorente/prediction_models/graphnet_predictions/{run_name}_graphnet.csv")
    ##print(f'predicted and saved in /remote/ceph/user/l/llorente/prediction_models/graphnet_predictions/{run_name}_graphnet.csv')