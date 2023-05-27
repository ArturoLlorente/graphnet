import os
import numpy as np
import pandas as pd
import dill

import torch
from torch.optim.adam import Adam

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from graphnet.data.dataset import Dataset, EnsembleDataset
from graphnet.data.sqlite import SQLiteDataset
from graphnet.data.constants import FEATURES, TRUTH

from graphnet.models import StandardModel
from graphnet.models.detector.icecube import IceCube86
from graphnet.models.graph_builders import KNNGraphBuilder
from graphnet.models.gnn import DynEdgeTITO
from graphnet.models.coarsening import Coarsening
from graphnet.models.task import IdentityTask
from graphnet.models.task.reconstruction import DirectionReconstructionWithKappa

from graphnet.training.labels import Direction
from graphnet.training.loss_functions import LogCoshLoss
from graphnet.training.loss_functions import VonMisesFisher3DLoss
from graphnet.training.callbacks import ProgressBar, PiecewiseLinearLR
from graphnet.training.utils import get_predictions, make_dataloader

from graphnet.utilities.logging import Logger

from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from graphnet.models.graph_builders import KNNGraphBuilder
from typing import Dict, List, Optional, Union, Callable, Tuple
from torch.utils.data import DataLoader
from torch_geometric.data import Batch, Data


def save_results(
    db: str, tag: str, results: pd.DataFrame, archive: str, model
) -> None:
    """Save trained model and prediction `results` in `db`."""
    db_name = db.split("/")[-1].split(".")[0]
    path = archive + "/" + db_name + "/" + tag
    os.makedirs(path, exist_ok=True)
    results.to_csv(path + "/results.csv")
    #model.save_state_dict(path + "/" + tag + "_state_dict.pth")
    #model.save(path + "/" + tag + "_model.pth")
    return

def scale_XYZ(x):
    x[:,0] = x[:,0]/764.431509
    x[:,1] = x[:,1]/785.041607
    x[:,2] = x[:,2]/1083.249944
    return x

def unscale_XYZ(x):
    x[:,0] = 764.431509*x[:,0]
    x[:,1] = 785.041607*x[:,1]
    x[:,2] = 1083.249944*x[:,2]
    return x

def remove_log10(x):
    return torch.pow(10, x)

def transform_to_log10(x):
    return torch.log10(x)

def collate_fn(graphs: List[Data]) -> Batch:
    """Remove graphs with less than two DOM hits.

    Should not occur in "production.
    """
    graphs = [g for g in graphs if g.n_pulses > 1]
    return Batch.from_data_list(graphs)

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
    db: Union[List[str], str],
    pulsemaps: Union[str, List[str]],
    features: List[str],
    truth: List[str],
    *,
    batch_size: int = 256,
    selection: Optional[List[int]],
    num_workers: int = 10,
    persistent_workers: bool = True,
    node_truth: List[str] = None,
    truth_table: str = "truth",
    node_truth_table: Optional[str] = None,
    string_selection: List[int] = None,
    loss_weight_table: Optional[str] = None,
    loss_weight_column: Optional[str] = None,
    index_column: str = 'event_no',
    labels: Optional[Dict[str, Callable]] = None,
) -> DataLoader:
    
    """Construct `DataLoader` instance."""
    # Check(s)
    if isinstance(pulsemaps, str):
        pulsemaps = [pulsemaps]
    if isinstance(db, list):
        assert len(selection) == len(db)
        split_selections = {}
        for k in range(len(db)):
            train_selection, validation_selection = split_selection(selection[k])
            split_selections[db[k]] = {'train': train_selection,
                                          'validation': validation_selection}
            print(db[k])
        train_datasets = []
        validate_datasets = []
        c = 0
        for database in db:
            print(database)
            train_datasets.append(SQLiteDataset(
                path=database,
                pulsemaps=pulsemaps,
                features=features,
                truth=truth,
                selection=split_selections[database]['train'],
                node_truth=node_truth,
                truth_table=truth_table,
                node_truth_table=node_truth_table,
                string_selection=string_selection,
                loss_weight_table=loss_weight_table,
                loss_weight_column=loss_weight_column,
                index_column=index_column,
            ))
            validate_datasets.append(SQLiteDataset(
                path=database,
                pulsemaps=pulsemaps,
                features=features,
                truth=truth,
                selection=split_selections[database]['validation'],
                node_truth=node_truth,
                truth_table=truth_table,
                node_truth_table=node_truth_table,
                string_selection=string_selection,
                loss_weight_table=loss_weight_table,
                loss_weight_column=loss_weight_column,
                index_column=index_column,
            ))
            c +=1
            # adds custom labels to dataset
            if isinstance(labels, dict):
                for label in labels.keys():
                    for train_dataset in train_datasets:
                        train_dataset.add_label(key=label, fn=labels[label])
                    for val_dataset in validate_datasets:
                        val_dataset.add_label(key=label, fn=labels[label])
                        
            train_dataset = EnsembleDataset(train_datasets)
            val_dataset = EnsembleDataset(validate_datasets)
            
            training_dataloader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                collate_fn=collate_fn,
                persistent_workers=persistent_workers,
                prefetch_factor=2,
            )
            validation_dataloader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                collate_fn=collate_fn,
                persistent_workers=persistent_workers,
                prefetch_factor=2,
            )
                        
    elif isinstance(db, str):
        
        train_selection, validate_selection = split_selection(selection)
        
        common_kwargs = dict(
            db=db,
            pulsemaps=pulsemaps,
            features=features,
            truth=truth,
            batch_size=batch_size,
            num_workers=num_workers,
            persistent_workers=persistent_workers,
            node_truth=node_truth,
            node_truth_table=node_truth_table,
            string_selection=string_selection,
            labels=labels,
        )
        training_dataloader = make_dataloader(
            selection=train_selection,
            shuffle = True,
            **common_kwargs,
        )

        validation_dataloader = make_dataloader(
            selection=validate_selection,
            shuffle=False,
            **common_kwargs,
        )
    else:
        assert 1 == 2, "dont use this code"



    return training_dataloader, validation_dataloader

def build_model(run_name, device, archive):
    model = torch.load(os.path.join(archive, f"{run_name}.pth"),pickle_module=dill)
    model.to('cuda:%s'%device[CUDA_DEVICE] if CUDA_DEVICE is not None else 'cpu')
    model.eval()
    model.inference()
    return model 

def train_and_predict_on_validation_set(
    *,
    target: Union[str, List[str]],
    run_name: str,
    archive: str, 
    training_dataloader: DataLoader, 
    validation_dataloader: DataLoader, 
    test_dataloader: DataLoader,
    n_epochs: int = 25,
    patience: int = 10, 
    device: int = None, 
    rop: bool = False, 
    coarsening: Optional[Coarsening] = None,
    nb_nearest_neighbours: int = 6,
    features_subset: Optional[slice] = slice(0, 4),
    dyntrans_layer_sizes: Optional[List[Tuple[int, ...]]] = [(256, 256), (256, 256), (256, 256)],
    global_pooling_schemes: Optional[List[str]] = ["max"],
    wandb: bool = False,
    only_load_model: bool = False,
    accumulate_grad_batches: int = 2,
    ):
    print(f"features: {features}")
    print(f"truth: {truth}")

    #logger = Logger()


    # Building model
    detector = IceCube86(graph_builder=KNNGraphBuilder(nb_nearest_neighbours=nb_nearest_neighbours))
    
    gnn = DynEdgeTITO(
            nb_inputs=detector.nb_outputs,
            features_subset=features_subset,
            dyntrans_layer_sizes=dyntrans_layer_sizes,
            global_pooling_schemes=global_pooling_schemes,
            )
    
    if target == 'classification_emuon_entry':
        task = IdentityTask(nb_outputs = 1,
                           hidden_size=gnn.nb_outputs,
                           target_labels=target, 
                           loss_function=LogCoshLoss(), 
                           transform_target = transform_to_log10, 
                           transform_inference = remove_log10,
                           loss_weight = None,)
        prediction_columns =[target + "_pred"]
        additional_attributes=[target, "event_no"]
    elif target == 'direction':
        task = DirectionReconstructionWithKappa(
                           hidden_size=gnn.nb_outputs,
                           target_labels=target,
                           loss_function=VonMisesFisher3DLoss(),  
                           loss_weight = None,
        )
        prediction_columns =["dir_x_pred", "dir_y_pred", "dir_z_pred", "dir_kappa_pred"]
        additional_attributes=['zenith', 'azimuth', "event_no", "energy"]
    else:
        assert 1 == 2, "Task not found"

    if rop:
        scheduler_class = ReduceLROnPlateau
        scheduler_kwargs = {'patience': 5}
        scheduler_config = {'frequency': 1, 'monitor': 'val_loss'}
    else:
        scheduler_class= PiecewiseLinearLR,
        scheduler_kwargs={
            'milestones': [0, len(training_dataloader) / 2, len(training_dataloader) * n_epochs],
            'factors': [1e-2, 1, 1e-02],
        },
        scheduler_config={
            'interval': 'step',
        },


    model = StandardModel(
        detector=detector,
        gnn=gnn,
        tasks=[task],
        optimizer_class=Adam,
        optimizer_kwargs={'lr': 1e-03, 'eps': 1e-03},
        scheduler_class= scheduler_class,
        scheduler_kwargs=scheduler_kwargs,
        scheduler_config=scheduler_config,
        coarsening = coarsening
     )
    print(model)
    
    # Training model
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=patience,
        ),
        ProgressBar(),
    ]
    
    trainer = Trainer(
        default_root_dir=archive + '/' + run_name,
        accelerator='gpu' if device is not None else None,
        gpus=device,
        max_epochs=n_epochs,
        callbacks=callbacks,
        log_every_n_steps=1,
        accumulate_grad_batches=accumulate_grad_batches,
        #logger= wandb_logger if wandb else None,
        #strategy='ddp'
        #resume_from_checkpoint = 
    )
    if only_load_model:
        model.load_state_dict('/remote/ceph/user/l/llorente/northeren_tracks/dynedgeTITO_muon_entry_direction_reco_direction_InIceDSTPulses_1e_p1_max_pulses600_coarsening_none_rop_True_size3_nb6_state_dict.pth')
    else:   
        try:
            trainer.fit(model, training_dataloader, validation_dataloader)
        except KeyboardInterrupt:
            print("[ctrl+c] Exiting gracefully.")
        
        model.save(os.path.join(archive, f"{run_name}.pth"))
        model.save_state_dict(os.path.join(archive, f"{run_name}_state_dict.pth"))

    #predict(model,trainer,target,validation_dataloader, additional_attributes = additional_attributes, device = device, tag = 'valid', prediction_columns = prediction_columns)
    predict(model,trainer,target,test_dataloader, additional_attributes = additional_attributes, device = device, tag = 'test', prediction_columns = prediction_columns)
    

def predict(model,
            trainer,
            target,
            dataloader, 
            additional_attributes,
            device,
            tag,
            prediction_columns):
    try:
        del truth[truth.index('interaction_time')]
    except ValueError:
        # not found in list
        pass
    
    device = 'cuda:%s'%CUDA_DEVICE if CUDA_DEVICE is not None else 'cpu'
    model.to(device)
    model.eval()
    model.inference()
    results = get_predictions(
        trainer = trainer,
        model = model,
        dataloader = dataloader,
        prediction_columns = prediction_columns,
        additional_attributes= additional_attributes,
        node_level = True if target == 'truth_flag' else False
    )
    
    save_results(db='/mnt/scratch/rasmus_orsoe/databases/dev_northeren_tracks_muon_labels_v3/dev_northern_tracks_muon_labels_v3_part_5.db', 
                tag=run_name + '_' + tag, 
                results=results, 
                archive=archive, 
                model=model)
    return

# Main function call
if __name__ == "__main__":
    
    target = 'direction'
    archive = "/remote/ceph/user/l/llorente/northeren_tracks"
    weight_column_name = None 
    weight_table_name =  None
    batch_size = 256
    CUDA_DEVICE = 3
    device = [CUDA_DEVICE] if CUDA_DEVICE is not None else None
    n_epochs = 1
    num_workers = 64
    patience = 1
    pulsemap = 'InIceDSTPulses'
    node_truth_table = None
    node_truth = None
    truth_table = 'northeren_tracks_muon_labels'
    index_column = 'event_no'
    max_pulses = 50
    labels = {'direction': Direction()}
    coarsening = None #DOMCoarsening()
    rop = True
    nb_nearest_neighbours = 6
    layer_size_scale = 3
    global_pooling_schemes = ["max"]
    features_subset = slice(0, 4)
    dyntrans_layer_sizes = [(256, 256),
                            (256, 256),
                            (256, 256)]
    accumulate_grad_batches = 4
    

    

    # Configurations
    torch.multiprocessing.set_sharing_strategy('file_system')

    # Constants
    features = FEATURES.ICECUBE86
    truth = TRUTH.ICECUBE86
    
    # get_list_of_databases:
    databases = '/mnt/scratch/rasmus_orsoe/databases/dev_northern_tracks_muon_labels_v3/dev_northern_tracks_muon_labels_v3_part_1.db'
                 #'/mnt/scratch/rasmus_orsoe/databases/dev_northern_tracks_muon_labels_v3/dev_northern_tracks_muon_labels_v3_part_2.db']
                    #'/mnt/scratch/rasmus_orsoe/databases/dev_northeren_tracks_muon_labels_v3/data/dev_northern_tracks_muon_labels_v3_part_3.db',
                    #'/mnt/scratch/rasmus_orsoe/databases/dev_northeren_tracks_muon_labels_v3/data/dev_northern_tracks_muon_labels_v3_part_4.db']

    test_database = '/mnt/scratch/rasmus_orsoe/databases/dev_northern_tracks_muon_labels_v3/dev_northern_tracks_muon_labels_v3_part_1.db'

    # get selections:
    selections_uncut = pd.read_csv('/home/iwsatlas1/oersoe/phd/northern_tracks/energy_reconstruction/selections/dev_northern_tracks_muon_labels_v3_part_1_regression_selection.csv'),
                        #pd.read_csv('/home/iwsatlas1/oersoe/phd/northern_tracks/energy_reconstruction/selections/dev_northern_tracks_muon_labels_v3_part_2_regression_selection.csv')]
                    #pd.read_csv('/home/iwsatlas1/oersoe/phd/northern_tracks/energy_reconstruction/selections/dev_northern_tracks_muon_labels_v3_part_3_regression_selection.csv').sample(frac=1)['event_no'].ravel().tolist(),
                    #pd.read_csv('/home/iwsatlas1/oersoe/phd/northern_tracks/energy_reconstruction/selections/dev_northern_tracks_muon_labels_v3_part_4_regression_selection.csv').sample(frac=1)['event_no'].ravel().tolist()]

    #selections = []
    for selection in selections_uncut: 
    #    selections.append(selection.loc[selection['n_pulses']<= max_pulses,:].sample(frac=1)['event_no'].ravel().tolist())
        selections = selection.loc[selection['n_pulses']<= max_pulses,:].sample(frac=1)['event_no'].ravel().tolist()
    #selection = selection.sample(frac = 1)['event_no'].ravel().tolist()
    test_selection = pd.read_csv('/home/iwsatlas1/oersoe/phd/northern_tracks/energy_reconstruction/selections/dev_northern_tracks_muon_labels_v3_part_1_regression_selection.csv')
    test_selection = test_selection.loc[test_selection['n_pulses']<=max_pulses,:].sample(frac=1)['event_no'].ravel().tolist()
    
    

    training_dataloader, validation_dataloader = make_dataloaders(db=databases,
                                                                pulsemaps=pulsemap,
                                                                features=features,
                                                                truth=truth,
                                                                batch_size=batch_size,
                                                                selection=selections,
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
                                                                )
    test_dataloader =  make_dataloader(db=test_database,
                                        pulsemaps=pulsemap,
                                        features=features,
                                        truth=truth,
                                        batch_size=batch_size,
                                        selection=test_selection,
                                        num_workers=num_workers,
                                        persistent_workers=True,
                                        node_truth=node_truth,
                                        truth_table=truth_table,
                                        node_truth_table=node_truth_table,
                                        string_selection=None,
                                        index_column=index_column,
                                        labels=labels,
                                        shuffle=False,
                                        )

    run_name = f"dynedgeTITO_muon_entry_direction_reco_{target}_{pulsemap}_{n_epochs}e_p{patience}_max_pulses{max_pulses}_coarsening_{'none' if coarsening is None else coarsening.__class__.__name__}_rop_{rop}_size{layer_size_scale}_nb{nb_nearest_neighbours}"
    
    train_and_predict_on_validation_set(target=target,
                                        run_name=run_name,
                                        archive=archive,
                                        training_dataloader=training_dataloader,
                                        validation_dataloader=validation_dataloader,
                                        test_dataloader=test_dataloader,
                                        n_epochs=n_epochs,
                                        patience=patience,
                                        device=device,
                                        rop=rop,
                                        coarsening=coarsening,
                                        nb_nearest_neighbours=nb_nearest_neighbours,
                                        features_subset=features_subset,
                                        dyntrans_layer_sizes=dyntrans_layer_sizes,
                                        global_pooling_schemes=global_pooling_schemes,
                                        only_load_model=True,
                                        accumulate_grad_batches=accumulate_grad_batches,
                                        )
    

