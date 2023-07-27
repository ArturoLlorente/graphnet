import os
import numpy as np
import pandas as pd

import torch
from torch.optim.adam import Adam
from torch.utils.data import DataLoader

from pytorch_lightning.callbacks import ModelCheckpoint, GradientAccumulationScheduler, LearningRateMonitor

from pytorch_lightning.loggers import WandbLogger

from graphnet.data.dataset import EnsembleDataset
from graphnet.data.sqlite import SQLiteDataset
from graphnet.data.constants import FEATURES, TRUTH

from graphnet.models import StandardModelTito, Model
from graphnet.models.detector.icecube import IceCube86
from graphnet.models.gnn import DynEdgeTITO
from graphnet.models.coarsening import Coarsening
from graphnet.models.task.reconstruction import DirectionReconstructionWithKappaTITO

from graphnet.training.labels import Direction
from graphnet.training.loss_functions import VonMisesFisher3DLossTITO
from graphnet.training.callbacks import ProgressBar, PiecewiseLinearLR

from graphnet.utilities.config import save_model_config

import numpy as np
from typing import Dict, List, Optional, Union, Callable, Tuple
from torch_geometric.data import Batch, Data
from torch_geometric.nn import knn_graph


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

N_SPLIT = 2
R_SPLIT = 0.25
plist = []
for i in range(N_SPLIT):
    plist.append(R_SPLIT**i)
NPLIST = []
thisp = 0
for p in plist:
    thisp += p
    NPLIST.append(thisp)
NPLIST = [p/np.sum(plist) for p in NPLIST]
    
def collate_fn_tito(graphs: List[Data]) -> Batch:
    """Remove graphs with less than two DOM hits.

    Should not occur in "production.
    """
    graphs = [g for g in graphs if g.n_pulses > 1]
    graphs.sort(key=lambda x: x.n_pulses)
    exclude_keys = ['muon','muon_stopped','noise','neutrino','v_e','v_u','v_t','track','dbang','corsika','y','z','time','charge','auxiliary','ptr'] #TODO1
    batch_list = []
  
    for minp, maxp in zip([0]+NPLIST[:-1], NPLIST):
        min_idx = int(minp*len(graphs))
        max_idx = int(maxp*len(graphs))
        this_graphs = graphs[min_idx:max_idx]
        if len(this_graphs) > 0:
            this_batch = Batch.from_data_list(this_graphs, exclude_keys=exclude_keys)
            batch_list.append(this_batch)

    return batch_list

class GraphBuilder(Model):  # pylint: disable=too-few-public-methods
    """Base class for graph building."""

    pass

class KNNGraphBuilderTITO(GraphBuilder):  # pylint: disable=too-few-public-methods
    """Builds graph from the k-nearest neighbours."""

    @save_model_config
    def __init__(
        self,
        nb_nearest_neighbours: int,
        columns,
    ):
        """Construct `KNNGraphBuilder`."""
        # Base class constructor
        super().__init__()

        # Member variable(s)
        self._nb_nearest_neighbours = nb_nearest_neighbours
        self._columns = columns

    def forward(self, data: Data) -> Data:
        """Forward pass."""
        # Constructs the adjacency matrix from the raw, DOM-level data and
        # returns this matrix
        
        if data.edge_index is not None:
            self.info(
                "WARNING: GraphBuilder received graph with pre-existing "
                "structure. Will overwrite."
            )
            
        x = data.x
        x[:,3] = x[:,3]*0.1

        edge_index = knn_graph(
            x[:, self._columns]/1000,
            self._nb_nearest_neighbours,
            data.batch,
        ).to(self.device)
        
        data.edge_index = edge_index
        return data
    
    
def make_dataloaders(
    db_train: Union[List[str], str],
    db_val: Union[List[str], str],
    pulsemaps: Union[str, List[str]],
    features: List[str],
    truth: List[str],
    *,
    batch_size: int = 256,
    train_selection: Optional[List[int]],
    val_selection: Optional[List[int]],
    num_workers: int = 10,
    persistent_workers: bool = True,
    node_truth: List[str] = None,
    truth_table: str = 'northeren_tracks_muon_labels',
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
    if isinstance(db_train, list):
        assert len(train_selection) == len(db_train)
        train_datasets = []
        c = 0
        for db_idx, database in enumerate(db_train):
            train_datasets.append(SQLiteDataset(
                path=database,
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
            c +=1
            # adds custom labels to dataset
        if isinstance(labels, dict):
            for label in labels.keys():
                for train_dataset in train_datasets:
                    train_dataset.add_label(key=label, fn=labels[label])
                    
        train_dataset = EnsembleDataset(train_datasets)

        if isinstance(db_val, list):
            assert len(val_selection) == len(db_val)
            val_datasets = []
            for db_idx, database in enumerate(db_val):
                val_datasets.append(SQLiteDataset(
                    path=database,
                    pulsemaps=pulsemaps,
                    features=features,
                    truth=truth,
                    selection=val_selection[db_idx],
                    node_truth=node_truth,
                    truth_table=truth_table,
                    node_truth_table=node_truth_table,
                    string_selection=string_selection,
                    loss_weight_table=loss_weight_table,
                    loss_weight_column=loss_weight_column,
                    index_column=index_column,
                ))
                
            if isinstance(labels, dict):
                for label in labels.keys():
                    for val_dataset in val_datasets:
                        val_dataset.add_label(key=label, fn=labels[label])
                        
            if len(val_datasets) > 1:
                val_dataset = EnsembleDataset(val_datasets)
            else:
                val_dataset = val_datasets[0]

        common_kwargs = dict(
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=collate_fn_tito,
            persistent_workers=persistent_workers,
            prefetch_factor=2,
            )
            
        training_dataloader = DataLoader(
            train_dataset,
            shuffle=True,
            **common_kwargs,
        )
        validation_dataloader = DataLoader(
            val_dataset,
            shuffle=False,
            **common_kwargs,
        )
    else:
        assert 1 == 2, "dont use this code"


    return training_dataloader, validation_dataloader

def build_training_model(
    *,
    target: Union[str, List[str]],
    training_dataloader: DataLoader, 
    n_epochs: int = 40,
    device: int = None, 
    coarsening: Optional[Coarsening] = None,
    nb_nearest_neighbours: int = 6,
    columns_nb_nearest_neighbours: Optional[List[int]] = None,
    dyntrans_layer_sizes: Optional[List[Tuple[int, ...]]] = [(256, 256), (256, 256), (256, 256)],
    global_pooling_schemes: List[str] = ["max"],
    use_global_features: bool = False,
    use_post_processing_layers: bool = False,
    accumulate_grad_batches: int = 2,
    ):
    print(f"features: {features}")
    print(f"truth: {truth}")


    # Building model
    detector = IceCube86(graph_builder=KNNGraphBuilderTITO(nb_nearest_neighbours=nb_nearest_neighbours, columns=columns_nb_nearest_neighbours))
    
    gnn = DynEdgeTITO(
            nb_inputs=detector.nb_outputs,
            dyntrans_layer_sizes=dyntrans_layer_sizes,
            global_pooling_schemes=global_pooling_schemes,
            use_global_features=use_global_features,
            use_post_processing_layers=use_post_processing_layers,
            )
    
    task = DirectionReconstructionWithKappaTITO(
                        hidden_size=gnn.nb_outputs,
                        target_labels=target,
                        loss_function=VonMisesFisher3DLossTITO(),  
    )
    prediction_columns =["dir_x_pred", "dir_y_pred", "dir_z_pred", "dir_kappa_pred"]
    additional_attributes=['zenith', 'azimuth', "event_no", "energy"]



    scheduler_class= PiecewiseLinearLR
    scheduler_kwargs={
        "milestones": [
            0,
            10  * len(training_dataloader)//(len(device)*accumulate_grad_batches[0]),
            len(training_dataloader)*n_epochs//(len(device)*accumulate_grad_batches[0]*(20/10)),
            len(training_dataloader)*n_epochs//(len(device)*accumulate_grad_batches[0]),
        ],
        "factors": [1e-03, 1, 1, 1e-03],
        "verbose": False,
    }
    scheduler_config={
        'interval': 'step',
    }


    model = StandardModelTito(
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
    
   
    model.prediction_columns = prediction_columns
    model.additional_attributes = additional_attributes

    return model

# Main function call
if __name__ == "__main__":
    
    target = 'direction'
    archive = "/remote/ceph/user/l/llorente/train_DynEdge_northeren_tracks"
    batch_size = 750
    CUDA_DEVICE = 2
    device = [CUDA_DEVICE] if CUDA_DEVICE is not None else None
    n_epochs = 40
    num_workers = 100
    patience = 5
    pulsemap = 'InIceDSTPulses'
    node_truth_table = None
    node_truth = None
    truth_table = 'truth'
    index_column = 'event_no'
    max_pulses = 500
    labels = {'direction': Direction()}
    coarsening = None
    nb_nearest_neighbours = 6
    columns_nb_nearest_neighbours = [0,1,2]
    global_pooling_schemes = ["max"]
    dyntrans_layer_sizes = [(256, 256),
                            (256, 256),
                            (256, 256),
                            (256, 256)]
    use_global_features = True
    use_post_processing_layers = True
    accumulate_grad_batches = {0: 1}
    wandb = True
    num_database_files = 4

    # Configurations
    torch.multiprocessing.set_sharing_strategy('file_system')

    # Constants
    features = FEATURES.ICECUBE86
    truth = TRUTH.ICECUBE86
    
    train_databases = ['/mnt/scratch/rasmus_orsoe/databases/dev_northern_tracks_muon_labels_v3/dev_northern_tracks_muon_labels_v3_part_1.db',
                    '/mnt/scratch/rasmus_orsoe/databases/dev_northern_tracks_muon_labels_v3/dev_northern_tracks_muon_labels_v3_part_2.db',
                    '/mnt/scratch/rasmus_orsoe/databases/dev_northern_tracks_muon_labels_v3/dev_northern_tracks_muon_labels_v3_part_3.db',
                    '/mnt/scratch/rasmus_orsoe/databases/dev_northern_tracks_muon_labels_v3/dev_northern_tracks_muon_labels_v3_part_4.db']
    # get selections:
    train_selections = [pd.read_csv('/home/iwsatlas1/oersoe/phd/northern_tracks/energy_reconstruction/selections/dev_northern_tracks_muon_labels_v3_part_1_regression_selection.csv'),
                    pd.read_csv('/home/iwsatlas1/oersoe/phd/northern_tracks/energy_reconstruction/selections/dev_northern_tracks_muon_labels_v3_part_2_regression_selection.csv'),
                    pd.read_csv('/home/iwsatlas1/oersoe/phd/northern_tracks/energy_reconstruction/selections/dev_northern_tracks_muon_labels_v3_part_3_regression_selection.csv'),
                    pd.read_csv('/home/iwsatlas1/oersoe/phd/northern_tracks/energy_reconstruction/selections/dev_northern_tracks_muon_labels_v3_part_4_regression_selection.csv')]
    
    # get_list_of_databases:
    if num_database_files == 1:
        databases = train_databases[0]
        selection = train_selections[0]
        selections = selection.loc[selection['n_pulses']<= max_pulses,:].sample(frac=1)['event_no'].ravel().tolist()
    else:
        databases = train_databases[:num_database_files]
        train_selection = []
        for selection in train_selections[:num_database_files]:
            train_selection.append(selection.loc[selection['n_pulses']<= max_pulses,:].sample(frac=1)['event_no'].ravel().tolist())

    val_database = ['/mnt/scratch/rasmus_orsoe/databases/dev_northern_tracks_muon_labels_v3/dev_northern_tracks_muon_labels_v3_part_5.db']
    val_selection = pd.read_csv('/home/iwsatlas1/oersoe/phd/northern_tracks/energy_reconstruction/selections/dev_northern_tracks_muon_labels_v3_part_5_regression_selection.csv')
    val_selection = [val_selection.loc[val_selection['n_pulses']<=max_pulses,:].sample(frac=1)['event_no'].ravel().tolist()]

    (training_dataloader, 
     validation_dataloader) = make_dataloaders(db_train=databases,
                                        db_val=val_database,
                                        pulsemaps=pulsemap,
                                        features=features,
                                        truth=truth,
                                        batch_size=batch_size,
                                        train_selection=train_selection,
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
                                        )
     
    
    run_name = f"DynEdgeTITO_northeren_tracks_{target}_{pulsemap}_{n_epochs}e_p{patience}_max_pulses{max_pulses}_layersize{len(dyntrans_layer_sizes)}_nb{nb_nearest_neighbours}"
    
    model = build_training_model(target=target,
                                training_dataloader=training_dataloader,
                                n_epochs=n_epochs,
                                device=device,
                                coarsening=coarsening,
                                nb_nearest_neighbours=nb_nearest_neighbours,
                                columns_nb_nearest_neighbours=columns_nb_nearest_neighbours,
                                dyntrans_layer_sizes=dyntrans_layer_sizes,
                                global_pooling_schemes=global_pooling_schemes,
                                use_global_features=use_global_features,
                                use_post_processing_layers=use_post_processing_layers,
                                accumulate_grad_batches=accumulate_grad_batches,
                                )
    
    callbacks = [
        ModelCheckpoint(
            dirpath=archive+'/model_checkpoint_graphnet/',
            filename=run_name+'-{epoch:02d}-{val_loss:.6f}',
            monitor= 'val_loss',
            save_top_k = n_epochs,
            every_n_epochs = 1,
            save_weights_only=False,
        ),
        ProgressBar(),
        GradientAccumulationScheduler(scheduling=accumulate_grad_batches),
    ]
    
    if len(device) > 1:
        distribution_strategy = 'ddp'
    else:
        distribution_strategy = None
        
    if wandb:
        wandb_logger = WandbLogger(project='direction_reco_DynEdgeTITO_northeren_tracks', name=run_name)
        callbacks.append(LearningRateMonitor(logging_interval='step'))
        
    else:
        wandb_logger = None
        
    model.fit(
        training_dataloader,
        validation_dataloader,
        callbacks=callbacks,
        max_epochs=n_epochs,
        gpus=device,
        distribution_strategy=distribution_strategy,
        check_val_every_n_epoch=1,
        logger=wandb_logger,
    )
        
    try:
        model.save(os.path.join(archive, f"{run_name}.pth"))
        model.save_state_dict(os.path.join(archive, f"{run_name}_state_dict.pth"))
        print(f"Model saved to {archive}/{run_name}.pth")
    except:
        print(f"Model {archive}/{run_name}.pth could not be saved!")