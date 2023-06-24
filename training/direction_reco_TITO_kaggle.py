import os
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import torch
from torch.optim.adam import Adam

from pytorch_lightning.callbacks import ModelCheckpoint, GradientAccumulationScheduler, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger 

from pytorch_lightning.profiler import PyTorchProfiler

from graphnet.data.dataset import Dataset, EnsembleDataset
from graphnet.data.sqlite import SQLiteDataset
from graphnet.data.constants import FEATURES, TRUTH

from graphnet.models import StandardModel, StandardModelTito, StandardModelSoftTito, StandardModel2
from graphnet.models.detector.detector import Detector
from graphnet.models.graph_builders import KNNGraphBuilder
from graphnet.models.gnn import DynEdgeTITO
from graphnet.models.task.reconstruction import DirectionReconstructionWithKappa

from graphnet.training.labels import Direction
from graphnet.training.loss_functions import VonMisesFisher3DLossTITO, VonMisesFisher3DLoss
from graphnet.training.callbacks import ProgressBar, PiecewiseLinearLR
from graphnet.training.utils import collate_fn

from graphnet.utilities.logging import Logger

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

class IceCubeKaggleTITO(Detector):
    """`Detector` class for Kaggle Competition."""

    # Implementing abstract class attribute
    features = FEATURES.KAGGLE

    def _forward(self, data: Data) -> Data:
        """Ingest data, build graph, and preprocess features.

        Args:
            data: Input graph data.

        Returns:
            Connected and preprocessed graph data.
        """
        # Check(s)
        self._validate_features(data)

        # Preprocessing
        data.x[:, 0] /= 500.0  # x
        data.x[:, 1] /= 500.0  # y
        data.x[:, 2] /= 500.0  # z
        data.x[:, 3] = (data.x[:, 3] - 1.0e04) / (500.0*0.23)  # 0.23 is speed of light in ice
        data.x[:, 4] = torch.log10(data.x[:, 4]) / 3.0  # charge

        return data
    
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


import random
from torch_geometric.nn import knn_graph
from graphnet.utilities.config import save_model_config
from graphnet.models import Model


class GraphBuilder(Model):  # pylint: disable=too-few-public-methods
    """Base class for graph building."""

    pass

TIME_PARAM_FOR_DIST = 1/10

class KNNGraphBuilderTITO(GraphBuilder):  # pylint: disable=too-few-public-methods
    """Builds graph from the k-nearest neighbours."""

    @save_model_config
    def __init__(
        self,
        nb_nearest_neighbours,
        columns,
    ):
        """Construct `KNNGraphBuilder`."""
        # Base class constructor
        super().__init__()

        # Member variable(s)
        assert len(nb_nearest_neighbours) == len(columns)
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
        edge_index_list = []
        x = data.x
        x[:,3] = x[:,3]*TIME_PARAM_FOR_DIST
        for idx in range(len(self._nb_nearest_neighbours)):
            nb_nearest_neighbour = self._nb_nearest_neighbours[idx]
            if type(nb_nearest_neighbour) == str:
                nb_nearest_neighbour_min, nb_nearest_neighbour_max = nb_nearest_neighbour.split('-')
                nb_nearest_neighbour = torch.randint(int(nb_nearest_neighbour_min), int(nb_nearest_neighbour_max), (1,))[0]
                #print('nb_nearest_neighbour', nb_nearest_neighbour)
            elif type(nb_nearest_neighbour) == list:
                nb_nearest_neighbour = random.choice(nb_nearest_neighbour)
                #print('nb_nearest_neighbour', nb_nearest_neighbour)
            edge_index = knn_graph(
                x[:, self._columns[idx]],
                nb_nearest_neighbour,
                data.batch,
            ).to(self.device)
            edge_index_list.append(edge_index)
        x[:,3] = x[:,3]/TIME_PARAM_FOR_DIST

        data.edge_index = edge_index_list
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
    seed: Optional[int] = None,
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
                path = database,
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
        #train_dataset = SQLiteDataset(
        #        path = database,
        #        pulsemaps=pulsemaps,
        #        features=features,
        #        truth=truth,
        #        selection=train_selection[db_idx],
        #        node_truth=node_truth,
        #        truth_table=truth_table,
        #        node_truth_table=node_truth_table,
        #        string_selection=string_selection,
        #        loss_weight_table=loss_weight_table,
        #        loss_weight_column=loss_weight_column,
        #        index_column=index_column,
        #    )
        #
        #if isinstance(labels, dict):
        #    for label in labels.keys():
        #        train_dataset.add_label(key=label, fn=labels[label])
        training_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=collate_fn_tito,
            persistent_workers=persistent_workers,
            prefetch_factor=2,
            generator=torch.Generator().manual_seed(seed),
        )
            
                        
    if isinstance(db_val, str):
        
        val_dataset = SQLiteDataset(
                        path = db_val,
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
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn_tito,
            persistent_workers=persistent_workers,
            prefetch_factor=2,
            generator=torch.Generator().manual_seed(seed),
        )
        
                    
    return training_dataloader, validation_dataloader, train_dataset, val_dataset


def build_model(
    *,
    target: Union[str, List[str]],
    device: Union[int, list] = None, 
    features_subset: Optional[slice] = slice(0, 4),
    dyntrans_layer_sizes: Optional[List[Tuple[int, ...]]] = [(256, 256), (256, 256), (256, 256)],
    global_pooling_schemes: Optional[List[str]] = ["max"],
    wandb: bool = False,
    only_load_model: bool = False,
    accumulate_grad_batches: dict = {0: 1},
    use_global_features: bool = True,
    use_post_processing_layers: bool = True,
    scheduler_class: Optional[type] = None,
    scheduler_kwargs: Optional[dict] = None,
    dataset: Dataset = None,
    ):
    print(f"features: {features}")
    print(f"truth: {truth}")

    

    # Initialise Weights & Biases (W&B) run
    if wandb:
        logger = Logger()
        # Make sure W&B output directory exists
        wandb_dir = "./wandb/"
        os.makedirs(wandb_dir, exist_ok=True)
        wandb_logger = WandbLogger(
            project="direction_reco_DynEdgeTito",
            save_dir=wandb_dir,
            log_model=True,
        )
        logger.info(f"features: {features}")
        logger.info(f"truth: {truth}")



    # Building model
    detector = IceCubeKaggleTITO(
        graph_builder=KNNGraphBuilder(nb_nearest_neighbours=6, columns=[0,1,2])
    )
    
    gnn = DynEdgeTITO(
            nb_inputs=detector.nb_outputs,
            features_subset=features_subset,
            dyntrans_layer_sizes=dyntrans_layer_sizes,
            global_pooling_schemes=global_pooling_schemes,
            use_global_features=use_global_features,
            use_post_processing_layers=use_post_processing_layers,
            )

    task = DirectionReconstructionWithKappa(
                        hidden_size=gnn.nb_outputs,
                        target_labels="direction",
                        loss_function=VonMisesFisher3DLossTITO(),
    )
    prediction_columns =['dir_x_pred', 'dir_y_pred', 'dir_z_pred', 'dir_kappa_pred']
    additional_attributes=['zenith', 'azimuth', 'event_id', 'energy']

    scheduler_config={
        "interval": "step",
    }


    model = StandardModel2(
        detector=detector,
        gnn=gnn,
        tasks=[task],
        optimizer_class=Adam,
        optimizer_kwargs={'lr': 1e-03, 'eps': 1e-03},
        dataset=dataset,
        scheduler_class= scheduler_class,
        scheduler_kwargs=scheduler_kwargs,
        scheduler_config=scheduler_config,
     )
    model.prediction_columns = prediction_columns
    model.additional_attributes = additional_attributes
    
    return model

# Main function call
if __name__ == "__main__":
    
    target = ['direction']
    archive = "/remote/ceph/user/l/llorente/training_1e_test/graphnet_model"
    weight_column_name = None 
    weight_table_name =  None
    batch_size = 1000
    n_round = 1
    #CUDA_DEVICE = 0
    #device = [CUDA_DEVICE] if CUDA_DEVICE is not None else None
    device = [1,2]
    num_workers = 128
    pulsemap = 'pulse_table'
    node_truth_table = None
    node_truth = None
    truth_table = 'meta_table'
    index_column = 'event_id'
    max_pulses = 2000
    labels = {'direction': Direction()}
    global_pooling_schemes = ["max"]
    features_subset = slice(0, 3)
    dyntrans_layer_sizes = [(256, 256),
                            (256, 256),
                            (256, 256),
                            (256, 256)]
    accumulate_grad_batches = {0: 1}
    wandb = False
    only_load_model = False
    num_database_files = 1
    use_global_features = True
    use_post_processing_layers = True
    train_max_pulses = 200
    val_max_pulses = 200
    scheduler_class = PiecewiseLinearLR
    seed = 42
    

    
    def set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    set_seed(seed=seed)

    # Configurations
    torch.multiprocessing.set_sharing_strategy('file_system')

    # Constants
    features = FEATURES.KAGGLE
    truth = TRUTH.KAGGLE
    
    
    database_path = '/mnt/scratch/kaggle_datasets/databases_merged/'
    all_databases = []
    selection_files = []
    for i in range(num_database_files):
        all_databases.append(database_path + 'batch_%02d.db'%(i+1))
        selection_files.append(pd.read_csv('/remote/ceph/user/l/llorente/kaggle/selection_files/pulse_information_%02d.csv'%(i+1)))
    
    #all_databases = ['/remote/ceph/user/l/llorente/kaggle/databases_testing/batch_test_1.db',
    #                 '/remote/ceph/user/l/llorente/kaggle/databases_testing/batch_test_2.db']
                     #'/remote/ceph/user/l/llorente/kaggle/databases_testing/batch_test_3.db',]
    #selection_files = [pd.read_csv('/remote/ceph/user/l/llorente/kaggle/databases_testing/batch_test_1_pulses.csv'),
    #                   pd.read_csv('/remote/ceph/user/l/llorente/kaggle/databases_testing/batch_test_2_pulses.csv')]
                       #pd.read_csv('/remote/ceph/user/l/llorente/kaggle/databases_testing/batch_test_3_pulses.csv'),]
    
    # get_list_of_databases:
    selections = []
    for selection in selection_files:
        selections.append(selection.loc[selection['n_pulses'] < train_max_pulses,:]['event_id'].ravel().tolist())

    val_database = '/remote/ceph/user/l/llorente/kaggle/databases_merged/batch_val.db'
    val_selection_file = pd.read_csv('/remote/ceph/user/l/llorente/kaggle/selection_files/pulse_information_val.csv')
    #val_database = '/remote/ceph/user/l/llorente/kaggle/databases_testing/batch_test_val.db'
    #val_selection_file = pd.read_csv('/remote/ceph/user/l/llorente/kaggle/databases_testing/batch_test_val_pulses.csv')
    val_selection = val_selection_file.loc[val_selection_file['n_pulses']<val_max_pulses,:].sample(frac=1)['event_id'].ravel().tolist()

    print("Loading databases")

    (training_dataloader, 
     validation_dataloader,
     train_dataset, 
     val_dataset) = make_dataloaders(db_train=all_databases,
                                        db_val=val_database,
                                        pulsemaps=pulsemap,
                                        features=features,
                                        truth=truth,
                                        batch_size=batch_size,
                                        train_selection=selections,
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
                                        seed=seed,
                                        )

    
    run_name = f"dynedgeTITO__direction_reco_{n_round}e_train_max_pulses{train_max_pulses}_val_max_pulses{val_max_pulses}_layersize{len(dyntrans_layer_sizes)}_use_GG_{use_global_features}_use_PP_{use_post_processing_layers}_features_subset_{features_subset}_batch_{batch_size}_nround{n_round}_testing_db"
    
    scheduler_kwargs={
        "milestones": [
            0,
            10  * len(training_dataloader)//(len(device)*accumulate_grad_batches[0]*num_database_files*55*n_round),
            len(training_dataloader)//(len(device)*accumulate_grad_batches[0]*(20/10)),
            len(training_dataloader)//(len(device)*accumulate_grad_batches[0]),                
        ],
        "factors": [1e-03, 1, 1, 1e-03],
        "verbose": False,
    }

    print(f"milestones are: {scheduler_kwargs['milestones']}")
    
    print("Starting training")
    model = build_model(target=target,
                          device=device,
                          features_subset=features_subset,
                          dyntrans_layer_sizes=dyntrans_layer_sizes,
                          global_pooling_schemes=global_pooling_schemes,
                          only_load_model=only_load_model,
                          accumulate_grad_batches=accumulate_grad_batches,
                          wandb=wandb,
                          scheduler_class=scheduler_class,
                          scheduler_kwargs=scheduler_kwargs,
                          dataset=train_dataset,
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
        LearningRateMonitor(logging_interval='step'),
    ]

    if len(device) > 1:
        distribution_strategy = 'ddp'
    else:
        distribution_strategy = None

##    model.fit(
##        training_dataloader,
##        validation_dataloader,
##        callbacks=callbacks,
##        max_epochs=n_round,
##        gpus=device,
##        distribution_strategy=distribution_strategy,
##        check_val_every_n_epoch=1,
##        precision=16,
##        logger=WandbLogger(project='train_6batch_1e_graphnet', name=run_name),
##        profiler=PyTorchProfiler( output_filename='profiler_results.txt', trace_every_n_steps=1),
##        #reload_dataloaders_every_n_epochs=1,
##    )
        
    #model.save(os.path.join(archive, f"{run_name}.pth"))
    #model.save_state_dict(os.path.join(archive, f"{run_name}_state_dict.pth"))
    #print(f"Model saved to {archive}/{run_name}.pth")