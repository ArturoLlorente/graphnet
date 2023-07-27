import torch
import numpy as np
import pandas as pd
import random

import torch
from torch import Tensor
from torch.nn import ModuleList
from torch.optim.adam import Adam
from torch.utils.data import Dataset, DataLoader


from torch_geometric.data import Data

from pytorch_lightning.callbacks import ModelCheckpoint, GradientAccumulationScheduler, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger 

from graphnet.utilities.config import save_model_config

from graphnet.models.task.reconstruction import DirectionReconstructionWithKappaTITO
from graphnet.models import Model
from graphnet.models.task import Task


from graphnet.training.loss_functions import VonMisesFisher3DLossTITO, VonMisesFisher3DLoss
from graphnet.training.callbacks import ProgressBar, PiecewiseLinearLR

from typing import Optional, Union, List, Dict, Any

def convert_horizontal_to_direction(azimuth, zenith):
    dir_z = np.cos(zenith)
    dir_x = np.sin(zenith) * np.cos(azimuth)
    dir_y = np.sin(zenith) * np.sin(azimuth)
    return dir_x, dir_y, dir_z
    

class DatasetStacking(Dataset):
    def __init__(self,
                 target_columns: Union[List[str], str] = ["direction_x", "direction_y", "direction_z", "direction_kappa"],
                 model_preds: Union[pd.DataFrame, List[pd.DataFrame]] = None,
                 use_mid_features: bool = True,
                 ):
        
    
                
        if isinstance(model_preds, pd.DataFrame):
            self.model_preds = [model_preds]
        elif isinstance(model_preds, List):
            self.model_preds = model_preds
        else:
            assert False, "prediction file must be a DataFrame or a list of DataFrames"
                
        self.target_columns = target_columns
        self.use_mid_features = use_mid_features
        
            
        x = []
        y = []
        event_ids = []
        idx1 = []
        idx2 = []   
        for idx, model_pred in enumerate(self.model_preds):
            columns = self.target_columns
            if "direction_kappa" in model_pred.columns:
                model_pred["direction_kappa"]=np.log1p(model_pred["direction_kappa"])
            if "direction_kappa1" in model_pred.columns:
                model_pred["direction_kappa1"]=np.log1p(model_pred["direction_kappa1"])
                #columns = columns + ["direction_x1", "direction_y1", "direction_z1", "direction_kappa1"]
            if self.use_mid_features:
                columns = columns + ["idx"+str(i) for i in range(128)]
            x.append(model_pred[columns].reset_index(drop=True))
            y.append(np.stack(convert_horizontal_to_direction(model_pred["azimuth"], model_pred["zenith"])).T)
            event_ids.append(model_pred["event_id"].values)
            #idx1.append(np.full(len(model_pred),idx))
            #idx2.append(np.arange(len(model_pred)))
            
        self.X = pd.concat(x, axis=1).values
       
        #self.X = pd.concat(x, axis=1)
        self.Y = np.concatenate(y, axis=0)
        self.event_ids = np.concatenate(event_ids, axis=0)
        
            
    def __len__(self):
        return self.X.shape[0]
    
    def n_columns(self):
        return self.X.shape[1]
    
    def __getitem__(self, index):
        #idx1 = self.idx1[index]
        #idx2 = self.idx2[index]
        x = self.X[index]
        y = self.Y[index]
        event_id = self.event_ids[index]
        return x, y, event_id
        

class StandardModelStacking(Model):
    """Main class for standard models in graphnet.

    This class chains together the different elements of a complete GNN-based
    model (detector read-in, GNN architecture, and task-specific read-outs).
    """

    @save_model_config
    def __init__(
        self,
        *,
        tasks: Union[Task, List[Task]],
        n_input_features: int = 3, 
        hidden_size: Optional[int] = 512,
        dataset: DatasetStacking,
        optimizer_class: type = Adam,
        optimizer_kwargs: Optional[Dict] = None,
        scheduler_class: Optional[type] = None,
        scheduler_kwargs: Optional[Dict] = None,
        scheduler_config: Optional[Dict] = None,
    ) -> None:
        """Construct `StandardModel`."""
        # Base class constructor
        super().__init__()

        # Check(s)
        if isinstance(tasks, Task):
            tasks = [tasks]
        assert isinstance(tasks, (list, tuple))
        assert all(isinstance(task, Task) for task in tasks)

        # Member variable(s)
        self._tasks = ModuleList(tasks)
        self._optimizer_class = optimizer_class
        self._optimizer_kwargs = optimizer_kwargs or dict()
        self._scheduler_class = scheduler_class
        self._scheduler_kwargs = scheduler_kwargs or dict()
        self._scheduler_config = scheduler_config or dict()
        self.n_input_features = n_input_features
        self._dataset = dataset
        
        mlp_layers = []
        layer_sizes = [n_input_features, hidden_size, hidden_size, hidden_size] # todo1
        for nb_in, nb_out in zip(layer_sizes[:-1], layer_sizes[1:]):
            mlp_layers.append(torch.nn.Linear(nb_in, nb_out))
            mlp_layers.append(torch.nn.LeakyReLU())
            mlp_layers.append(torch.nn.Dropout(0.0))

        self._mlp = torch.nn.Sequential(*mlp_layers)

            
    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure the model's optimizer(s)."""
        optimizer = self._optimizer_class(
            self.parameters(), **self._optimizer_kwargs
        )
        config = {
            "optimizer": optimizer,
        }
        if self._scheduler_class is not None:
            scheduler = self._scheduler_class(
                optimizer, **self._scheduler_kwargs
            )
            config.update(
                {
                    "lr_scheduler": {
                        "scheduler": scheduler,
                        **self._scheduler_config,
                    },
                }
            )
        return config

    def forward(self, x):
        x = x.float()
        x = self._mlp(x)
        x = [task(x) for task in self._tasks]
        return x

    def training_step(self, xye, idx) -> Tensor:
        """Perform training step."""
        x,y,event_ids = xye
        preds = self(x)
        batch = Data(x=x, direction=y)
        vlosses = self._tasks[1].compute_loss(preds[1], batch)
        vloss = torch.sum(vlosses)
        
        tlosses = self._tasks[0].compute_loss(preds[0], batch)
        tloss = torch.sum(tlosses)
        
        if self.current_epoch == 0:
            vloss_weight = 1
        else:
            current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
            vloss_weight = current_lr / 1e-03

        loss = vloss*vloss_weight + tloss

        return {"loss": loss, 'vloss': vloss, 'tloss': tloss, 'vloss_weight': vloss_weight}

    def validation_step(self, xye, idx) -> Tensor:
        """Perform validation step."""
        x,y,event_ids = xye
        preds = self(x)
        batch = Data(x=x, direction=y)
        vlosses = self._tasks[1].compute_loss(preds[1], batch)
        vloss = torch.sum(vlosses)
        
        tlosses = self._tasks[0].compute_loss(preds[0], batch)
        tloss = torch.sum(tlosses)
        
        current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        vloss_weight = current_lr / 1e-03

        loss = vloss*vloss_weight + tloss
        return {"loss": loss, 'vloss': vloss, 'tloss': tloss, 'vloss_weight': vloss_weight}

    def inference(self) -> None:
        """Activate inference mode."""
        for task in self._tasks:
            task.inference()

    def train(self, mode: bool = True) -> "Model":
        """Deactivate inference mode."""
        super().train(mode)
        if mode:
            for task in self._tasks:
                task.train_eval()
        return self

    def predict(
        self,
        dataloader: DataLoader,
        gpus: Optional[Union[List[int], int]] = None,
        distribution_strategy: Optional[str] = None,
    ) -> List[Tensor]:
        """Return predictions for `dataloader`."""
        self.inference()
        return super().predict(
            dataloader=dataloader,
            gpus=gpus,
            distribution_strategy=distribution_strategy,
        )
    
    
    
    
def build_model_stacking(
    *,
    dataset: Dataset = None,
    hidden_size: Optional[int] = 512,
    scheduler_class: Optional[type] = None,
    scheduler_kwargs: Optional[dict] = None,
    ):

    
    task = DirectionReconstructionWithKappaTITO(
        hidden_size=hidden_size,
        target_labels="direction",
        loss_function=VonMisesFisher3DLoss(),
    )
        
    task2 = DirectionReconstructionWithKappaTITO(
                        hidden_size=hidden_size,
                        target_labels="direction",
                        loss_function=VonMisesFisher3DLossTITO(),
    )
    prediction_columns =['dir_x_pred', 'dir_y_pred', 'dir_z_pred', 'dir_kappa_pred']
    additional_attributes=['zenith', 'azimuth', 'event_id']

    scheduler_config={
        "interval": "step",
    }


    model = StandardModelStacking(
        tasks=[task2, task],
        n_input_features=dataset.n_columns(),
        hidden_size=hidden_size,
        dataset=dataset,
        optimizer_class=Adam,
        optimizer_kwargs={'lr': 1e-03, 'eps': 1e-03},
        scheduler_class= scheduler_class,
        scheduler_kwargs=scheduler_kwargs,
        scheduler_config=scheduler_config,
     )
    model.prediction_columns = prediction_columns
    model.additional_attributes = additional_attributes
    
    return model

seed = 42
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(seed=seed)

prediction_df = []
for file_idx in range(6):
    prediction_df.append(pd.read_csv(f"/remote/ceph/user/l/llorente/prediction_models/graphnet_predictions/model{file_idx+1}_batch1-55_graphnet.csv"))

train_dataset = DatasetStacking(target_columns=['direction_x', 'direction_y', 'direction_z', 'direction_kappa', 'direction_x1', 'direction_y1', 'direction_z1', 'direction_kappa1'],
                                   model_preds=prediction_df)
val_dataset = DatasetStacking(target_columns=['direction_x', 'direction_y', 'direction_z', 'direction_kappa', 'direction_x1', 'direction_y1', 'direction_z1', 'direction_kappa1'],
                                     model_preds=prediction_df)
train_dataloader = DataLoader(train_dataset, batch_size=1000, shuffle=True, num_workers=64)
val_dataloader = DataLoader(val_dataset, batch_size=1000, shuffle=False, num_workers=64)



## Start training
dataloader = "dummy"
device = [3]
#train_dataset = "dummy"
hidden_size = 512
accumulate_grad_batches = {0: 2}

scheduler_kwargs={
    "milestones": [
        0,
        len(dataloader)//(len(device)*accumulate_grad_batches[0]*100),
        len(dataloader)//(len(device)*accumulate_grad_batches[0]*2),
        len(dataloader)//(len(device)*accumulate_grad_batches[0]),                
    ],
    "factors": [1e-03, 1, 1, 1e-04],
    "verbose": False,
}

model  = build_model_stacking(
    dataset=train_dataset,
    hidden_size=512,
    scheduler_class=PiecewiseLinearLR,
    scheduler_kwargs=scheduler_kwargs,
    )

runName = 'graphnet_stacking'
callbacks = [
    ModelCheckpoint(
        dirpath='/remote/ceph/user/l/llorente/training_1e_test/',
        filename=runName+'-{epoch:02d}-{trn_tloss:.6f}',
        save_weights_only=False,
    ),
    ProgressBar(),
]

model.fit(train_dataloader=train_dataloader,
          val_dataloader=val_dataloader,
          callbacks=callbacks,
          max_epochs=4,
          gpus=device,)

from tqdm.auto import tqdm
CKPT = f'/remote/ceph/user/l/llorente/tito_solution/model_graphnet/stacking-6models-last.pth'
state_dict =  torch.load(CKPT, torch.device('cpu'))

if 'state_dict' in state_dict.keys():
    state_dict = state_dict['state_dict']
model.load_state_dict(state_dict)
USE_ALL_FEA_IN_PRED=True
validateMode=True

event_ids = []
zenith = []
azimuth = []
preds = []
print('start predict')

real_x = []
real_y = []
real_z = []
with torch.no_grad():
    model.eval()
    model.to(f'cuda:{device[0]}')
    for batch in tqdm(val_dataloader):
        
        pred = model(batch[0].to(f'cuda:{device[0]}'))
        #preds.append(pred[0])
        if USE_ALL_FEA_IN_PRED:
            preds.append(torch.cat(pred, axis=-1))
        else:
            preds.append(pred[0])
        event_ids.append(batch[2])
        if validateMode:
            real_x.append(batch[1][:,0])
            real_y.append(batch[1][:,1])
            real_z.append(batch[1][:,2])
preds = torch.cat(preds).to('cpu').detach().numpy()
#zenith = torch.cat(zenith).to('cpu').numpy()
#azimuth = torch.cat(azimuth).to('cpu').numpy()
real_x = torch.cat(real_x).to('cpu').numpy()
real_y = torch.cat(real_y).to('cpu').numpy()
real_z = torch.cat(real_z).to('cpu').numpy()
#results = pd.DataFrame(preds, columns=model.prediction_columns)
if USE_ALL_FEA_IN_PRED:
    if preds.shape[1] == 128+8:
        columns = ['direction_x','direction_y','direction_z','direction_kappa1','direction_x1','direction_y1','direction_z1','direction_kappa'] + [f'idx{i}' for i in range(128)]
    else:
        columns = ['direction_x','direction_y','direction_z','direction_kappa'] + [f'idx{i}' for i in range(128)]
else:
    columns=model.prediction_columns
results = pd.DataFrame(preds, columns=columns)
results['event_id'] = np.concatenate(event_ids)
if validateMode:
    #results['zenith'] = zenith#np.concatenate(zenith)
    #results['azimuth'] = azimuth#np.concatenate(azimuth)
    results['real_x'] = real_x
    results['real_y'] = real_y
    results['real_z'] = real_z
    
results.sort_values('event_id')
results.to_csv(f'/remote/ceph/user/l/llorente/prediction_models/stacking_tito/model_checkpoint_graphnet/predictions.csv')