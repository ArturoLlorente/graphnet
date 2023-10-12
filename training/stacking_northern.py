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

from graphnet.models.task.reconstruction import DirectionReconstructionWithKappa
from graphnet.models import Model, StandardModelStacking
from graphnet.models.task import Task


from graphnet.training.loss_functions import VonMisesFisher3DLoss
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
        event_nos = []
        idx1 = []
        idx2 = []   
        for idx, model_pred in enumerate(self.model_preds):
            columns = self.target_columns
            if "direction_kappa" in model_pred.columns:
                model_pred["direction_kappa"]=np.log1p(model_pred["direction_kappa"])
            if self.use_mid_features:
                columns = columns + ["idx"+str(i) for i in range(128)]
            x.append(model_pred[columns].reset_index(drop=True))
            y.append(np.stack(convert_horizontal_to_direction(model_pred["azimuth"], model_pred["zenith"])).T)
            event_nos.append(model_pred["event_no"].values)
            #idx1.append(np.full(len(model_pred),idx))
            #idx2.append(np.arange(len(model_pred)))
            
        self.X = pd.concat(x, axis=1).values
        self.Y = np.concatenate(y, axis=0)
        self.event_nos = np.concatenate(event_nos, axis=0)
        
            
    def __len__(self):
        return self.X.shape[0]
    
    def n_columns(self):
        return self.X.shape[1]
    
    def __getitem__(self, index):
        #idx1 = self.idx1[index]
        #idx2 = self.idx2[index]
        x = self.X[index]
        y = self.Y[index]
        event_no = self.event_nos[index]
        return x, y, event_no
        
def build_model_stacking(
    *,
    dataset: Dataset = None,
    hidden_size: Optional[int] = 512,
    scheduler_class: Optional[type] = None,
    accumulate_grad_batches: Optional[dict] = None,
    ):

    
    task = DirectionReconstructionWithKappa(
        hidden_size=hidden_size,
        target_labels="direction",
        loss_function=VonMisesFisher3DLoss(),
    )
        
    prediction_columns =['dir_x_pred', 'dir_y_pred', 'dir_z_pred', 'dir_kappa_pred']
    additional_attributes=['zenith', 'azimuth', 'event_no']

    scheduler_config={
        "interval": "step",
    }
    scheduler_kwargs={
        "milestones": [
            0,
            len(train_dataset)//(len(device)*accumulate_grad_batches[0]*100),
            len(train_dataset)//(len(device)*accumulate_grad_batches[0]*2),
            len(train_dataset)//(len(device)*accumulate_grad_batches[0]),                
        ],
        "factors": [1e-03, 1, 1, 1e-04],
        "verbose": False,
    }

    model = StandardModelStacking(
        tasks=[task],
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

model_ids = [1,3]
for model_id in model_ids:
    prediction_df = [pd.read_csv(f'/remote/ceph/user/l/llorente/train_DynEdgeTITO_northern_Oct23/prediction_models/model{model_id}_northern_tracks_graphnet.csv')]

train_dataset = DatasetStacking(target_columns=['direction_x', 'direction_y', 'direction_z', 'direction_kappa'],
                                    model_preds=prediction_df)
val_dataset = DatasetStacking(target_columns=['direction_x', 'direction_y', 'direction_z', 'direction_kappa'],
                                    model_preds=prediction_df)
train_dataloader = DataLoader(train_dataset, batch_size=2000, shuffle=True, num_workers=16)
val_dataloader = DataLoader(val_dataset, batch_size=2000, shuffle=False, num_workers=16)



## Start training
device = [2]
hidden_size = 132*len(model_ids)
accumulate_grad_batches = {0: 2}

model  = build_model_stacking(
    dataset=train_dataset,
    hidden_size=hidden_size,
    scheduler_class=PiecewiseLinearLR,
    accumulate_grad_batches=accumulate_grad_batches,
    )

runName = 'northern_graphnet_stacking'
callbacks = [
    ModelCheckpoint(
        dirpath='/remote/ceph/user/l/llorente/train_DynEdgeTITO_northern_Oct23/model_stacking',
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

model.save('/remote/ceph/user/l/llorente/train_DynEdgeTITO_northern_Oct23/model_stacking/'+runName+'-last.pth')
model.save_state_dict('/remote/ceph/user/l/llorente/train_DynEdgeTITO_northern_Oct23/model_stacking/'+ runName+'-last_state_dict.pth')

#from tqdm.auto import tqdm
#CKPT = f'/remote/ceph/user/l/llorente/tito_solution/model_graphnet/stacking-6models-last.pth'
#state_dict =  torch.load(CKPT, torch.device('cpu'))
#
#if 'state_dict' in state_dict.keys():
#    state_dict = state_dict['state_dict']
#model.load_state_dict(state_dict)
#USE_ALL_FEA_IN_PRED=True
#validateMode=True
#
#event_nos = []
#zenith = []
#azimuth = []
#preds = []
#print('start predict')
#
#real_x = []
#real_y = []
#real_z = []
#with torch.no_grad():
#    model.eval()
#    model.to(f'cuda:{device[0]}')
#    for batch in tqdm(val_dataloader):
#        
#        pred = model(batch[0].to(f'cuda:{device[0]}'))
#        #preds.append(pred[0])
#        if USE_ALL_FEA_IN_PRED:
#            preds.append(torch.cat(pred, axis=-1))
#        else:
#            preds.append(pred[0])
#        event_nos.append(batch[2])
#        if validateMode:
#            real_x.append(batch[1][:,0])
#            real_y.append(batch[1][:,1])
#            real_z.append(batch[1][:,2])
#preds = torch.cat(preds).to('cpu').detach().numpy()
##zenith = torch.cat(zenith).to('cpu').numpy()
##azimuth = torch.cat(azimuth).to('cpu').numpy()
#real_x = torch.cat(real_x).to('cpu').numpy()
#real_y = torch.cat(real_y).to('cpu').numpy()
#real_z = torch.cat(real_z).to('cpu').numpy()
##results = pd.DataFrame(preds, columns=model.prediction_columns)
#if USE_ALL_FEA_IN_PRED:
#    if preds.shape[1] == 128+8:
#        columns = ['direction_x','direction_y','direction_z','direction_kappa1','direction_x1','direction_y1','direction_z1','direction_kappa'] + [f'idx{i}' for i in range(128)]
#    else:
#        columns = ['direction_x','direction_y','direction_z','direction_kappa'] + [f'idx{i}' for i in range(128)]
#else:
#    columns=model.prediction_columns
#results = pd.DataFrame(preds, columns=columns)
#results['event_no'] = np.concatenate(event_nos)
#if validateMode:
#    #results['zenith'] = zenith#np.concatenate(zenith)
#    #results['azimuth'] = azimuth#np.concatenate(azimuth)
#    results['real_x'] = real_x
#    results['real_y'] = real_y
#    results['real_z'] = real_z
#    
#results.sort_values('event_no')
#results.to_csv(f'/remote/ceph/user/l/llorente/prediction_models/stacking_tito/model_checkpoint_graphnet/predictions.csv')