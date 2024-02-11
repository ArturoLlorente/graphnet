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

from graphnet.models.task import Task
class DirectionReconstructionWithKappaTITO(Task):
    """Reconstructs direction with kappa from the 3D-vMF distribution."""

    # Requires three features: untransformed points in (x,y,z)-space.
    default_target_labels = [
        "direction"
    ]  # contains dir_x, dir_y, dir_z see https://github.com/graphnet-team/graphnet/blob/95309556cfd46a4046bc4bd7609888aab649e295/src/graphnet/training/labels.py#L29
    default_prediction_labels = [
        "dir_x_pred",
        "dir_y_pred",
        "dir_z_pred",
        "direction_kappa",
    ]
    nb_inputs = 3

    def _forward(self, x: Tensor) -> Tensor:
        # Transform outputs to angle and prepare prediction
        #kappa = torch.linalg.vector_norm(x, dim=1) + eps_like(x)
        kappa = torch.linalg.vector_norm(x, dim=1)# + eps_like(x)
        kappa = torch.clamp(kappa, min=torch.finfo(x.dtype).eps)
        vec_x = x[:, 0] / kappa
        vec_y = x[:, 1] / kappa
        vec_z = x[:, 2] / kappa
        return torch.stack((vec_x, vec_y, vec_z, kappa), dim=1)
    
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
        #self.model_preds[2] = self.model_preds[2].merge(self.model_preds[1]["event_no"], on="event_no", how="inner")
        
        x = []
        for model_pred in self.model_preds:
            columns = self.target_columns
            if "direction_kappa" in model_pred.columns:
                model_pred["direction_kappa"]=np.log1p(model_pred["direction_kappa"])
            if self.use_mid_features:
                columns = columns + ["idx"+str(i) for i in range(128)]
            x.append(model_pred[columns].reset_index(drop=True))
            
            
        self.X = pd.concat(x, axis=1).values
        self.Y = np.stack(convert_horizontal_to_direction(model_pred["azimuth"], model_pred["zenith"])).T
        self.event_nos =  model_pred["event_no"].values
        
    def __len__(self):
        return self.X.shape[0]
    
    def n_columns(self):
        return self.X.shape[1]
    
    def __getitem__(self, index):
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
    scheduler_kwargs: Optional[dict] = None,
    ):

    
    task = DirectionReconstructionWithKappaTITO(
        hidden_size=hidden_size,
        target_labels="direction",
        loss_function=VonMisesFisher3DLoss(),
    )

    task2 = DirectionReconstructionWithKappa(
        hidden_size=hidden_size,
        target_labels="direction",
        loss_function=VonMisesFisher3DLoss(),
    )
        
    prediction_columns =['dir_x_pred', 'dir_y_pred', 'dir_z_pred', 'dir_kappa_pred']
    additional_attributes=['zenith', 'azimuth', 'event_no', 'energy']

    scheduler_config={
        "interval": "step",
    }

    model = StandardModelStacking(
        tasks=[task, task2],
        n_input_features=dataset.n_columns(),
        hidden_size=hidden_size,
        dataset=dataset,
        optimizer_class=Adam,
        optimizer_kwargs={'lr': 1e-03, 'eps': 1e-03},
        scheduler_class=scheduler_class,
        scheduler_kwargs=scheduler_kwargs,
        scheduler_config=scheduler_config,
     )
    model.prediction_columns = prediction_columns
    model.additional_attributes = additional_attributes
    
    return model

if __name__ == "__main__":

    model_names = ['model1_northern_load_tito',
                   'model2_northern_load_tito',
                   'model3_northern_load_tito', 
                   'model4_northern_load_tito',
                   'model5_northern_load_tito',
                   'model6_northern_load_tito',]
    run_names = ''
    prediction_df = []
    for model_name in model_names:
        prediction_df.append(pd.read_csv(f'/remote/ceph/user/l/llorente/train_DynEdgeTITO_northern_Oct23/prediction_models/{model_name}.csv'))
        run_names = run_names + model_name[:11] + '_'

    device = [2]
    hidden_size = 132*len(model_names)
    accumulate_grad_batches = {0: 1}
    max_epochs = 20
    batch_size = 5000
    num_workers = 16

    INFERENCE = True

    runName = f'northern_graphnet_models{run_names}_stacking'
    print("run name is: " , runName)
    
    training_predictions = []
    val_predictions = []
    for pred_df in prediction_df:
        #training_predictions.append(pred_df[int(len(pred_df)*0.8):])
        #val_predictions.append(pred_df[:int(len(pred_df)*0.8)])
        val_predictions.append(pred_df)
        

    train_dataset = DatasetStacking(target_columns=['direction_x', 'direction_y', 'direction_z', 'direction_kappa'],
                                        model_preds=training_predictions)
    val_dataset = DatasetStacking(target_columns=['direction_x', 'direction_y', 'direction_z', 'direction_kappa'],
                                        model_preds=val_predictions)
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    scheduler_kwargs = {
        "milestones": [
            0,
            len(train_dataloader)//(len(device)*accumulate_grad_batches[0]*30),
            len(train_dataloader)*max_epochs//(len(device)*accumulate_grad_batches[0]*2),
            len(train_dataloader)*max_epochs//(len(device)*accumulate_grad_batches[0]),                
        ],
        "factors": [1e-03, 1, 1, 1e-03],
        "verbose": False,
        }

    ## Start training

    model  = build_model_stacking(
        dataset=train_dataset,
        hidden_size=hidden_size,
        scheduler_class=PiecewiseLinearLR,
        accumulate_grad_batches=accumulate_grad_batches,
        scheduler_kwargs=scheduler_kwargs,
        )

    if not INFERENCE:
        callbacks = [
            ModelCheckpoint(
                dirpath='/remote/ceph/user/l/llorente/train_DynEdgeTITO_northern_Oct23/model_stacking',
                filename=runName+'-{epoch:02d}-{trn_tloss:.6f}',
                save_weights_only=False,
            ),
            ProgressBar(),
        ]

        model.fit(train_dataloader=train_dataloader,
                #val_dataloader=val_dataloader,
                callbacks=callbacks,
                max_epochs=max_epochs,
                gpus=device,)

        #model.save('/remote/ceph/user/l/llorente/train_DynEdgeTITO_northern_Oct23/model_stacking/'+runName+'-last.pth')
        model.save_state_dict('/remote/ceph/user/l/llorente/train_DynEdgeTITO_northern_Oct23/model_stacking/'+ runName+'-last_state_dict.pth')
    else:


        from tqdm.auto import tqdm
        CKPT = f'/remote/ceph/user/l/llorente/tito_solution/model_graphnet/stacking-6models-last.pth'
        state_dict =  torch.load(CKPT, torch.device('cpu'))

        if 'state_dict' in state_dict.keys():
            state_dict = state_dict['state_dict']
        model.load_state_dict(state_dict)
        validateMode=True

        event_nos = []
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
                preds.append(pred[0])
                event_nos.append(batch[2])
                
                if validateMode:
                    real_x.append(batch[1][:,0])
                    real_y.append(batch[1][:,1])
                    real_z.append(batch[1][:,2])

        preds = torch.cat(preds).to('cpu').detach().numpy()
        real_x = torch.cat(real_x).to('cpu').numpy()
        real_y = torch.cat(real_y).to('cpu').numpy()
        real_z = torch.cat(real_z).to('cpu').numpy()
        results = pd.DataFrame(preds, columns=model.prediction_columns)
        
        results['event_no'] = np.concatenate(event_nos)
        
        if validateMode:
            results['real_x'] = real_x
            results['real_y'] = real_y
            results['real_z'] = real_z

        results_merged = results.copy()  # Make a copy of the results DataFrame
        for prediction_df in prediction_df:
            event_azimuth_zenith = prediction_df[["event_no", "azimuth", "zenith"]]
            results_merged = results_merged.merge(event_azimuth_zenith, on="event_no", how="left", suffixes=("", "_prediction"))
        results_merged.sort_values('event_no')

        results.to_csv(f'/remote/ceph/user/l/llorente/prediction_models/stacking_tito/model_checkpoint_graphnet/model_stacking_northern_load_tito.csv')