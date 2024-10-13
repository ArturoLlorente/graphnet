import os
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import torch
import pickle
from typing import Dict, Any, Callable, Union

from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    GradientAccumulationScheduler,
    LearningRateMonitor,
)
from torch.utils.data import DataLoader



from graphnet.models import StandardModelStacking
from graphnet.models.task.reconstruction import DirectionReconstructionWithKappa
from graphnet.training.labels import Direction, Direction_flipped
from graphnet.training.loss_functions import VonMisesFisher3DLoss, LossFunction
from graphnet.training.callbacks import ProgressBar
from graphnet.training.utils import make_dataloader
from graphnet.training.utils import collate_fn, collator_sequence_buckleting


from utils import make_dataloaders, rename_state_dict_keys, DatasetStacking


#class DirectionReconstructionWithKappaTITO(Task):
#    default_target_labels = ["direction"]
#    default_prediction_labels = [
#        "dir_x_pred",
#        "dir_y_pred",
#        "dir_z_pred",
#        "dir_kappa_pred",
#    ]
#    nb_inputs = 3
#
#    def _forward(self, x: torch.Tensor) -> torch.Tensor:
#        kappa = torch.linalg.vector_norm(x, dim=1)
#        kappa = torch.clamp(kappa, min=torch.finfo(x.dtype).eps)
#        vec_x = x[:, 0] / kappa
#        vec_y = x[:, 1] / kappa
#        vec_z = x[:, 2] / kappa
#        return torch.stack((vec_x, vec_y, vec_z, kappa), dim=1)
    
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
    config: Dict[str, Any],
): 
     
    task = DirectionReconstructionWithKappa(
            hidden_size=config["hidden_size"],
            target_labels="direction",
            loss_function=VonMisesFisher3DLoss(),
        )
    task2 = DirectionReconstructionWithKappa(
        hidden_size=config["hidden_size"],
        target_labels="direction",
        loss_function=VonMisesFisher3DLoss(),
    )

    model = StandardModelStacking(        
                n_input_features=config["n_input_features"],
                hidden_size=config["hidden_size"],
                tasks= [task,task2],
                )
    return model


# Main function call
if __name__ == "__main__":

    config = {
        "archive": "/scratch/users/allorana/tito_cascades_retrain",
        "target": "direction",
        "batch_size": 28,
        "num_workers": 0,
        "index_column": "event_no",
        "labels": {"direction": Direction()},
        "persistent_workers": True,
        "prediction_columns": ["direction_x", "direction_y", "direction_z", "direction_kappa"],
        "ckpt_path": '/scratch/users/allorana/models_tito/stacking-6models-last.pth',
        "prefetch_factor": None,
        "gpus": [],
        "hidden_size": 512,
        "event_type": "track",
        "additional_attributes": [ "zenith", "azimuth", "event_no", "energy"],
    }


    run_name = (
        #f"{model_name}_retrain_IceMix_batch{config['batch_size']}_optimizer_Adam_LR{config['scheduler_kwargs']['max_lr']}_annealStrat_{config['scheduler_kwargs']['anneal_strategy']}_"
        #f"ema_decay_{config['ema_decay']}_1epoch_11_02"
        f"test_tito_stacking_{config['event_type']}"
    )

    # Configurations
    torch.multiprocessing.set_sharing_strategy("file_system")
    torch.multiprocessing.set_start_method("spawn")
    
    df_preds = []
    cascades_tito_path = '/scratch/users/allorana/prediction_cascades_tito'
    for idx in range(1, 7):
        df_preds.append(pd.read_csv(f"{cascades_tito_path}/model{idx}_baseline_{config['event_type']}.csv"))

    test_dataset = DatasetStacking(
        target_columns = ["direction_x", "direction_y", "direction_z", "direction_kappa", "direction_x1", "direction_y1", "direction_z1", "direction_kappa1"],
        model_preds=df_preds,
        use_mid_features=True)
    
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
    )
    
    config["n_input_features"]=test_dataset.n_columns()
    
    model  = build_model(config=config)
    model.prediction_columns = ["direction_x", "direction_y", "direction_z", "direction_kappa"]
    
    state_dict = torch.load(config["ckpt_path"], torch.device("cpu"))
    if 'state_dict' in state_dict.keys():
        state_dict = state_dict['state_dict']
    model.load_state_dict(state_dict)
    
    USE_ALL_FEA_IN_PRED = False
    validateMode=True
    device = f'cuda:{config["gpus"][0]}' if len(config["gpus"]) > 0 else 'cpu'

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
        model.to(device)
        for batch in tqdm(test_dataloader):
            pred = model(batch[0].to(device))
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
    results.to_csv(f'/scratch/users/allorana/prediction_cascades_tito/stacking_baseline_track.csv')
