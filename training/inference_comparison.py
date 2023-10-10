import torch
import pandas as pd
import numpy as np
from collections import OrderedDict

from typing import Dict, List, Optional
from graphnet.models import Model
from graphnet.training.utils import make_dataloader
from torch.utils.data import DataLoader
from graphnet.data.constants import FEATURES, TRUTH

def get_predictions(
    model: Model,
    dataloader: DataLoader,
    prediction_columns: List[str],
    *,
    node_level: bool = False,
    additional_attributes: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Get `model` predictions on `dataloader`."""
    # Gets predictions from model on the events in the dataloader.
    # NOTE: dataloader must NOT have shuffle = True!

    # Check(s)
    if additional_attributes is None:
        additional_attributes = []
    assert isinstance(additional_attributes, list)

    # Set model to inference mode
    model.inference()

    # Get predictions
    predictions_torch = model.predict(dataloader)
    predictions_list = [
        p[0].detach().cpu().numpy() for p in predictions_torch
    ]  # Assuming single task
    predictions = np.concatenate(predictions_list, axis=0)
    try:
        assert len(prediction_columns) == predictions.shape[1]
    except IndexError:
        predictions = predictions.reshape((-1, 1))
        assert len(prediction_columns) == predictions.shape[1]

    # Get additional attributes
    attributes: Dict[str, List[np.ndarray]] = OrderedDict(
        [(attr, []) for attr in additional_attributes]
    )
    for batch in dataloader:
        for attr in attributes:
            attribute = batch[attr].detach().cpu().numpy()
            if node_level:
                if attr == "event_no":
                    attribute = np.repeat(
                        attribute, batch["n_pulses"].detach().cpu().numpy()
                    )
            attributes[attr].extend(attribute)

    data = np.concatenate(
        [predictions]
        + [
            np.asarray(values)[:, np.newaxis] for values in attributes.values()
        ],
        axis=1,
    )

    results = pd.DataFrame(
        data, columns=prediction_columns + additional_attributes
    )
    return results

model = torch.load('/remote/ceph/user/l/llorente/tito_solution/model_graphnet/model1-last.pth')
device = 'cuda:0'

model = model.to(device)
model.eval()
model.inference()

test_max_pulses = 200
test_path = '/remote/ceph/user/l/llorente/kaggle/databases_merged/batch_val.db'
test_selection_file = '/remote/ceph/user/l/llorente/kaggle/selection_files/pulse_information_val.csv'
test_selection = test_selection_file.loc[test_selection_file['n_pulses']<test_max_pulses,:]['event_id'].ravel().tolist()

test_dataloader =  make_dataloader(db = test_path,
                                    selection = test_selection,
                                    pulsemaps = 'pulse_table',
                                    num_workers = 128,
                                    features = FEATURES.KAGGLE,
                                    shuffle = False,
                                    truth = TRUTH.KAGGLE,
                                    batch_size = 100,
                                    truth_table = 'meta_table',
                                    index_column='event_id',
                                    )


results = get_predictions(model, 
                          test_dataloader,
                          prediction_columns=['dir_x_pred', 'dir_y_pred', 'dir_z_pred', 'dir_kappa_pred'],
                          additional_attributes=['zenith', 'azimuth', 'event_id', 'energy']),
