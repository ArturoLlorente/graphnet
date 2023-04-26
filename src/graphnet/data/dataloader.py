"""Base `Dataloader` class(es) used in `graphnet`."""

from typing import Any, Callable, Dict, List, Union

import torch.utils.data
from torch_geometric.data import Batch, Data

from graphnet.data.dataset import Dataset
from graphnet.utilities.config import DatasetConfig


def collate_fn(graphs: List[Data]) -> Batch:
    """Remove graphs with less than two DOM hits.

    Should not occur in "production.
    """
    graphs = [g for g in graphs if g.n_pulses > 1]
    return Batch.from_data_list(graphs)


def collate_fn_tito(graphs: List[Data], split_list=[0.8, 1.0]) -> list:
    """Implementation of the solution used in the TITO solution for the.

    Kaggle competition "Icecube - Neutrinos in Deep Ice" ended on 20.04.2023.
    """
    graphs = [g for g in graphs if g.n_pulses > 1]
    graphs.sort(key=lambda x: x.n_pulses)
    batch_list = []
    for minp, maxp in zip([0] + split_list[:-1], split_list):
        min_idx = int(minp * len(graphs))
        max_idx = int(maxp * len(graphs))
        this_graphs = graphs[min_idx:max_idx]
        this_batch = Batch.from_data_list(this_graphs)
        batch_list.append(this_batch)
    return batch_list


def do_shuffle(selection_name: str) -> bool:
    """Check whether to shuffle selection with name `selection_name`."""
    return "train" in selection_name.lower()


class DataLoader(torch.utils.data.DataLoader):
    """Class for loading data from a `Dataset`."""

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 1,
        shuffle: bool = False,
        num_workers: int = 10,
        persistent_workers: bool = True,
        collate_fn: Callable = collate_fn,
        **kwargs: Any,
    ) -> None:
        """Construct `DataLoader`."""
        # Base class constructor
        super().__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_fn,
            persistent_workers=persistent_workers,
            prefetch_factor=2,
            **kwargs,
        )

    @classmethod
    def from_dataset_config(
        cls,
        config: DatasetConfig,
        **kwargs: Any,
    ) -> Union["DataLoader", Dict[str, "DataLoader"]]:
        """Construct `DataLoader`s based on selections in `DatasetConfig`."""
        if isinstance(config.selection, dict):
            assert "shuffle" not in kwargs, (
                "When passing a `DatasetConfig` with multiple selections, "
                "`shuffle` is automatically inferred from the selection name, "
                "and thus should not specified as an argument."
            )
            datasets = Dataset.from_config(config)
            assert isinstance(datasets, dict)
            data_loaders: Dict[str, DataLoader] = {}
            for name, dataset in datasets.items():
                data_loaders[name] = cls(
                    dataset,
                    shuffle=do_shuffle(name),
                    **kwargs,
                )

            return data_loaders

        else:
            assert "shuffle" in kwargs, (
                "When passing a `DatasetConfig` with a single selections, you "
                "need to specify `shuffle` as an argument."
            )
            dataset = Dataset.from_config(config)
            assert isinstance(dataset, Dataset)
            return cls(dataset, **kwargs)
