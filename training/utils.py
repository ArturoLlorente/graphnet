from graphnet.data.dataset import EnsembleDataset
from graphnet.data.dataset import SQLiteDataset, ParquetDataset
from typing import Dict, List, Optional, Union, Any
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

def make_dataloaders(
    db: Union[List[str], str],
    train_selection: Optional[List[int]],
    val_selection: Optional[List[int]],
    backend: str,
    config: Dict[str, Any],
) -> DataLoader:

    """Construct `DataLoader` instance."""
    # Check(s)
    if isinstance(config["pulsemap"], str):
        config["pulsemap"] = [config["pulsemap"]]
    if backend == "sqlite":
        #assert len(train_selection) == len(db)
        train_datasets = []
        pbar = tqdm(total=len(db))
        for db_idx, database in enumerate(db):

            train_datasets.append(
                SQLiteDataset(
                    path=database,
                    graph_definition=config["graph_definition"],
                    pulsemaps=config["pulsemap"],
                    features=config["features"],
                    truth=config["truth"],
                    selection=None,#train_selection[db_idx],
                    truth_table=config["truth_table"],
                    index_column=config["index_column"],
                )
            )
            pbar.update(1)

        if isinstance(config["labels"], dict):
            for label in config["labels"].keys():
                for train_dataset in train_datasets:
                    train_dataset.add_label(
                        key=label, fn=config["labels"][label]
                    )

        train_dataset = EnsembleDataset(train_datasets)
    elif backend == "parquet":
        assert isinstance(train_selection, list)
        train_dataset = ParquetDataset(
            path=db,
            graph_definition=config["graph_definition"],
            pulsemaps=config["pulsemap"],
            features=config["features"],
            truth=config["truth"],
            selection=train_selection,
            truth_table=config["truth_table"],
            index_column=config["index_column"],
        )
        if isinstance(config["labels"], dict):
            for label in config["labels"].keys():
                train_dataset.add_label(
                    key=label, fn=config["labels"][label]
                )

    training_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        collate_fn=config["collate_fn"],
        persistent_workers=config["persistent_workers"],
        prefetch_factor=config["prefetch_factor"],
#        multiprocessing_context = "spawn",
    )

    if val_selection is not None:
        if backend == "sqlite":
            val_dataset = SQLiteDataset(
                path=db[-1],
                graph_definition=config["graph_definition"],
                pulsemaps=config["pulsemap"],
                features=config["features"],
                truth=config["truth"],
                selection=val_selection,
                truth_table=config["truth_table"],
                index_column=config["index_column"],
            )
        elif backend == "parquet":
            val_dataset = ParquetDataset(
                path=db,
                graph_definition=config["graph_definition"],
                pulsemaps=config["pulsemap"],
                features=config["features"],
                truth=config["truth"],
                selection=val_selection,
                truth_table=config["truth_table"],
                index_column=config["index_column"],
            )

        if isinstance(config["labels"], dict):
            for label in config["labels"].keys():
                val_dataset.add_label(
                    key=label, fn=config["labels"][label]
                )

        validation_dataloader = DataLoader(
            dataset=val_dataset,
            batch_size=config["batch_size"],
            shuffle=False,
            num_workers=config["num_workers"],
            collate_fn=config["collate_fn"],
            persistent_workers=config["persistent_workers"],
            prefetch_factor=config["prefetch_factor"],
            multiprocessing_context = "spawn",
        )
    else:
        validation_dataloader = None
        val_dataset = None

    return (
        training_dataloader,
        validation_dataloader,
        train_dataset,
        val_dataset,
    )
    
