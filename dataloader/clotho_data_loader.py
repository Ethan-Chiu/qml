from typing import Callable, Union, Tuple, AnyStr, Optional
from functools import partial
from pathlib import Path

from torch.utils.data import random_split
from torch.utils.data.dataloader import DataLoader

from .clotho_dataset import ClothoDataset
from .collate_fn import clotho_collate_fn

__all__ = ['get_clotho_loader']


def get_clotho_loader(data_dir: Path,
                      split: str,
                      csv_path: Path,
                      batch_size: int,
                      nb_t_steps_pad: Union[AnyStr, Tuple[int, int]],
                      shuffle: Optional[bool] = True,
                      drop_last: Optional[bool] = True,
                      input_pad_at: Optional[str] = 'start',
                      output_pad_at: Optional[str] = 'end',
                      num_workers: Optional[int] = 1) \
        -> DataLoader:
    
    dataset: ClothoDataset = ClothoDataset(
        data_dir=data_dir, 
        split=split,
        csv_path=csv_path)

    # collate_fn: Callable = partial(
    #     clotho_collate_fn,
    #     nb_t_steps=nb_t_steps_pad,
    #     input_pad_at=input_pad_at,
    #     output_pad_at=output_pad_at)

    collate_fn = clotho_collate_fn
    
    train_size = int(0.9 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_dataloader = DataLoader(
        dataset=train_dataset, 
        batch_size=batch_size,
        shuffle=shuffle, 
        num_workers=num_workers,
        drop_last=drop_last, 
        # collate_fn=collate_fn
    )

    test_dataloader = DataLoader(
        dataset=test_dataset, 
        batch_size=batch_size,
        shuffle=shuffle, 
        num_workers=num_workers,
        drop_last=drop_last, 
    )

    return train_dataloader, test_dataloader
