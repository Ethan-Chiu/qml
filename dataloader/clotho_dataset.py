from typing import Tuple, List, AnyStr, Union
from pathlib import Path

import pandas as pd
import torchaudio
import random
from numpy import ndarray
from torch.utils.data import Dataset


__all__ = ['ClothoDataset']


class ClothoDataset(Dataset):

    def __init__(self, data_dir: Path,
                 split: AnyStr,
                 csv_path: Path) \
            -> None:
        super(ClothoDataset, self).__init__()

        the_dir: Path = data_dir.joinpath(split)
        self.audio_dir = the_dir

        self.data = pd.read_csv(csv_path)

        print("data", self.data[:5])


    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[ndarray, ndarray]:
        audio_file = self.data.iloc[idx, 0]
        audio_path = self.audio_dir.joinpath(audio_file)

        # waveform, sample_rate = torchaudio.load(audio_path)

        # Randomly select one of the captions
        captions = self.data.iloc[idx, 1:6].values
        index = random.randint(0, len(captions) - 1)
        caption = captions[index]
        caption_id = f"{audio_file[:-4]}_{index}"

        return str(audio_path), caption, caption_id
    