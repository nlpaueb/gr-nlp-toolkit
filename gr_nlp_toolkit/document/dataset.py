import torch
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co


class DatasetImpl(Dataset):
    def __init__(self, input_ids):
        self._input_ids = input_ids

    def __getitem__(self, index) -> T_co:
        return {
            "input": [torch.tensor(self._input_ids[index], dtype=torch.long), torch.tensor(len(self._input_ids[index]))]
        }

    def __len__(self):
        return 1

    @property
    def input_ids(self):
        return self._input_ids

    @input_ids.setter
    def input_ids(self, value):
        self._input_ids = value
