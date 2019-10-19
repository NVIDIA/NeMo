import os

from torch.utils.data import Dataset


class DSTDataset(Dataset):
    def __init__(self, data_file):
        self.data_file = data_file

    def prepare_data(self):
        pass
