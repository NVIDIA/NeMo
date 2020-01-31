from nemo.backends.pytorch import DataLayerNM
from nemo.collections.nlp.data.datasets import *

__all__ = ['TextDataLayer']


class TextDataLayer(DataLayerNM):
    """
    Generic Text Data Layer NM which wraps PyTorch's dataset

    Args:
        dataset_type: type of dataset used for this datalayer
        dataset_params (dict): all the params for the dataset
    """

    def __init__(self, dataset_type, dataset_params, **kwargs):
        super().__init__(**kwargs)
        self._dataset = dataset_type(**dataset_params)

    def __len__(self):
        return len(self._dataset)

    @property
    def dataset(self):
        return self._dataset

    @property
    def data_iterator(self):
        return None
