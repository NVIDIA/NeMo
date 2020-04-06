import torch
from torch.utils import data as pt_data

import nemo
from nemo.collections.nlp.data.datasets.sgd_dataset.SGDDataset import SGDDataset
from nemo.collections.nlp.nm.data_layers.text_datalayer import TextDataLayer
from nemo.core.neural_types import ChannelType, EmbeddedTextType, LabelsType, LengthsType, NeuralType
from nemo.utils.decorators import add_port_docs

__all__ = ['SGDDataLayer']


class SGDDataLayer(TextDataLayer):
    """
    Data layer for Schema Guided Dialogue State Tracking Dataset.

    Args:
        TODO: fix
    """

    @property
    @add_port_docs()
    def output_ports(self):
        """Returns definitions of module output ports.
        TODO update
        input_ids: indices of tokens which constitute batches of text segments
            0: AxisType(BatchTag)

            1: AxisType(TimeTag)

        input_type_ids: indices of token types (e.g., sentences A & B in BERT)
            0: AxisType(BatchTag)

            1: AxisType(TimeTag)

        input_mask: bool tensor with 0s in place of tokens to be masked
            0: AxisType(BatchTag)

            1: AxisType(TimeTag)

        """
        return {
            "example_id_num": NeuralType(('B'), ChannelType()),
            "service_id": NeuralType(('B'), ChannelType()),
            "is_real_example": NeuralType(('B'), ChannelType()),
            "utterance_ids": NeuralType(('B', 'T'), ChannelType()),
            "utterance_segment": NeuralType(('B', 'T'), ChannelType()),
            "utterance_mask": NeuralType(('B', 'T'), ChannelType()),
            "num_categorical_slots": NeuralType(('B'), LengthsType()),
            "categorical_slot_status": NeuralType(('B', 'T'), LabelsType()),
            "num_categorical_slot_values": NeuralType(('B', 'T'), LengthsType()),
            "categorical_slot_values": NeuralType(('B', 'T'), LabelsType()),
            "num_noncategorical_slots": NeuralType(('B'), LengthsType()),
            "noncategorical_slot_status": NeuralType(('B', 'T'), LabelsType()),
            "noncategorical_slot_value_start": NeuralType(('B', 'T'), LabelsType()),
            "noncategorical_slot_value_end": NeuralType(('B', 'T'), LabelsType()),
            "start_char_idx": NeuralType(('B', 'T'), LabelsType()),
            "end_char_idx": NeuralType(('B', 'T'), LabelsType()),
            "num_slots": NeuralType(('B'), LengthsType()),
            "requested_slot_status": NeuralType(('B', 'T'), LabelsType()),
            "num_intents": NeuralType(('B'), LengthsType()),
            "intent_status": NeuralType(('B'), LabelsType()),
            "cat_slot_emb": NeuralType(('B', 'T', 'C'), EmbeddedTextType()),
            "cat_slot_value_emb": NeuralType(('B', 'T', 'C', 'C'), EmbeddedTextType()),
            "noncat_slot_emb": NeuralType(('B', 'T', 'C'), EmbeddedTextType()),
            "req_slot_emb": NeuralType(('B', 'T', 'C'), EmbeddedTextType()),
            "intent_emb": NeuralType(('B', 'T', 'C'), EmbeddedTextType()),
        }

    def __init__(
        self,
        dataset_split,
        schema_emb_processor,
        dialogues_processor,
        dataset_type=SGDDataset,
        shuffle=False,
        batch_size=1,
        num_workers=-1,
    ):

        dataset_params = {
            'dataset_split': dataset_split,
            'schema_emb_processor': schema_emb_processor,
            'dialogues_processor': dialogues_processor,
        }
        super().__init__(dataset_type, dataset_params, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
