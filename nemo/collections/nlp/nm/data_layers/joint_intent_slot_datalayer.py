from nemo.collections.nlp.data import BertJointIntentSlotDataset, BertJointIntentSlotInferDataset
from nemo.collections.nlp.nm.data_layers.text_datalayer import TextDataLayer
from nemo.core import NeuralType, AxisType, BatchTag, TimeTag


class BertJointIntentSlotDataLayer(TextDataLayer):
    """
    Creates the data layer to use for the task of joint intent
    and slot classification with pretrained model.

    All the data processing is done in BertJointIntentSlotDataset.

    input_mask: used to ignore some of the input tokens like paddings

    loss_mask: used to mask and ignore tokens in the loss function

    subtokens_mask: used to ignore the outputs of unwanted tokens in
    the inference and evaluation like the start and end tokens

    Args:
        dataset (BertJointIntentSlotDataset):
            the dataset that needs to be converted to DataLayerNM
    """

    @property
    def output_ports(self):
        """Returns definitions of module output ports.

        input_ids:
            0: AxisType(BatchTag)

            1: AxisType(TimeTag)

        input_type_ids:
            0: AxisType(BatchTag)

            1: AxisType(TimeTag)

        input_mask:
            0: AxisType(BatchTag)

            1: AxisType(TimeTag)

        loss_mask:
            0: AxisType(BatchTag)

            1: AxisType(TimeTag)

        subtokens_mask:
            0: AxisType(BatchTag)

            1: AxisType(TimeTag)

        intents:
            0: AxisType(BatchTag)

        slots:
            0: AxisType(BatchTag)

            1: AxisType(TimeTag)
        """
        return {
            "input_ids": NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag)}),
            "input_type_ids": NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag)}),
            "input_mask": NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag)}),
            "loss_mask": NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag)}),
            "subtokens_mask": NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag)}),
            "intents": NeuralType({0: AxisType(BatchTag)}),
            "slots": NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag)}),
        }

    def __init__(
        self,
        input_file,
        slot_file,
        pad_label,
        tokenizer,
        max_seq_length,
        num_samples=-1,
        shuffle=False,
        batch_size=64,
        ignore_extra_tokens=False,
        ignore_start_end=False,
        dataset_type=BertJointIntentSlotDataset,
        **kwargs
    ):
        kwargs['batch_size'] = batch_size
        dataset_params = {
            'input_file': input_file,
            'slot_file': slot_file,
            'pad_label': pad_label,
            'tokenizer': tokenizer,
            'max_seq_length': max_seq_length,
            'num_samples': num_samples,
            'shuffle': shuffle,
            'ignore_extra_tokens': ignore_extra_tokens,
            'ignore_start_end': ignore_start_end,
        }
        super().__init__(dataset_type, dataset_params, **kwargs)


class BertJointIntentSlotInferDataLayer(TextDataLayer):
    """
    Creates the data layer to use for the task of joint intent
    and slot classification with pretrained model. This is for

    All the data processing is done in BertJointIntentSlotInferDataset.

    input_mask: used to ignore some of the input tokens like paddings

    loss_mask: used to mask and ignore tokens in the loss function

    subtokens_mask: used to ignore the outputs of unwanted tokens in
    the inference and evaluation like the start and end tokens

    Args:
        dataset (BertJointIntentSlotInferDataset):
            the dataset that needs to be converted to DataLayerNM
    """

    @property
    def output_ports(self):
        """Returns definitions of module output ports.

        input_ids:
            0: AxisType(BatchTag)

            1: AxisType(TimeTag)

        input_type_ids:
            0: AxisType(BatchTag)

            1: AxisType(TimeTag)

        input_mask:
            0: AxisType(BatchTag)

            1: AxisType(TimeTag)

        loss_mask:
            0: AxisType(BatchTag)

            1: AxisType(TimeTag)

        subtokens_mask:
            0: AxisType(BatchTag)

            1: AxisType(TimeTag)

        """
        return {
            "input_ids": NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag)}),
            "input_type_ids": NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag)}),
            "input_mask": NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag)}),
            "loss_mask": NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag)}),
            "subtokens_mask": NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag)}),
        }

    def __init__(
        self, queries, tokenizer, max_seq_length, batch_size=1, dataset_type=BertJointIntentSlotInferDataset, **kwargs
    ):
        kwargs['batch_size'] = batch_size
        dataset_params = {'queries': queries, 'tokenizer': tokenizer, 'max_seq_length': max_seq_length}
        super().__init__(dataset_type, dataset_params, **kwargs)