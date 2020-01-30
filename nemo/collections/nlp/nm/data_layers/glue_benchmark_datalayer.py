__all_ = ['GlueClassificationDataLayer', 'GlueRegressionDataLayer']
from nemo.collections.nlp.data import GLUEDataset
from nemo.collections.nlp.nm.data_layers.text_datalayer import TextDataLayer
from nemo.core import AxisType, BatchTag, CategoricalTag, NeuralType, RegressionTag, TimeTag


class GlueClassificationDataLayer(TextDataLayer):
    """
    Creates the data layer to use for the GLUE classification tasks,
    more details here: https://gluebenchmark.com/tasks

    All the data processing is done in GLUEDataset.

    Args:
        dataset_type (GLUEDataset):
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

            labels:
                0: AxisType(CategoricalTag)
        """
        return {
            "input_ids": NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag)}),
            "input_type_ids": NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag)}),
            "input_mask": NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag)}),
            "labels": NeuralType({0: AxisType(CategoricalTag)}),
        }

    def __init__(
        self,
        data_dir,
        tokenizer,
        max_seq_length,
        processor,
        evaluate=False,
        token_params={},
        num_samples=-1,
        shuffle=False,
        batch_size=64,
        dataset_type=GLUEDataset,
        **kwargs
    ):
        kwargs['batch_size'] = batch_size
        dataset_params = {
            'data_dir': data_dir,
            'output_mode': 'classification',
            'processor': processor,
            'evaluate': evaluate,
            'token_params': token_params,
            'tokenizer': tokenizer,
            'max_seq_length': max_seq_length,
        }

        super().__init__(dataset_type, dataset_params, **kwargs)


class GlueRegressionDataLayer(TextDataLayer):
    """
    Creates the data layer to use for the GLUE STS-B regression task,
    more details here: https://gluebenchmark.com/tasks

    All the data processing is done in GLUEDataset.

    Args:
        dataset_type (GLUEDataset):
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

            labels:
                0: AxisType(RegressionTag)
        """
        return {
            "input_ids": NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag)}),
            "input_type_ids": NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag)}),
            "input_mask": NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag)}),
            "labels": NeuralType({0: AxisType(RegressionTag)}),
        }

    def __init__(
        self,
        data_dir,
        tokenizer,
        max_seq_length,
        processor,
        evaluate=False,
        token_params={},
        num_samples=-1,
        shuffle=False,
        batch_size=64,
        dataset_type=GLUEDataset,
        **kwargs
    ):
        kwargs['batch_size'] = batch_size
        dataset_params = {
            'data_dir': data_dir,
            'output_mode': 'regression',
            'processor': processor,
            'evaluate': evaluate,
            'token_params': token_params,
            'tokenizer': tokenizer,
            'max_seq_length': max_seq_length,
        }

        super().__init__(dataset_type, dataset_params, **kwargs)
