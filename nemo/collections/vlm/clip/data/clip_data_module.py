from copy import deepcopy
from typing import Any, Dict, Literal, Optional

import megatron.energon.flavors.crude
import torch
from torchvision import transforms
from torchvision.transforms import v2 as torchvision_transforms
import fiddle as fdl
import lightning.pytorch as pl
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from megatron.core import parallel_state
from megatron.energon import WorkerConfig, get_savable_loader, get_train_dataset, DefaultTaskEncoder, basic_sample_keys, \
    Cooker, SkipSample
from torch.utils.data import DataLoader
from typing_extensions import Self

from nemo.collections.multimodal.data.clip.augmentations.augmentations import image_transform
from nemo.collections.multimodal.data.energon.config import MultiModalSampleConfig
from nemo.collections.multimodal.data.energon.task_encoder import MultiModalTaskEncoder
from nemo.lightning.io.mixin import IOMixin, serialization, track_io
from nemo.lightning.pytorch.plugins import MegatronDataSampler
from nemo.utils import logging

class EnergonMultiModalDataModule(pl.LightningDataModule, IOMixin):
    """
    A PyTorch Lightning DataModule for handling multimodal datasets with images and text.

    This data module is designed to work with multimodal datasets that involve both images and text.
    It provides a seamless interface to load training and validation data, manage batching, and handle
    the state of the data pipeline across training epochs. The module integrates with the Megatron-Energon
    framework for efficient data handling in large-scale distributed training.

    Attributes:
    path (str): Path to the energon dataset.
    tokenizer (Tokenizer): The tokenizer used for processing text.
    image_processor (ImageProcessor): The image processor used for preprocessing images.
    seq_length (int): The maximum sequence length for tokenized text.
    micro_batch_size (int): The batch size for training and validation.
    num_workers (int): Number of workers for data loading.
    pin_memory (bool): Whether to pin memory in the DataLoader.
    multimodal_sample_config (MultiModalSampleConfig): Configuration object for multimodal samples.
    task_encoder (MultiModalTaskEncoder): Encoder responsible for encoding and batching samples.
    init_global_step (int): The initial global step for the trainer, used for resuming training.
    data_sampler (SequentialMegatronSampler): Sampler responsible for generating sequential samples.
    train_dataloader_object (Optional): The DataLoader object for training data.
    val_dataloader_object (Optional): The DataLoader object for validation data.
    """

    def __init__(
        self,
        path: str,
        tokenizer,
        image_processor,
        seq_length: int = 2048,
        micro_batch_size: int = 1,
        global_batch_size: int = 1,
        num_workers: int = 1,
        pin_memory: bool = True,
        multimodal_sample_config: Optional[MultiModalSampleConfig] = MultiModalSampleConfig(),
        task_encoder: Optional[MultiModalTaskEncoder] = None,
        decoder_seq_length: Optional[int] = None,
    ) -> None:
        """
        Initialize the EnergonMultiModalDataModule.

        Parameters:
        path (str): Path to the dataset.
        tokenizer (Tokenizer): The tokenizer used for processing text.
        image_processor (ImageProcessor): The image processor used for preprocessing images.
        seq_length (int, optional): The maximum sequence length for tokenized text. Defaults to 2048.
        micro_batch_size (int, optional): The batch size for training and validation. Defaults to 1.
        num_workers (int, optional): Number of workers for data loading. Defaults to 1.
        pin_memory (bool, optional): Whether to pin memory in the DataLoader. Defaults to True.
        multimodal_sample_config (MultiModalSampleConfig, optional): Configuration object for multimodal samples.
        Defaults to MultiModalSampleConfig().
        task_encoder (MultiModalTaskEncoder, optional): Encoder responsible for encoding and batching samples.
        If not provided, a default (MultimodalTaskEncoder) encoder will be created. Defaults to None.
        """

        super().__init__()
        self.path = path
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.seq_length = seq_length
        self.decoder_seq_length = decoder_seq_length
        self.micro_batch_size = micro_batch_size
        self.global_batch_size = global_batch_size
        self.micro_batch_size = micro_batch_size
        self.global_batch_size = global_batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.multimodal_sample_config = multimodal_sample_config
        self.task_encoder = task_encoder or MultiModalTaskEncoder(
            tokenizer=self.tokenizer,
            image_processor=self.image_processor,
            multimodal_sample_config=multimodal_sample_config,
        )
        self.init_global_step = 0
        self.data_sampler = SequentialMegatronSampler(
            seq_len=self.seq_length,
            decoder_seq_len=self.decoder_seq_length,
            micro_batch_size=self.micro_batch_size,
            global_batch_size=self.global_batch_size,
        )
        self.train_dataloader_object = None
        self.val_dataloader_object = None

    def io_init(self, **kwargs) -> fdl.Config[Self]:

        cfg_kwargs = {k: deepcopy(v) for k, v in kwargs.items() if k not in ['image_processor', 'task_encoder']}

        for val in cfg_kwargs.values():
            if not serialization.find_node_traverser(type(val)):
                track_io(type(val))
        cfg = fdl.Config(type(self), **cfg_kwargs)
        return cfg

    def datasets_provider(self, worker_config, split: Literal['train', 'val'] = 'val'):
        """
        Provide the dataset for training or validation.

        This method retrieves the dataset for the specified split (either 'train' or 'val') and configures
        it according to the worker configuration.

        Parameters:
        worker_config: Configuration for the data loader workers.
        split (Literal['train', 'val'], optional): The data split to retrieve ('train' or 'val'). Defaults to 'val'.

        Returns:
        Dataset: The dataset configured for the specified split.
        """
        if split not in {'train', 'val'}:
            raise ValueError("Invalid value for split. Allowed values are 'train' or 'val'.")
        _dataset = get_train_dataset(
            self.path,
            batch_size=self.micro_batch_size,
            task_encoder=self.task_encoder,
            worker_config=worker_config,
            max_samples_per_sequence=None,
            shuffle_buffer_size=100,
            split_part=split,
        )
        return _dataset

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        """
        Initialize and return the training DataLoader.

        This method initializes the DataLoader for the training dataset. It uses the global step
        from the trainer to configure the data sampler and ensures that the parallel state is initialized
        correctly for distributed training.

        Returns:
        TRAIN_DATALOADERS: The DataLoader for the training dataset.
        """
        if self.trainer:
            self.init_global_step = self.trainer.global_step
        self.data_sampler.init_global_step = self.init_global_step
        logging.info(f"Multimodal train dataloader initializing with init_global_step {self.init_global_step}")
        if self.train_dataloader_object:
            return self.train_dataloader_object
        if not parallel_state.is_initialized():
            logging.info(
                f"Muiltimodal data loader parallel state is not initialized,"
                f"using default worker config with no_workers {self.num_workers}"
            )
            worker_config = WorkerConfig.default_worker_config(self.num_workers)
        else:
            rank = parallel_state.get_data_parallel_rank()
            world_size = parallel_state.get_data_parallel_world_size()
            data_parallel_group = parallel_state.get_data_parallel_group()
            logging.info(
                f" Multimodal  train dataloader initializing with"
                f"rank {rank} world_size {world_size} data_parallel_group {data_parallel_group} ****** "
            )
            worker_config = WorkerConfig(
                rank=rank,
                world_size=world_size,
                num_workers=self.num_workers,
                data_parallel_group=data_parallel_group,
                worker_debug_path=None,
                worker_log_level=0,
            )
        train_dataset = self.datasets_provider(worker_config, split='train')
        energon_dataloader = get_savable_loader(train_dataset, worker_config=worker_config)
        self.train_dataloader_object = energon_dataloader
        return self.train_dataloader_object

    def val_dataloader(self) -> EVAL_DATALOADERS:
        """
        Initialize and return the validation DataLoader.

        This method initializes the DataLoader for the validation dataset. It ensures that the parallel state
        is initialized correctly for distributed training and returns a configured DataLoader object.

        Returns:
        EVAL_DATALOADERS: The DataLoader for the validation dataset.
        """
        if self.val_dataloader_object:
            return self.val_dataloader_object

        if not parallel_state.is_initialized():
            logging.info(
                f"Muiltimodal val data loader parallel state is not initialized,"
                "using default worker config with no_workers {self.num_workers}"
            )
            worker_config = WorkerConfig.default_worker_config(self.num_workers)
        else:
            rank = parallel_state.get_data_parallel_rank()
            world_size = parallel_state.get_data_parallel_world_size()
            data_parallel_group = parallel_state.get_data_parallel_group()

            logging.info(f"rank {rank} world_size {world_size} data_parallel_group {data_parallel_group}")
            worker_config = WorkerConfig(
                rank=rank,
                world_size=world_size,
                num_workers=self.num_workers,
                data_parallel_group=data_parallel_group,
                worker_debug_path=None,
                worker_log_level=0,
            )
        val_dataset = self.datasets_provider(worker_config, split='val')
        energon_loader = get_savable_loader(val_dataset, worker_config=worker_config)
        self.val_dataloader_object = energon_loader
        return self.val_dataloader_object

    def test_dataloader(self) -> None:
        """
        Return None as test dataset split does not exist.

        This method overrides the test_dataloader method and returns None since the test dataset split
        is not defined or used in this module.

        Returns:
        None
        """
        logging.warning(f"Multimodal dataloader test dataset split does not exist")
        return None

    def state_dict(self) -> Dict[str, Any]:
        """
        Save the state of the data module.

        This method is called when saving a checkpoint. It generates and saves the state of the data module,
        including the state of the dataloader and the number of consumed samples.

        Returns:
        Dict[str, Any]: A dictionary containing the state of the data module.
        """

        if self.trainer:
            dataloader_obj = self.trainer.train_dataloader
            state = dataloader_obj.save_state()
            consumed_samples = self.data_sampler.compute_consumed_samples(
                self.trainer.global_step - self.init_global_step
            )
            logging.info(f"Multimodal data loader saving dataloader state dict consumed samples {consumed_samples}")
            return {'dataloader_state': state, 'consumed_samples': consumed_samples}

        logging.warning("trainer object not connected to data module object returning empty state")
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """
        Load the state of the data module from a checkpoint.

        This method is called when loading a checkpoint. It restores the state of the data module,
        including the state of the dataloader and the number of consumed samples.

        Parameters:
        state_dict (Dict[str, Any]): The state dictionary containing the saved state of the data module.
        """
        if not 'dataloader_state' in state_dict:
            logging.warning(
                f"Data loader state cannot be resumed from state_dict,"
                f"it does not have the required key dataloader_state. It has {state_dict.keys()}"
            )
            return

        state = state_dict['dataloader_state']
        try:
            if self.trainer:
                self.trainer.datamodule.train_dataloader().restore_state(state)
                logging.info(f" Multimodal dataloader state restored")
            else:
                logging.error(f"Cannot restore state from state_dict {state_dict}")
                raise ValueError(
                    f"Cannot restore state from state_dict: "
                    f"Is the trainer object is initialized and attached to datamodule???"
                )
        except Exception as e:
            raise RuntimeError(f"Failed to dataloader restore state due to: {e}")

        try:
            from megatron.core.num_microbatches_calculator import update_num_microbatches

        except (ImportError, ModuleNotFoundError):
            logging.warning("Megatron num_microbatches_calculator not found, using Apex version.")
            from apex.transformer.pipeline_parallel.utils import update_num_microbatches

        consumed_samples = state_dict['consumed_samples']
        self.data_sampler.init_consumed_samples = consumed_samples
        self.data_sampler.prev_consumed_samples = consumed_samples
        logging.info(f"Multimodal dataloader load state dict with consumed_samples {consumed_samples}")
        update_num_microbatches(
            consumed_samples=consumed_samples,
            consistency_check=False,
        )


class ClipDataModule(EnergonMultiModalDataModule):
    """
    A PyTorch Lightning DataModule for handling multimodal datasets with images and text.

    This data module is designed to work with multimodal datasets that involve both images and text.
    It provides a seamless interface to load training and validation data, manage batching, and handle
    the state of the data pipeline across training epochs. The module integrates with the Megatron-Energon
    framework for efficient data handling in large-scale distributed training.

    Attributes:
    path (str): Path to the energon dataset.
    tokenizer (Tokenizer): The tokenizer used for processing text.
    image_processor (ImageProcessor): The image processor used for preprocessing images.
    seq_length (int): The maximum sequence length for tokenized text.
    micro_batch_size (int): The batch size for training and validation.
    num_workers (int): Number of workers for data loading.
    pin_memory (bool): Whether to pin memory in the DataLoader.
    multimodal_sample_config (MultiModalSampleConfig): Configuration object for multimodal samples.
    task_encoder (MultiModalTaskEncoder): Encoder responsible for encoding and batching samples.
    init_global_step (int): The initial global step for the trainer, used for resuming training.
    data_sampler (SequentialMegatronSampler): Sampler responsible for generating sequential samples.
    train_dataloader_object (Optional): The DataLoader object for training data.
    val_dataloader_object (Optional): The DataLoader object for validation data.
    """

    def __init__(
        self,
        path: str,
        seq_length: int = 2048,
        micro_batch_size: int = 1,
        global_batch_size: int = 8,
        num_workers: int = 1,
        pin_memory: bool = True,
        task_encoder: DefaultTaskEncoder = None,
        use_train_split_for_val: bool = False,
        virtual_epoch_length: int = 1_000_000_000,  # a hack to avoid energon end of epoch warning
        packing_buffer_size: int | None = None,
        max_samples_per_sequence: int | None = None,
    ) -> None:
        """
        Initialize the EnergonMultiModalDataModule.

        Parameters:
        path (str): Path to the dataset.
        tokenizer (Tokenizer): The tokenizer used for processing text.
        image_processor (ImageProcessor): The image processor used for preprocessing images.
        seq_length (int, optional): The maximum sequence length for tokenized text. Defaults to 2048.
        micro_batch_size (int, optional): The batch size for training and validation. Defaults to 1.
        num_workers (int, optional): Number of workers for data loading. Defaults to 1.
        pin_memory (bool, optional): Whether to pin memory in the DataLoader. Defaults to True.
        """

        super().__init__(
            path=path,
            tokenizer=None,
            image_processor=None,
            seq_length=seq_length,
            micro_batch_size=micro_batch_size,
            global_batch_size=global_batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            task_encoder=task_encoder,
        )
        self.use_train_split_for_val = use_train_split_for_val
        self.virtual_epoch_length = virtual_epoch_length
        self.num_workers_val = 1
        self.packing_buffer_size = packing_buffer_size
        self.max_samples_per_sequence = max_samples_per_sequence

    def datasets_provider(self, worker_config, split: Literal['train', 'val'] = 'val'):
        """
        Provide the dataset for training or validation.

        This method retrieves the dataset for the specified split (either 'train' or 'val') and configures
        it according to the worker configuration.

        Parameters:
        worker_config: Configuration for the data loader workers.
        split (Literal['train', 'val'], optional): The data split to retrieve ('train' or 'val'). Defaults to 'val'.

        Returns:
        Dataset: The dataset configured for the specified split.
        """
        if split not in {'train', 'val'}:
            raise ValueError("Invalid value for split. Allowed values are 'train' or 'val'.")
        if self.use_train_split_for_val:
            split = 'train'
        _dataset = get_train_dataset(
            self.path,
            batch_size=self.micro_batch_size,
            task_encoder=self.task_encoder,
            worker_config=worker_config,
            max_samples_per_sequence=self.max_samples_per_sequence,
            shuffle_buffer_size=None,
            split_part=split,
            virtual_epoch_length=self.virtual_epoch_length,
            packing_buffer_size=self.packing_buffer_size,
        )
        return _dataset

    def val_dataloader(self) -> EVAL_DATALOADERS:
        """
        Initialize and return the validation DataLoader.

        This method initializes the DataLoader for the validation dataset. It ensures that the parallel state
        is initialized correctly for distributed training and returns a configured DataLoader object.

        Returns:
        EVAL_DATALOADERS: The DataLoader for the validation dataset.
        """
        if self.use_train_split_for_val:
            return self.train_dataloader()
        if self.val_dataloader_object:
            return self.val_dataloader_object

        if not parallel_state.is_initialized():
            message = (
                "Muiltimodal val data loader parallel state is not initialized "
                f"using default worker config with no_workers {self.num_workers}"
            )
            logging.info(message)

            worker_config = WorkerConfig.default_worker_config(self.num_workers_val)
        else:
            rank = parallel_state.get_data_parallel_rank()
            world_size = parallel_state.get_data_parallel_world_size()
            data_parallel_group = parallel_state.get_data_parallel_group()

            logging.info(f"rank {rank} world_size {world_size} data_parallel_group {data_parallel_group}")
            worker_config = WorkerConfig(
                rank=rank,
                world_size=world_size,
                num_workers=self.num_workers_val,
                data_parallel_group=data_parallel_group,
                worker_debug_path=None,
                worker_log_level=0,
            )
        val_dataset = self.datasets_provider(worker_config, split='val')
        energon_loader = get_savable_loader(val_dataset, worker_config=worker_config)
        self.val_dataloader_object = energon_loader
        return self.val_dataloader_object

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """
        Load the state of the data module from a checkpoint.

        This method is called when loading a checkpoint. It restores the state of the data module,
        including the state of the dataloader and the number of consumed samples.

        Parameters:
        state_dict (Dict[str, Any]): The state dictionary containing the saved state of the data module.
        """
        try:
            super().load_state_dict(state_dict)
        except Exception as e:
            logging.warning(f"datamodule.load_state_dict failed  {e}")

class SequentialMegatronSampler(MegatronDataSampler):
    """
    A data sampler for sequential sampling in Megatron, designed to handle large datasets efficiently.

    This class extends the MegatronDataSampler to support sequential sampling for large datasets.
    It includes functionality for handling micro-batches and tracking consumed samples across training steps.

    Attributes:
    seq_len (int): The sequence length for each sample.
    micro_batch_size (int): The number of samples in each micro-batch.
    init_consumed_samples (int): The initial number of samples that have been consumed (used for resuming training).
    prev_consumed_samples (int): Tracks the number of consumed samples before the current step.
    if_first_step (int): Flag to indicate if it's the first training step.
    prev_global_batch_size (Optional[int]): The global batch size from the previous step.
    init_global_step (int): The initial global step at the start of training.
    """

    def __init__(
        self,
        seq_len: int,
        micro_batch_size: int = 4,
        global_batch_size: int = 8,
        init_consumed_samples: int = 0,
        decoder_seq_len: Optional[int] = None,
        init_global_step=0,
    ):
        """
        Initialize the SequentialMegatronSampler.

        Parameters:
        seq_len (int): The sequence length for each sample.
        micro_batch_size (int, optional): The number of samples in each micro-batch. Defaults to 4.
        init_consumed_samples (int, optional): The initial number of samples that have been consumed. Defaults to 0.
        init_global_step (int, optional): The initial global step at the start of training. Defaults to 0.
        """
        super().__init__(
            seq_len=seq_len,
            decoder_seq_len=decoder_seq_len,
            micro_batch_size=micro_batch_size,
            global_batch_size=global_batch_size,
            init_consumed_samples=init_consumed_samples,
            init_global_step=init_global_step,
        )

    def transform_dataloader(self, dataloader: DataLoader) -> DataLoader:
        """
        Transform the DataLoader for sequential sampling.

        This method returns the DataLoader as is, but it can be overridden to apply specific transformations to
        the DataLoader if needed.

        Parameters:
        dataloader (DataLoader): The original DataLoader to be transformed.

        Returns:
        DataLoader: The transformed DataLoader.
        """
        return dataloader

    @property
    def megatron_data_kwargs(self) -> Dict[str, Any]:
        """
        Return the keyword arguments required for Megatron data handling.

        This property provides the necessary arguments that Megatron uses to handle data, including sequence length,
        micro-batch size, and the number of micro-batches.

        Returns:
        Dict[str, Any]: A dictionary containing the Megatron data handling arguments.
        """
        return {
            "seq_length": self.seq_len,
            "micro_batch_size": self.micro_batch_size,
            "num_microbatches": self.num_microbatches,
        }

def cook_raw_iamges(sample: dict) -> dict:
    """
    Processes a raw sample dictionary from energon dataset and returns a new dictionary with specific keys.

    Args:
        sample (dict): The input dictionary containing the raw sample data.

    Returns:
        dict: A new dictionary containing the processed sample data with the following keys:
            - All keys from the result of `basic_sample_keys(sample)`
            - 'jpg': original images
            - 'png': contains control images
            - 'txt': contains raw text
    """
    if "jpg" not in sample or "txt" not in sample:
        logging.info(f"Raw sample {sample} does not contain a jpg or txt file")
        raise SkipSample

    return dict(
        **basic_sample_keys(sample),
        image=sample['jpg'],
        txt=sample['txt'],
        image_12=sample['txt']
    )


class RawImageDiffusionTaskEncoder(DefaultTaskEncoder, IOMixin):
    cookers = [
        # Cooker(cook),
        Cooker(cook_raw_iamges),
    ]

# megatron.energon.flavors.crude.CrudeWebdataset
class ClipTaskEncoder(DefaultTaskEncoder, IOMixin):
    cookers = [Cooker(cook_raw_iamges)]
    def __init__(self, img_h=224, img_w=224, img_mean=None, img_std=None, max_length=77,
                 is_train=True):
        super().__init__()
        logging.warning(f"Processor or tokenizer are not provided! Fall back to `openai/clip-vit-large-patch14`.")
        from transformers import AutoProcessor
        from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer

        processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")
        self.tokenizer = AutoTokenizer("openai/clip-vit-large-patch14")
        self.image_processor = processor.image_processor
        # self.transform =  torchvision_transforms.Compose([
        #     torchvision_transforms.Resize((224, 224))])

        img_size = (img_h, img_w)
        # TODO(ask Yu): I don't know how can I pass training flag to Task encoder
        self.img_transform = image_transform(
            img_size,
            is_train=is_train,
            mean=img_mean,
            std=img_std,
        )
        self.toPIL = transforms.ToPILImage()
        self.max_length = max_length


    def encode_sample(self, sample: dict) -> dict:
        sample_new = {}
        # transforms.ToPILImage()(sample["image"])
        sample_new["images"] = self.img_transform(self.toPIL(sample["image"]))
        sample_new["captions"] = self.tokenizer.tokenizer(sample["txt"], return_tensors="pt", truncation=True,
                                                          padding='max_length', max_length=self.max_length).input_ids
        return sample_new


