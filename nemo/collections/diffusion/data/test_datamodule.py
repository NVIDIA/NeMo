import os
import time
import nemo_run as run
from nemo.collections.diffusion.data.diffusion_taskencoder import BasicDiffusionTaskEncoder
from nemo.collections.diffusion.train import multimodal_datamodule
import fiddle as fdl
import pytest
import torch
from megatron.core import parallel_state
import torch.autograd.profiler as profiler
from tqdm import tqdm
import numpy as np

from nemo.collections.multimodal.data.energon.base import SimpleMultiModalDataModule

# Fixture to initialize distributed training only once
@pytest.fixture(scope="session", autouse=True)
def initialize_distributed():
    if not torch.distributed.is_initialized():
        rank = int(os.environ['LOCAL_RANK'])
        world_size = torch.cuda.device_count()
        torch.cuda.set_device(rank)
        torch.distributed.init_process_group(backend='nccl', world_size=world_size, rank=rank)
        parallel_state.initialize_model_parallel()

# Fixture to get the value of the custom command-line option
@pytest.fixture
def path():
    return os.getenv('DATA_DIR')

def test_datamodule(path): 
    config = multimodal_datamodule()
    config.path = path
    config.num_workers = 120
    config.seq_length = 260
    config.task_encoder.seq_length = 260
    datamodule = fdl.build(config)
    # datamodule = SimpleMultiModalDataModule(
    #     path=path,
    #     seq_length=260,
    #     micro_batch_size=1,
    #     num_workers=256,
    #     tokenizer=None,
    #     image_processor=None,
    #     task_encoder=BasicDiffusionTaskEncoder(seq_length=260, text_embedding_padding_size=512,
    #     ),
    # )
    

    
    for i, batch in enumerate(datamodule.train_dataloader()):
        print(batch['seq_len_q'])
        if i == 1:
            start_time = time.time()
        if i > 100:
            break

    elapsed_time = time.time() - start_time
    print(f"Elapsed time for loading 100 batches: {elapsed_time} seconds, {elapsed_time/100} seconds per batch")

def test_taskencoder():
    taskencoder = BasicDiffusionTaskEncoder(
        text_embedding_padding_size=512,
        seq_length=260,
    )

    start_time = time.time()
    for i in tqdm(range(100)):
        sample = {
            'pth': torch.randn(3, 1, 30, 30),
            'pickle': np.random.randn(256, 1024),
            'json': {'image_height': 1, 'image_width': 1},
        }
        taskencoder.encode_sample(sample)

    elapsed_time = time.time() - start_time
    print(f"Elapsed time for loading 100 batches: {elapsed_time} seconds")
