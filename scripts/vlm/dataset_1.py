# Function to generate random string of a specified length
import random
import string
import time

import torch
from megatron.energon.epathlib import EPath
from megatron.energon.flavors.webdataset.structs import ShardInfo
from tqdm import tqdm


def generate_random_string(length):
    # Define the characters you want to use
    characters = string.ascii_letters + string.digits + string.punctuation  # Letters, digits, and punctuation
    # Use random.choices() to select random characters
    random_string = ''.join(random.choices(characters, k=length))
    return random_string



saving = []
for i in tqdm(range(3765636)):
    random_string = generate_random_string(10)
    saving.append(ShardInfo(name=random_string,
                            path=EPath(f"/{random_string}"),
                            offset = 0, count = 1000, byte_offset = None, byte_size = None))

start = time.time()
print(f"saving: {len(saving)}, start: {start}")
# Save as pkl
torch.save(saving, './dataset_1_data.pt')
print(f"Dataset saved in {time.time() - start} seconds")


train_ds = get_train_dataset(
    '/my/dataset/path',
    batch_size=32,
    shuffle_buffer_size=None,
    max_samples_per_sequence=None,
    worker_config=simple_worker_config,
)

data = SimpleMultiModalDataModule(
    # "/workspace/data/cc3m_training_single_sample",
    args.data_path,
    seq_length=decoder_seq_length,
    image_processor=None,
    micro_batch_size=args.mbs,
    global_batch_size=args.gbs,
    num_workers=8,
    task_encoder=task_encoder,
    tokenizer=task_encoder.tokenizer,
)

train_loader = get_loader(train_ds)

for batch in train_loader:
    # Do something with batch
    # Infer, gradient step, ...
    pass