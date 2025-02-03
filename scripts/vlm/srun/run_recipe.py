
from nemo.collections import vlm



from nemo.collections.multimodal.data.energon import EnergonMultiModalDataModule
from nemo.collections.vlm.clip.data.clip_data_module import ClipTaskEncoder

# Paths and configuration
data_path = "/workspace/data/cc3m_training"
decoder_seq_length = 80
mbs = 500
gbs = 1000
num_workers = 16

# Load the task encoder for train and validation
train_task_encoder = ClipTaskEncoder(max_length=decoder_seq_length)
valid_task_encoder = ClipTaskEncoder(max_length=decoder_seq_length, is_train=False)
data = EnergonMultiModalDataModule(
    data_path,
    seq_length=decoder_seq_length,
    image_processor=None,
    micro_batch_size=mbs,
    global_batch_size=gbs,
    num_workers=num_workers,
    task_encoder=train_task_encoder,
    tokenizer=train_task_encoder.tokenizer,
    validation_task_encoder=valid_task_encoder,
    image_decode="pil",
    ignore_decoder_errors=True,
)
pretrain = vlm.clip_b32.pretrain_recipe(num_gpus_per_node=2)
pretrain.data = data
import nemo_run as run
run.run(pretrain, direct=True)