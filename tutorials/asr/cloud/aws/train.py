from pathlib import Path

import sagemaker
import wget
from omegaconf import OmegaConf
from sagemaker import get_execution_role
from sagemaker.pytorch import PyTorch

from nemo.utils.notebook_utils import download_an4

sess = sagemaker.Session()

code_dir = Path('./code/')
config_dir = code_dir / 'conf/'
data_dir = Path('./data/')

code_dir.mkdir(exist_ok=True)
config_dir.mkdir(exist_ok=True)
config_path = str(config_dir / "config.yaml")
wget.download("https://raw.githubusercontent.com/NVIDIA/NeMo/main/examples/asr/conf/config.yaml", config_path)
wget.download(
    "https://raw.githubusercontent.com/NVIDIA/NeMo/main/examples//asr/asr_ctc/speech_to_text_ctc.py", str(code_dir)
)
with open(code_dir / 'requirements.txt', 'w') as f:
    f.write("nemo_toolkit[all]")

conf = OmegaConf.load(config_path)
conf.model.train_ds.manifest_filepath = ("/opt/ml/input/data/training/an4/train_manifest.json",)
conf.model.validation_ds.manifest_filepath = "/opt/ml/input/data/testing/an4/test_manifest.json"
conf.model.pretrained_model_name = "nvidia/stt_en_conformer_ctc_large"
conf.trainer.accelerator = "gpu"
conf.trainer.max_epochs = 1
OmegaConf.save(conf, config_dir / 'config.yaml')

# within the SageMaker container, mount_dir will be where our data is stored.
download_an4(
    data_dir=str(data_dir),
    train_mount_dir="/opt/ml/input/data/training/",
    test_mount_dir="/opt/ml/input/data/testing/",
)

# Upload to the default bucket
prefix = "an4"
bucket = sess.default_bucket()
# loc = sess.upload_data(path=str(data_dir), bucket=bucket, key_prefix=prefix)
loc = "s3://sagemaker-us-west-1-314164138468/an4"
print(loc)
channels = {"training": loc, "testing": loc}

role = get_execution_role()

output_path = "s3://" + sess.default_bucket() + "/nemo-output/"

local_mode = True

if local_mode:
    instance_type = "local_gpu"
else:
    instance_type = "ml.p2.xlarge"

est = PyTorch(
    entry_point="speech_to_text_ctc.py",
    source_dir="code",  # directory of your training script
    role=role,
    instance_type=instance_type,
    instance_count=1,
    framework_version="1.12.0",
    py_version="py38",
    volume_size=250,
    output_path=output_path,
    hyperparameters={'config-path': 'conf'},
)

est.fit(inputs=channels)
