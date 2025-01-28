from typing import Optional

from megatron.energon import DefaultTaskEncoder, basic_sample_keys, \
    Cooker, SkipSample
from torchvision import transforms

from nemo.collections.multimodal.data.clip.augmentations.augmentations import image_transform
from nemo.collections.multimodal.data.clip.clip_dataset import tokenize
from nemo.lightning.io.mixin import IOMixin
from nemo.utils import logging


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
    )


class ClipTaskEncoder(DefaultTaskEncoder, IOMixin):
    cookers = [Cooker(cook_raw_iamges)]
    def __init__(self, img_h: int = 224, img_w: int = 224, img_mean: int = None,
                 img_std: int = None, max_length: int = 77, tokenizer: Optional = None,
                 image_processor: Optional = None, is_train: bool =True):
        super().__init__()

        self.tokenizer = tokenizer
        self.image_processor = image_processor

        if image_processor is None or tokenizer is None:
            logging.warning(f"Processor or tokenizer are not provided! Fall back to `openai/clip-vit-large-patch14`.")
            from transformers import AutoProcessor
            from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer

            processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")
            self.tokenizer = AutoTokenizer("openai/clip-vit-large-patch14")
            self.image_processor = processor.image_processor

        img_size = (img_h, img_w)
        self.img_size = img_size

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
        sample_new["images"] = self.img_transform(sample["image"])
        sample_new["captions"] = tokenize(sample["txt"], self.tokenizer, context_length=self.max_length)
        return sample_new


