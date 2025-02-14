from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

import torch
from megatron.energon import SimilarityInterleavedSample, VQASample, batch_list, batch_pad_stack, stateless
from torch.nn.utils.rnn import pad_sequence

from nemo.collections.multimodal.data.energon.config import ImageTextRawBatch, ImageTextSample, MultiModalSampleConfig
from nemo.collections.multimodal.data.energon.sample_encoder import SampleEncoder, VQASampleEncoder
from nemo.collections.multimodal.data.energon.task_encoder import MultiModalTaskEncoder
from nemo.collections.vlm.llava_next.data.interleaved_sample_encoder import LlavaNextSimilarityInterleavedSampleEncoder
from nemo.collections.vlm.llava_next.data.sample import LlavaNextTextRawBatch, LlavaNextTextSample, \
    PackedLlavaNextTextSample, PackedLlavaNextTextRawBatch
from nemo.collections.vlm.llava_next.data.vqa_sample_encoder import LlavaNextSampleEncoder
from nemo.utils import logging
from transformers import AutoProcessor


class LlavaNextTaskEncoder(MultiModalTaskEncoder):
    """LlavaNextTaskEncoder"""

    def __init__(self, tokenizer, image_processor, multimodal_sample_config,
                 packed_sequence=False,
                 packed_sequence_size=-1,
                 num_image_embeddings_per_tile=576,
                 ):
        """
        Initialize the LlavaNextTaskEncoder.

        This encoder extends MultiModalTaskEncoder to specifically handle LlavaNeXT,
        overriding  encoders for VQA sample type.

        Parameters:
        tokenizer (Tokenizer): The tokenizer for processing text data across sample types.
        image_processor (ImageProcessor): The image processor for preprocessing images.
        multimodal_sample_config (MultiModalSampleConfig): Configuration settings for multimodal samples.

        """
        super().__init__(tokenizer, image_processor, multimodal_sample_config, packed_sequence,
                         packed_sequence_size, num_image_embeddings_per_tile)
        self.encoders: Dict[str, SampleEncoder] = {
            VQASample.__name__: LlavaNextSampleEncoder(tokenizer, image_processor, multimodal_sample_config),
            SimilarityInterleavedSample.__name__: LlavaNextSimilarityInterleavedSampleEncoder(
                tokenizer=tokenizer, image_processor=image_processor, multimodal_sample_config=multimodal_sample_config
            ),
        }

    def batch(self, samples: List[Union[LlavaNextTextSample, PackedLlavaNextTextSample]]
              ) -> Union[LlavaNextTextRawBatch, PackedLlavaNextTextRawBatch]:
        """
        Batch multiple encoded samples into a single batch structure for model input.

        This method combines individual sample fields (keys, images, tokens, labels, etc.) and
        pads or stacks them as needed to create a unified batch.

        Parameters:
        samples (List[LlavaNextTextSample]): A list of LlavaNextTextSample instances to be batched.

        Returns:
        LlavaNextTextRawBatch: A batch containing all input samples' images, tokens, labels,
            loss masks, and other metadata prepared for model processing.
        """
        # import pdb; pdb.set_trace()
        if self.packed_sequence:
            if len(samples) > 1:
                raise ValueError(
                    "Micro batch size should be 1 when training with packed sequence, but your micro batch size "
                    f"is {len(samples)}. \nThe following config is equivalent to your current setting for "
                    f"a packed dataset. Please update your config to the following: \n"
                    f"Set micro batch size to 1 (currently {len(samples)})\n"
                    f"Set global batch size to `global_batch_size // {len(samples)}` "
                    f"Set packed sequence length to `original_sample_seq_len * {len(samples)}` "
                    f"(currently {self.packed_sequence_size}) \n"
                    f"For details please visit "
                    f"https://docs.nvidia.com/nemo-framework/user-guide/latest/sft_peft/packed_sequence.html"
                )
            # The batching are taken care by packing.
            sample = samples[0]
            return PackedLlavaNextTextRawBatch(
                __keys__=sample.__key__,
                images=sample.images,
                tokens=sample.tokens,
                labels=sample.labels,
                loss_mask=sample.loss_mask,
                num_media_tiles=sample.num_media_tiles,
                image_sizes=sample.image_sizes,
                attention_mask=sample.attention_mask,
                position_ids=sample.position_ids,
                packed_seq_params=sample.packed_seq_params,
            )
        else:
            keys, images, tokens, labels, loss_mask, num_media_tiles, image_sizes, attention_mask = (
                [],
                [],
                [],
                [],
                [],
                [],
                [],
                [],
            )
            for sample in samples:
                keys.append(sample.__key__)
                images.append(sample.images)
                tokens.append(sample.tokens)
                labels.append(sample.labels)
                loss_mask.append(sample.loss_mask)
                num_media_tiles.append(sample.num_media_tiles)
                image_sizes.append(sample.image_sizes)
                attention_mask.append(sample.attention_mask)

            batch_keys = batch_list(keys)

            batch_images = torch.cat(images, dim=0)

            batch_tokens = pad_sequence(tokens, batch_first=True)
            batch_labels = pad_sequence(labels, batch_first=True)
            image_sizes = torch.cat(image_sizes, dim=0)
            batch_loss_mask = batch_pad_stack(loss_mask)
            batch_attention_mask = batch_pad_stack(attention_mask)
            batch_list_num_media_tiles = batch_list(num_media_tiles)
            # if batch_list_num_media_tiles is nested lists, each sample has multiple images with different tiles
            # we need to flatten the list so len is num_images (in the batch)
            # image_sizes is also expected to be num_images, 2
            batch_list_num_media_tiles = flatten_if_nested(batch_list_num_media_tiles)
            batch_num_media_tiles = torch.tensor(batch_list_num_media_tiles, dtype=torch.int)

            assert (
                image_sizes.shape[0] == batch_num_media_tiles.shape[0]
            ), "image_sizes and batch_num_media_tiles must have the same length"

            return LlavaNextTextRawBatch(
                    __keys__=batch_keys,
                    images=batch_images,
                    tokens=batch_tokens,
                    labels=batch_labels,
                    loss_mask=batch_loss_mask,
                    num_media_tiles=batch_num_media_tiles,
                    image_sizes=image_sizes,
                    attention_mask=batch_attention_mask,
                )



    def select_samples_to_pack(self, samples: List[Union[LlavaNextTextSample, PackedLlavaNextTextSample]]):
        """Selects which samples will be packed together.

        NOTE: Energon dataloader calls this method internally if packing is used.
        Please see https://nvidia.github.io/Megatron-Energon/packing.html
        """
        from nemo.collections.vlm.neva.data.sequence_packing import greedy_knapsack, predict_seq_len_llava_next
        # import pdb; pdb.set_trace()
        media_token_id = self.sample_config.image_token.token_id

        lengths = [
            predict_seq_len_llava_next(
                sample.tokens,
                media_token_index=media_token_id,
                num_image_embeddings_per_tile=self.num_image_embeddings_per_tile,
                num_media_tiles=sample.num_media_tiles
            )
            for sample in samples
        ]

# predict_seq_len_llava_next( samples[0].tokens, media_token_index=self.sample_config.image_token.token_id, num_image_embeddings_per_tile=self.num_image_embeddings_per_tile, num_media_tiles=samples[0].num_media_tiles)
        packed_samples = greedy_knapsack(lengths, samples, self.packed_sequence_size)
        avg_samples_per_bin = round(len(lengths) / len(packed_samples))
        logging.info(
            f"[Seq Packing Info] - Packing seq len: {self.packed_sequence_size}, "
            f"Buffered samples: {len(lengths)}, Total number of bins: {len(packed_samples)}, "
            f"Average samples per bin: {avg_samples_per_bin}"
        )
        return packed_samples

    @stateless
    def pack_selected_samples(self, samples):
        """
        Function to pack a list of ImageTaskSample into a single ImageTaskSamplePacked.

        NOTE: Energon dataloader calls this method internally if packing is used.
        Please see https://nvidia.github.io/Megatron-Energon/packing.html

        Args:
            samples: List of ImageTaskSample instances to pack into one sample.

        Returns:
            ImageTaskSamplePacked instance.
        """
        # import pdb; pdb.set_trace()
        from nemo.collections.vlm.neva.data.sequence_packing import convert_to_packed

        keys, images, tokens, labels, loss_mask, num_media_tiles, image_sizes, attention_mask = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )
        for sample in samples:
            num_media_tiles.append(sample.num_media_tiles)
            image_sizes.append(sample.image_sizes)

        image_sizes = torch.cat(image_sizes, dim=0)

        # num_media_tiles: [3, 3, 3, 3]: Shape torch.Size([4])
        batch_list_num_media_tiles = batch_list(num_media_tiles)
        # if batch_list_num_media_tiles is nested lists, each sample has multiple images with different tiles
        # we need to flatten the list so len is num_images (in the batch)
        # image_sizes is also expected to be num_images, 2
        batch_list_num_media_tiles = flatten_if_nested(batch_list_num_media_tiles)
        batch_num_media_tiles = torch.tensor(batch_list_num_media_tiles, dtype=torch.int)


        packed_images = torch.cat([sample.images for sample in samples], dim=0)
        media_token_id = self.sample_config.image_token.token_id
        packed_tokens, packed_labels, packed_position_ids, packed_loss_mask, packed_seq_params = convert_to_packed(
            tokens=[sample.tokens for sample in samples],
            labels=[sample.labels for sample in samples],
            num_image_embeddings_per_tile=self.num_image_embeddings_per_tile,
            media_token_index=media_token_id,
            ignore_index=self.sample_config.ignore_place_holder,
        )

        return PackedLlavaNextTextSample(
            __key__=",".join([s.__key__ for s in samples]),
            __restore_key__=(),  # Will be set by energon based on `samples`
            tokens=packed_tokens,
            labels=packed_labels,
            images=packed_images,
            position_ids=packed_position_ids,
            loss_mask=packed_loss_mask,
            packed_seq_params=packed_seq_params,
            attention_mask=None, # We don't need attention mask for packed samples
            num_media_tiles=sample.num_media_tiles,
            image_sizes=sample.image_sizes,
        )

        PackedLlavaNextTextRawBatch(
                        __keys__=sample.__key__,
                        images=sample.images,
                        tokens=sample.tokens,
                        labels=sample.labels,
                        loss_mask=sample.loss_mask,
                        num_media_tiles=sample.num_media_tiles,
                        image_sizes=sample.image_sizes,
                        attention_mask=sample.attention_mask,
                        position_ids=sample.position_ids,
                        packed_seq_params=sample.packed_seq_params,
                    )


from itertools import chain


def flatten_if_nested(lst):
    # Check if the first element is a list (assuming consistent structure)
    if any(isinstance(i, list) for i in lst):
        return list(chain.from_iterable(lst))
    return lst


if __name__ == '__main__':
    import argparse

    from megatron.energon import WorkerConfig, get_loader, get_train_dataset
    from transformers import AutoProcessor

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True, help='path to the dataset directory')
    args = parser.parse_args()
    processor = AutoProcessor.from_pretrained("llava-hf/llava-v1.6-vicuna-7b-hf")
    tokenizer = processor.tokenizer
    multimodal_sample_config = MultiModalSampleConfig()
    multimodal_sample_config.conversation_template_config.system = None

    worker_config = WorkerConfig.default_worker_config(0)
    multimodal_sample_config = MultiModalSampleConfig()
    multimodal_sample_config.image_following_text = False
    train_loader = get_loader(
        get_train_dataset(
            args.data_path,
            batch_size=4,
            shuffle_buffer_size=100,
            max_samples_per_sequence=100,
            task_encoder=LlavaNextTaskEncoder(
                tokenizer=tokenizer,
                image_processor=processor.image_processor,
                multimodal_sample_config=MultiModalSampleConfig()
            ),
            worker_config=worker_config,
        ),
        worker_config=worker_config,
    )

    print(f"data loader length {len(train_loader)}")
    for index, each_batch in enumerate(train_loader):
        print(f"batch index {index} tokens shape {each_batch['tokens'].shape} ")
