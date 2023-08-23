# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""

export PYTHONPATH="/lustre/fsw/portfolios/adlr/users/chzhu/sources/NeMo_sandeepsub_ul2:${PYTHONPATH}"
python scripts/nlp_language_modeling/ul2/ul2_expand_output_vocab.py \
    --nemo_file="/lustre/fsw/portfolios/adlr/users/chzhu/checkpoints/megatron_gpt_843M.nemo" \
    --num_sentinel_tokens=2000 \
    --out_file="/lustre/fsw/portfolios/adlr/users/chzhu/checkpoints/megatron_gpt_843M_extras_2000_proper.nemo" \
    --alpha=0.5

"""

import argparse
import os
import tempfile

import torch
from omegaconf import OmegaConf, open_dict
from pytorch_lightning.trainer.trainer import Trainer
from torch.utils.data import DataLoader, Dataset

from nemo.collections.nlp.models.language_modeling.megatron_base_model import MegatronBaseModel
from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer
from nemo.collections.nlp.parts.nlp_overrides import NLPSaveRestoreConnector
from nemo.utils import AppState, model_utils


def parse_args():
    parser = argparse.ArgumentParser(description='Expand output vocab for UL2 model')
    parser.add_argument('-i', '--nemo_file', type=str, help='Path to the model checkpoint')
    parser.add_argument('-o', '--out_file', type=str, default=None, help='Path to save the modified model checkpoint')

    parser.add_argument(
        '-m',
        '--tok_model_suffix',
        type=str,
        default='mt_nlg_plus_multilingual_ja_zh_the_stack_frac_015_256k.model',
        help='Tokenizer model to use for parsing the model checkpoint',
    )
    parser.add_argument(
        '-n', '--num_sentinel_tokens', type=int, required=True, help='Number of sentinal tokens to add to the vocab'
    )

    parser.add_argument(
        '--library', type=str, default='sentencepiece', help='Library to use for parsing the model checkpoint'
    )
    parser.add_argument('--type', type=str, default='GPT2BPETokenizer', help='Tokenizer type')
    parser.add_argument('--vocab_file', type=str, default=None, help='Tokenizer vocab file')
    parser.add_argument('--merge_file', type=str, default=None, help='Tokenizer merge file')
    parser.add_argument('--sentencepiece_legacy', type=str, default="true", help='Sentencepiece legacy mode')
    parser.add_argument('--alpha', type=float, default=1.0, help='Scale the existing extra ids and copied extra ids.')

    args = parser.parse_args()

    # Parse legacy mode of sentencepiece
    if args.sentencepiece_legacy is not None and str(args.sentencepiece_legacy).lower() in [
        'true',
        '1',
        't',
        'y',
        'yes',
    ]:
        args.sentencepiece_legacy = True
    else:
        args.sentencepiece_legacy = False

    # If sentencepiece, force to legacy mode
    if args.library == 'sentencepiece':
        args.sentencepiece_legacy = True

    return args


def update_model_config(tokenizer_cfg, config):
    # Update the tokenizer configs here.
    for key in tokenizer_cfg.keys():
        if key == 'model':
            # We never override the tokenizer model, keep this unchanged.
            continue
        else:
            config.tokenizer[key] = tokenizer_cfg[key]
    return config


def build_tokenizer(args: argparse.Namespace, tok_model_path: str):
    tokenizer = get_nmt_tokenizer(
        library=args.library,
        model_name=args.type,
        tokenizer_model=tok_model_path,
        vocab_file=args.vocab_file,
        merges_file=args.merge_file,
        delimiter=None,
        legacy=args.sentencepiece_legacy,
    )

    # Pack config to omegaconf
    cfg = OmegaConf.create(
        dict(
            library=args.library,
            type=args.type,
            model=tok_model_path,
            vocab_file=args.vocab_file,
            merges_file=args.merge_file,
            sentencepiece_legacy=args.sentencepiece_legacy,
            num_sentinel_tokens=args.num_sentinel_tokens,
            expand_tokens_dataset_type="ul2",
            add_sentinel_tokens_first=True,  # since we already have some sentinel tokens, we don't want other special tokens to be added in between
        )
    )

    #  Add UL2 tokens
    tokenizer = MegatronBaseModel._expand_tokenizer(tokenizer, cfg, dataset_type=cfg.expand_tokens_dataset_type)

    return tokenizer, cfg


def get_word_embedding_key(state_dict) -> str:
    name = None
    if 'model.language_model.embedding.word_embeddings.weight' in state_dict:
        name = 'model.language_model.embedding.word_embeddings.weight'
    elif 'model.encoder_embedding.word_embeddings.weight' in state_dict:
        name = 'model.encoder_embedding.word_embeddings.weight'
    elif 'model.decoder_embedding.word_embeddings.weight' in state_dict:
        name = 'model.decoder_embedding.word_embeddings.weight'
    elif 'model.word_embeddings.weight' in state_dict:
        name = 'model.word_embeddings.weight'

    return name


def get_output_layer_key(state_dict) -> str:
    name = None
    for key in state_dict.keys():
        if 'output_layer' in key:
            name = key
            break

    return name


def expand_tensor(model_cfg, tokenizer, key, state_dict, alpha):
    original_shape = state_dict[key].shape
    original_vocab_size = original_shape[0]
    print("Original shape :", original_shape)

    # Use vocab size as final expansion dim
    new_vocab_size = len(tokenizer.vocab)
    print("New vocab size :", new_vocab_size)

    # Add buffer of dummy tokens for divisibility of vocab tokens
    divisible_by_val = model_cfg.get('make_vocab_size_divisible_by', 1)
    if new_vocab_size % divisible_by_val != 0:
        dummy_tokens = divisible_by_val - (new_vocab_size % divisible_by_val)
        final_vocab_size = new_vocab_size + dummy_tokens
        print("Adding Dummy Tokens :", dummy_tokens)
    else:
        final_vocab_size = new_vocab_size

    # Final expanded shape
    final_vocab_shape = [final_vocab_size, original_shape[1]]
    new_shape = torch.Size(final_vocab_shape)
    print("New shape :", new_shape)

    # Expand vocab
    new_output_layer = torch.zeros(new_shape, dtype=state_dict[key].dtype)
    new_output_layer[:original_vocab_size, :] = state_dict[key].clone()

    # Copy scaled sentinal token weights to make the softmax divident roughly unchanged
    first_extra_id = tokenizer.tokens_to_ids(['<extra_id_0>'])[0]

    num_existing_extra_tokens = original_vocab_size - first_extra_id

    new_output_layer[original_vocab_size - num_existing_extra_tokens : original_vocab_size] *= alpha
    existing_embeddings = new_output_layer[
        original_vocab_size - num_existing_extra_tokens : original_vocab_size
    ].clone()

    for i in range(0, final_vocab_size - original_vocab_size, num_existing_extra_tokens):
        sidx = i + original_vocab_size
        eidx = sidx + num_existing_extra_tokens
        if new_output_layer[sidx:eidx].shape[0] == num_existing_extra_tokens:
            new_output_layer[sidx:eidx] = existing_embeddings
        else:
            # Last iteration with remainder.
            remainder = (final_vocab_size - original_vocab_size) % num_existing_extra_tokens
            new_output_layer[-remainder:] = existing_embeddings[-remainder:]

    # Perform assertion that data was correctly copied.
    assert torch.allclose(state_dict[key][:first_extra_id], new_output_layer[:first_extra_id])

    # Inplace replacement
    state_dict[key] = new_output_layer

    return state_dict


def process_model(args) -> str:
    connector = NLPSaveRestoreConnector()

    with tempfile.TemporaryDirectory() as tmpdir:
        # Extract the model from the checkpoint
        connector._unpack_nemo_file(args.nemo_file, tmpdir)
        import pdb

        pdb.set_trace()
        # Load the model config
        config_path = os.path.join(tmpdir, connector.model_config_yaml)

        all_files = os.listdir(tmpdir)
        for fname in all_files:
            if fname.endswith(args.tok_model_suffix):
                tok_model_path = os.path.join(tmpdir, fname)
                break
        tokenizer, tokenizer_cfg = build_tokenizer(args, tok_model_path)
        config = OmegaConf.load(config_path)

        # Update model config
        config = update_model_config(tokenizer_cfg, config)

        tp_size = config.get('tensor_model_parallel_size', 1)
        pp_size = config.get('pipeline_model_parallel_size', 1)

        if tp_size > 1:
            raise RuntimeError(
                "GPT model's vocab and output embeddings cannot be expanded for tensor parallelism > 1.\n"
                "Please use the `megatron_change_num_partitions.py` script to change the number of partitions.\n"
                "Then run this script to expand the vocabulary and output embeddings.\n"
                "Finally run the `megatron_change_num_partitions.py` script again to restore the number of partitions."
            )

        appstate = AppState()
        appstate.tensor_model_parallel_rank = 1
        appstate.pipeline_model_parallel_rank = 1
        appstate.tensor_model_parallel_size = tp_size
        appstate.pipeline_model_parallel_size = pp_size
        appstate.model_parallel_size = tp_size * pp_size

        input_embedding_size = None
        input_embed_processed = False

        output_layer_size = None
        output_layer_processed = False

        # Load the TP 1 PP Y model checkpoint

        # Only need to update 2 PP - PP 0 for the input embedding and PP -1 for the output layer.
        pp_checks = [0]
        if pp_size > 1:
            pp_checks = [pp_size - 1]

        for pp in pp_checks:
            for tp in range(1):  # tp size
                appstate.tensor_model_parallel_rank = tp
                appstate.pipeline_model_parallel_rank = pp

                checkpoint_path = os.path.join(tmpdir, connector.model_weights_ckpt)
                checkpoint_path = model_utils.inject_model_parallel_rank(checkpoint_path)

                print("Parsing checkpoint at location: ", checkpoint_path)
                state_dict = torch.load(checkpoint_path, map_location='cpu')

                # Get word embedding key
                word_embedding_key = get_word_embedding_key(state_dict)
                output_layer_key = get_output_layer_key(state_dict)

                if input_embedding_size is None and word_embedding_key is not None:
                    input_embedding_size = state_dict[word_embedding_key].shape

                if output_layer_size is None and output_layer_key is not None:
                    output_layer_size = state_dict[output_layer_key].shape

                # Expand input embedding
                if not input_embed_processed and word_embedding_key is not None:
                    print()
                    print(f"Found embedding weights ({output_layer_key}). Modifying...")

                    state_dict = expand_tensor(
                        model_cfg=config,
                        tokenizer=tokenizer,
                        key=word_embedding_key,
                        state_dict=state_dict,
                        alpha=args.alpha,
                    )

                # Expand output layer
                if not output_layer_processed and output_layer_key is not None:
                    print()
                    print(f"Found output layer weights ({output_layer_key}). Modifying...")

                    state_dict = expand_tensor(
                        model_cfg=config,
                        tokenizer=tokenizer,
                        key=output_layer_key,
                        state_dict=state_dict,
                        alpha=args.alpha,
                    )

                # Save the modified checkpoint inplace
                print()
                print("Saving state dict ...")
                torch.save(state_dict, checkpoint_path)

        # Save the full nemo file
        save_filepath = args.out_file
        if save_filepath is None:
            save_filepath = os.path.splitext(args.nemo_file)[0] + '_expanded_vocab.nemo'

        # Save NeMo config
        OmegaConf.save(config, config_path, resolve=True)

        connector._make_nemo_file_from_folder(save_filepath, tmpdir)
        return save_filepath


def main():
    args = parse_args()
    save_filepath = process_model(args)
    print("Finished saving NeMo file at: ", save_filepath)


if __name__ == '__main__':
    main()
