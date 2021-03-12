# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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


import glob
import json
import os
import pickle
import tarfile
import tempfile

import youtokentome as yttm
from pytorch_lightning import Trainer

from nemo.collections.nlp.data.machine_translation.machine_translation_dataset import TranslationDataset
from nemo.collections.nlp.data.machine_translation.one_side_dataset import TranslationOneSideDataset
from nemo.collections.nlp.models.machine_translation.mt_enc_dec_config import MTEncDecModelConfig
from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer, get_tokenizer
from nemo.utils import logging


class MTDataPreproc:
    """ Automatically trains tokenizers and preprocesses machine translation data based on the MTEncDecModelConfig.
        For training NMT models with datasets larger than 5M sentence pairs, 
        it can be inefficient to train them without first creating a tarred dataset. 
        If the user wants to change the tokenizer, vocab size, or batch size, for example, 
        they must reprocess the data with the correct configuration. 
        With MTDataPreproc users can sweep through data configurations and the tarred dataset will 
        be automatically created according to the model configuration.
        To train tokenizer model and create tarred dataset specify in configuration:
            model.preproc_out_dir=/path/to/preproc_out
            model.encoder_tokenizer.vocab_size=32000
            model.decoder_tokenizer.vocab_size=32000 
            model.train_ds.use_tarred_dataset=True 
            model.train_ds.src_file_name=/path/to/src.txt
            model.train_ds.tgt_file_name=/path/to/tgt.txt
            model.train_ds.tokens_in_batch=16000 
        Once a dataset has been constructed based on this configuration, MTDataPreproc will not process it again.
        If a previously trained tokenizer model or tarred dataset is found, MTDataPreproc will not preprocess the data.

        Note: the only tokenizer currently supported is YouTokenToMe.
    """

    def __init__(self, cfg: MTEncDecModelConfig, trainer: Trainer = None) -> None:
        self._cfg = cfg
        self.global_rank = 0
        self.world_size = 1
        if trainer is not None:
            self.global_rank = (trainer.node_rank * trainer.num_gpus) + trainer.local_rank
            self.world_size = trainer.num_nodes * trainer.num_gpus

        if hasattr(cfg, 'train_ds'):
            supported_tokenizers = ['yttm', 'huggingface']
            if (
                not cfg.encoder_tokenizer.get('library') in supported_tokenizers
                or not cfg.decoder_tokenizer.get('library') in supported_tokenizers
            ):
                raise NotImplementedError(f"Currently we only support {supported_tokenizers}.")

            # Prepare tokenizers
            if cfg.encoder_tokenizer.get('library') == 'yttm' or cfg.decoder_tokenizer.get('library') == 'yttm':

                # Train tokenizer models if using yttm and they don't exist
                if (
                    cfg.encoder_tokenizer.get('library') == 'yttm'
                    and cfg.encoder_tokenizer.get('tokenizer_model') is None
                ) or (
                    cfg.decoder_tokenizer.get('library') == 'yttm'
                    and cfg.decoder_tokenizer.get('tokenizer_model') is None
                ):
                    if cfg.get('preproc_out_dir') is None:
                        raise ValueError('Tokenizer model training required but cfg.preproc_out_dir is None.')
                    if cfg.train_ds.get('src_file_name') is None or cfg.train_ds.get('tgt_file_name') is None:
                        raise ValueError(
                            'src_file_name and tgt_file_name needed to train tokenizers but could not be found.'
                        )
                    # train tokenizer model on training data
                    self.encoder_tokenizer_model, self.decoder_tokenizer_model = MTDataPreproc.train_yttm_tokenizers(
                        out_dir=cfg.get('preproc_out_dir'),
                        src_fname=cfg.train_ds.get('src_file_name'),
                        tgt_fname=cfg.train_ds.get('tgt_file_name'),
                        shared_tokenizer=cfg.get('shared_tokenizer'),
                        encoder_tokenizer_vocab_size=cfg.encoder_tokenizer.get('vocab_size'),
                        decoder_tokenizer_vocab_size=cfg.decoder_tokenizer.get('vocab_size'),
                        encoder_tokenizer_name=cfg.encoder_tokenizer.get('library'),
                        decoder_tokenizer_name=cfg.decoder_tokenizer.get('library'),
                        encoder_tokenizer_coverage=cfg.encoder_tokenizer.get('coverage', 0.999),
                        decoder_tokenizer_coverage=cfg.decoder_tokenizer.get('coverage', 0.999),
                        global_rank=self.global_rank,
                    )
                    # update config
                    self._cfg.encoder_tokenizer.tokenizer_model = self.encoder_tokenizer_model
                    self._cfg.decoder_tokenizer.tokenizer_model = self.decoder_tokenizer_model
                else:
                    self.encoder_tokenizer_model = cfg.encoder_tokenizer.get('tokenizer_model')
                    self.decoder_tokenizer_model = cfg.decoder_tokenizer.get('tokenizer_model')

            self.encoder_tokenizer, self.decoder_tokenizer = self.get_enc_dec_tokenizers(
                encoder_tokenizer_name=cfg.encoder_tokenizer.get('library'),
                encoder_model_name=cfg.encoder.get('model_name'),
                encoder_tokenizer_model=self.encoder_tokenizer_model,
                encoder_bpe_dropout=cfg.encoder_tokenizer.get('bpe_dropout', 0.0),
                decoder_tokenizer_name=cfg.decoder_tokenizer.get('library'),
                decoder_model_name=cfg.decoder.get('model_name'),
                decoder_tokenizer_model=self.decoder_tokenizer_model,
                decoder_bpe_dropout=cfg.decoder_tokenizer.get('bpe_dropout', 0.0),
            )

            # If using tarred dataset for training, automatically create it if needed
            if cfg.train_ds.get('use_tarred_dataset'):
                if cfg.train_ds.get('tar_files') is None or cfg.train_ds.get('metadata_file') is None:
                    if cfg.get('preproc_out_dir') is None:
                        raise ValueError('Data preprocessing required but cfg.preproc_out_dir is None.')
                    if cfg.train_ds.get('src_file_name') is None or cfg.train_ds.get('tgt_file_name') is None:
                        raise ValueError(
                            'src_file_name and tgt_file_name needed to create tarred dataset but could not be found.'
                        )
                    # Preprocess data and cache for use during training
                    if self.global_rank == 0:
                        logging.info(
                            f"Using tarred dataset for src: {cfg.train_ds.get('src_file_name')} and tgt: {cfg.train_ds.get('tgt_file_name')}"
                        )
                    self.train_tar_files, self.train_metadata_file = MTDataPreproc.preprocess_parallel_dataset(
                        clean=cfg.train_ds.clean,
                        src_fname=cfg.train_ds.get('src_file_name'),
                        tgt_fname=cfg.train_ds.get('tgt_file_name'),
                        out_dir=cfg.get('preproc_out_dir'),
                        encoder_tokenizer=self.encoder_tokenizer,
                        decoder_tokenizer=self.decoder_tokenizer,
                        max_seq_length=cfg.train_ds.get('max_seq_length', 512),
                        tokens_in_batch=cfg.train_ds.get('tokens_in_batch', 8192),
                        lines_per_dataset_fragment=cfg.train_ds.get('lines_per_dataset_fragment', 1000000),
                        num_batches_per_tarfile=cfg.train_ds.get('num_batches_per_tarfile', 1000),
                        min_seq_length=1,
                        pkl_file_prefix=cfg.train_ds.get('pkl_file_preifx', 'parallel'),
                        global_rank=self.global_rank,
                        world_size=self.world_size,
                    )
                    # update config
                    self._cfg.train_ds.tar_files = self.tar_files_to_string(self.train_tar_files)
                    self._cfg.train_ds.metadata_file = self.train_metadata_file
                    logging.info(
                        f"Using tarred dataset created at {self._cfg.train_ds.tar_files} and metadata created at {self._cfg.train_ds.metadata_file}"
                    )
                else:
                    self.train_tar_files = cfg.train_ds.get('tar_files')
                    self.train_metadata_file = cfg.train_ds.get('metadata_file')
                    logging.info(
                        f"Using tarred dataset from config at {self.train_tar_files} and metadata from {self.train_metadata_file}"
                    )

    def tar_files_to_string(self, tar_files):
        """ Tar files are generated in the following format: basename.number.tar
            Where number is an integer from 1 to the number of tar files.
            We convert this list to a string that can be used in the model config to specify
            tarred datasets: basename_OP_1..num_tar_files_CL_.tar 

        Args:
            tar_files (List[str]): List of tar files generated by preprocess_parallel_dataset
        """
        num_tar_files = len(tar_files)
        split_on_dot = tar_files[0].split('.')
        basename = '.'.join(split_on_dot[0:-2])
        tar_file_string = f'{basename}._OP_1..{num_tar_files}_CL_.tar'
        return tar_file_string

    @staticmethod
    def get_enc_dec_tokenizers(
        encoder_tokenizer_name=None,
        encoder_tokenizer_model=None,
        encoder_bpe_dropout=0.0,
        encoder_model_name=None,
        decoder_tokenizer_name=None,
        decoder_tokenizer_model=None,
        decoder_bpe_dropout=0.0,
        decoder_model_name=None,
    ):

        # if encoder_tokenizer_name != 'yttm' or decoder_tokenizer_name != 'yttm':
        #     raise NotImplementedError(f"Currently we only support yttm tokenizer.")

        encoder_tokenizer = get_nmt_tokenizer(
            library=encoder_tokenizer_name,
            model_name=encoder_model_name,
            tokenizer_model=encoder_tokenizer_model,
            bpe_dropout=encoder_bpe_dropout,
        )
        decoder_tokenizer = get_nmt_tokenizer(
            library=decoder_tokenizer_name,
            model_name=decoder_model_name,
            tokenizer_model=decoder_tokenizer_model,
            bpe_dropout=decoder_bpe_dropout,
        )

        return encoder_tokenizer, decoder_tokenizer

    @staticmethod
    def get_monolingual_tokenizer(
        tokenizer_name=None, tokenizer_model=None, bpe_dropout=0.0,
    ):
        if tokenizer_name != 'yttm':
            raise NotImplementedError(f"Currently we only support yttm tokenizer.")

        tokenizer = get_tokenizer(
            tokenizer_name=tokenizer_name, tokenizer_model=tokenizer_model, bpe_dropout=bpe_dropout,
        )

        return tokenizer

    # TODO: add local or global rank 0 decorator
    @staticmethod
    def preprocess_parallel_dataset(
        clean,
        src_fname,
        tgt_fname,
        out_dir,
        encoder_tokenizer,
        decoder_tokenizer,
        max_seq_length,
        min_seq_length,
        tokens_in_batch,
        lines_per_dataset_fragment,
        num_batches_per_tarfile,
        pkl_file_prefix,
        global_rank,
        world_size,
    ):
        """Create tarred dataset from large paired translation data.

        Args:
            clean (str): Cleans source and target sentences to get rid of noisy data.
            src_fname (str): path to source text data
            tgt_fname (str): path to target text data
            out_dir (str): path to write tarred dataset
            encoder_tokenizer (Any): tokenizer for encoder 
            decoder_tokenizer (Any): tokenizer for decoder
            max_seq_length (int): maximum sequence length 
            min_seq_length (int): minimum sequence length 
            tokens_in_batch (int): tokens per batch per GPU, effectively batch size 
            lines_per_dataset_fragment (int): number of lines to consider for bucketing and padding
            num_batches_per_tarfile (int): number of batches (pickle files) within each tarfile
        """

        os.makedirs(out_dir, exist_ok=True)

        tar_file_ctr = 1
        num_files_in_tar = 0
        num_lines = 0
        shard_num = 0
        global_batch_ctr = 0
        tmp_f_src = tempfile.NamedTemporaryFile(delete=False, mode='w')
        tmp_f_tgt = tempfile.NamedTemporaryFile(delete=False, mode='w')
        tar_file_path = os.path.join(out_dir, '%s-batches.tokens.%d.%d.tar' % (pkl_file_prefix, tokens_in_batch, 1))
        metadata_path = os.path.join(out_dir, f'metadata.tokens.{tokens_in_batch}.json')
        if global_rank == 0:
            if os.path.isfile(tar_file_path) and os.path.isfile(metadata_path):
                logging.info(
                    f'Tarred dataset {tar_file_path} and metadata file {metadata_path} exists and will be used. Remove if reprocessing.'
                )
            else:
                tar_file_ptr = tarfile.open(tar_file_path, 'w')
                with open(src_fname, 'r') as f_src, open(tgt_fname) as f_tgt:
                    for src_line, tgt_line in zip(f_src, f_tgt):
                        tmp_f_src.write(src_line)
                        tmp_f_tgt.write(tgt_line)
                        num_lines += 1

                        if num_lines == lines_per_dataset_fragment:
                            tmp_f_src.close()
                            tmp_f_tgt.close()
                            (
                                tar_file_ptr,
                                global_batch_ctr,
                                num_files_in_tar,
                                tar_file_ctr,
                            ) = MTDataPreproc.write_parallel_batches_to_tarfiles(
                                out_dir=out_dir,
                                num_batches_per_tarfile=num_batches_per_tarfile,
                                clean=clean,
                                max_seq_length=max_seq_length,
                                min_seq_length=min_seq_length,
                                src_fname=tmp_f_src.name,
                                tgt_fname=tmp_f_tgt.name,
                                num_tokens=tokens_in_batch,
                                encoder_tokenizer=encoder_tokenizer,
                                decoder_tokenizer=decoder_tokenizer,
                                num_files_in_tar=num_files_in_tar,
                                tar_file_ptr=tar_file_ptr,
                                tar_file_ctr=tar_file_ctr,
                                global_batch_ctr=global_batch_ctr,
                                pkl_file_prefix=pkl_file_prefix,
                            )

                            num_lines = 0
                            shard_num += 1

                            os.remove(tmp_f_src.name)
                            os.remove(tmp_f_tgt.name)

                            tmp_f_src = tempfile.NamedTemporaryFile(delete=False, mode='w')
                            tmp_f_tgt = tempfile.NamedTemporaryFile(delete=False, mode='w')

                tmp_f_src.close()
                tmp_f_tgt.close()
                (
                    tar_file_ptr,
                    global_batch_ctr,
                    num_files_in_tar,
                    tar_file_ctr,
                ) = MTDataPreproc.write_parallel_batches_to_tarfiles(
                    out_dir=out_dir,
                    num_batches_per_tarfile=num_batches_per_tarfile,
                    clean=clean,
                    max_seq_length=max_seq_length,
                    min_seq_length=min_seq_length,
                    src_fname=tmp_f_src.name,
                    tgt_fname=tmp_f_tgt.name,
                    num_tokens=tokens_in_batch,
                    encoder_tokenizer=encoder_tokenizer,
                    decoder_tokenizer=decoder_tokenizer,
                    num_files_in_tar=num_files_in_tar,
                    tar_file_ptr=tar_file_ptr,
                    tar_file_ctr=tar_file_ctr,
                    global_batch_ctr=global_batch_ctr,
                    pkl_file_prefix=pkl_file_prefix,
                )
                tar_file_ptr.close()
                os.remove(tmp_f_src.name)
                os.remove(tmp_f_tgt.name)

                if num_files_in_tar != num_batches_per_tarfile:
                    os.remove(
                        os.path.join(
                            out_dir, '%s-batches.tokens.%d.%d.tar' % (pkl_file_prefix, tokens_in_batch, tar_file_ctr)
                        )
                    )
                    global_batch_ctr -= num_files_in_tar
                    logging.info('Dropping %d batches because of overflow' % (num_files_in_tar))

                json.dump({'num_batches': global_batch_ctr}, open(metadata_path, 'w'))

        tar_file_paths = glob.glob(f'{out_dir}/{pkl_file_prefix}-batches.tokens.{tokens_in_batch}.*.tar')

        num_tar_files = len(tar_file_paths)
        if num_tar_files < world_size:
            raise ValueError(
                (
                    f'Number of tar files found: {num_tar_files} is less than world size: {world_size}. '
                    f'There should be at least one tar file per GPU (ideally many tar files per GPU). '
                    f'This may be due to dataset size, it is advisable to use at least 5M sentence pairs for tarred datasets. '
                    f'Decrease num_batches_per_tarfile or num_tokens_per_batch to increase the number of tarfiles. '
                    f'Also using shard_strategy=replicate will use all available tarfiles for every GPU. '
                )
            )

        return tar_file_paths, metadata_path

    @staticmethod
    def preprocess_monolingual_dataset(
        clean,
        fname,
        out_dir,
        tokenizer,
        max_seq_length,
        min_seq_length,
        tokens_in_batch,
        lines_per_dataset_fragment,
        num_batches_per_tarfile,
        pkl_file_prefix,
        global_rank,
        world_size,
    ):
        """Create tarred dataset from a large monolingual corpus.

        Args:
            clean (str): Cleans sentences to get rid of very long or short sentences.
            fname (str): Path to source text data
            out_dir (str): Path to write tarred dataset
            tokenizer (Any): Path to tokenizer model
            max_seq_length (int): maximum sequence length 
            min_seq_length (int): minimum sequence length 
            tokens_in_batch (int): tokens per batch per GPU, effectively batch size 
            lines_per_dataset_fragment (int): number of lines to consider for bucketing and padding
            num_batches_per_tarfile (int): number of batches (pickle files) within each tarfile
            global_rank (int): if set to zero, data will be processed on this node
            world_size (int): total number of processes being run (for training only, set to 1 when preproc only)
        """
        os.makedirs(out_dir, exist_ok=True)

        tar_file_ctr = 1
        num_files_in_tar = 0
        num_lines = 0
        shard_num = 0
        global_batch_ctr = 0
        tmp_f = tempfile.NamedTemporaryFile(delete=False, mode='w')
        tar_file_ptr = tarfile.open(
            os.path.join(out_dir, '%s-batches.tokens.%d.%d.tar' % (pkl_file_prefix, tokens_in_batch, 1)), 'w'
        )
        metadata_path = os.path.join(out_dir, f'metadata.tokens.{tokens_in_batch}.json')
        with open(fname, 'r') as f:
            for line in f:
                tmp_f.write(line)
                num_lines += 1

                if num_lines == lines_per_dataset_fragment:
                    tmp_f.close()
                    (
                        tar_file_ptr,
                        global_batch_ctr,
                        num_files_in_tar,
                        tar_file_ctr,
                    ) = MTDataPreproc.write_monolingual_batches_to_tarfiles(
                        out_dir,
                        num_batches_per_tarfile,
                        clean,
                        max_seq_length,
                        min_seq_length,
                        tmp_f.name,
                        tokens_in_batch,
                        tokenizer,
                        num_files_in_tar=num_files_in_tar,
                        tar_file_ptr=tar_file_ptr,
                        tar_file_ctr=tar_file_ctr,
                        global_batch_ctr=global_batch_ctr,
                        pkl_file_prefix=pkl_file_prefix,
                    )

                    num_lines = 0
                    shard_num += 1
                    os.remove(tmp_f.name)
                    tmp_f = tempfile.NamedTemporaryFile(delete=False, mode='w')

        tmp_f.close()
        (
            tar_file_ptr,
            global_batch_ctr,
            num_files_in_tar,
            tar_file_ctr,
        ) = MTDataPreproc.write_monolingual_batches_to_tarfiles(
            out_dir,
            num_batches_per_tarfile,
            clean,
            max_seq_length,
            min_seq_length,
            tmp_f.name,
            tokens_in_batch,
            tokenizer,
            num_files_in_tar=num_files_in_tar,
            tar_file_ptr=tar_file_ptr,
            tar_file_ctr=tar_file_ctr,
            global_batch_ctr=global_batch_ctr,
            pkl_file_prefix=pkl_file_prefix,
        )
        tar_file_ptr.close()
        os.remove(tmp_f.name)

        if num_files_in_tar != num_batches_per_tarfile:
            os.remove(
                os.path.join(out_dir, '%s-batches.tokens.%d.%d.tar' % (pkl_file_prefix, tokens_in_batch, tar_file_ctr))
            )
            global_batch_ctr -= num_files_in_tar
            print('Dropping %d batches because of overflow' % (num_files_in_tar))

        json.dump({'num_batches': global_batch_ctr}, open(os.path.join(out_dir, 'metadata.json'), 'w'))

        tar_file_paths = glob.glob(f'{out_dir}/{pkl_file_prefix}-batches.tokens.{tokens_in_batch}.*.tar')

        num_tar_files = len(tar_file_paths)
        if num_tar_files < world_size:
            raise ValueError(
                (
                    f'Number of tar files found: {num_tar_files} is less than world size: {world_size}. '
                    f'There should be at least one tar file per GPU (ideally many tar files per GPU). '
                    f'This may be due to dataset size, it is advisable to use at least 5M sentence pairs for tarred datasets. '
                    f'Decrease num_batches_per_tarfile or num_tokens_per_batch to increase the number of tarfiles. '
                    f'Also using shard_strategy=replicate will use all available tarfiles for every GPU. '
                )
            )

        return tar_file_paths, metadata_path

    @staticmethod
    def train_yttm_tokenizers(
        out_dir,
        src_fname,
        tgt_fname,
        shared_tokenizer,
        encoder_tokenizer_name,
        encoder_tokenizer_vocab_size,
        encoder_tokenizer_coverage,
        decoder_tokenizer_name,
        decoder_tokenizer_vocab_size,
        decoder_tokenizer_coverage,
        global_rank,
    ):
        encoder_tokenizer_model = None
        decoder_tokenizer_model = None
        os.makedirs(out_dir, exist_ok=True)

        if shared_tokenizer:
            if encoder_tokenizer_name != 'yttm' or decoder_tokenizer_name != 'yttm':
                raise NotImplementedError(f"Currently we only support yttm for shared tokenizer.")

            encoder_tokenizer_model = os.path.join(
                out_dir, 'shared_tokenizer.%d.BPE.model' % (encoder_tokenizer_vocab_size)
            )
            decoder_tokenizer_model = encoder_tokenizer_model
            if global_rank == 0:
                if os.path.isfile(encoder_tokenizer_model):
                    logging.info(
                        f'Shared tokenizer model {encoder_tokenizer_model} already exists. Remove file if training a new tokenizer model.'
                    )
                else:
                    logging.info(
                        f'Shared tokenizer model {encoder_tokenizer_model} not found. Training tokenizer model.'
                    )
                    with tempfile.TemporaryDirectory() as tmp:
                        concat_data_path = os.path.join(tmp, 'concat_dataset.txt')
                        os.system('cat %s %s > %s' % (src_fname, tgt_fname, concat_data_path))
                        yttm.BPE.train(
                            data=concat_data_path,
                            vocab_size=encoder_tokenizer_vocab_size,
                            model=os.path.join(out_dir, encoder_tokenizer_model),
                            coverage=encoder_tokenizer_coverage,
                            n_threads=-1,
                        )
        else:
            if encoder_tokenizer_name == 'yttm':
                encoder_tokenizer_model = os.path.join(
                    out_dir, 'tokenizer.encoder.%d.BPE.model' % (encoder_tokenizer_vocab_size)
                )
                if global_rank == 0:
                    if os.path.isfile(encoder_tokenizer_model):
                        logging.info(
                            f'Encoder tokenizer model {encoder_tokenizer_model} already exists. Remove file if training a new tokenizer model.'
                        )
                    else:
                        logging.info(
                            f'Encoder tokenizer model {encoder_tokenizer_model} not found. Training tokenizer model.'
                        )
                        yttm.BPE.train(
                            data=src_fname,
                            vocab_size=encoder_tokenizer_vocab_size,
                            model=encoder_tokenizer_model,
                            coverage=encoder_tokenizer_coverage,
                            n_threads=-1,
                        )

            if decoder_tokenizer_name == 'yttm':
                decoder_tokenizer_model = os.path.join(
                    out_dir, 'tokenizer.decoder.%d.BPE.model' % (decoder_tokenizer_vocab_size)
                )
                if global_rank == 0:
                    if os.path.isfile(decoder_tokenizer_model):
                        logging.info(
                            f'Decoder tokenizer model {decoder_tokenizer_model} already exists. Remove file if training a new tokenizer model.'
                        )
                    else:
                        logging.info(
                            f'Decoder tokenizer model {decoder_tokenizer_model} not found. Training tokenizer model.'
                        )
                        yttm.BPE.train(
                            data=tgt_fname,
                            vocab_size=decoder_tokenizer_vocab_size,
                            model=decoder_tokenizer_model,
                            coverage=decoder_tokenizer_coverage,
                            n_threads=-1,
                        )

        return encoder_tokenizer_model, decoder_tokenizer_model

    @staticmethod
    def write_parallel_batches_to_tarfiles(
        out_dir,
        num_batches_per_tarfile,
        clean,
        max_seq_length,
        min_seq_length,
        src_fname,
        tgt_fname,
        num_tokens,
        encoder_tokenizer,
        decoder_tokenizer,
        num_files_in_tar,
        tar_file_ptr,
        tar_file_ctr,
        global_batch_ctr,
        pkl_file_prefix,
    ):
        """
        Writes current fragment of the overall parallel corpus to tarfiles by:
        (1) Creating a minibatches using a TranslationDataset object.
        (2) Writing each minibatch to a pickle file.
        (3) Adding pickle files to a tarfile until it reaches num_batches_per_tarfile.
        """

        dataset = TranslationDataset(
            dataset_src=src_fname,
            dataset_tgt=tgt_fname,
            tokens_in_batch=num_tokens,
            clean=clean,
            max_seq_length=max_seq_length,
            min_seq_length=min_seq_length,
            max_seq_length_diff=max_seq_length,
            max_seq_length_ratio=max_seq_length,
            cache_ids=False,
            cache_data_per_node=False,
            use_cache=False,
        )
        dataset.batchify(encoder_tokenizer, decoder_tokenizer)

        for _, batch in dataset.batches.items():
            global_batch_ctr += 1
            pickle.dump(
                batch, open(os.path.join(out_dir, '%s-batch-%d.pkl' % (pkl_file_prefix, global_batch_ctr)), 'wb')
            )

            if num_files_in_tar == num_batches_per_tarfile:
                tar_file_ctr += 1
                tar_file_ptr.close()
                tar_file_ptr = tarfile.open(
                    os.path.join(out_dir, '%s-batches.tokens.%d.%d.tar' % (pkl_file_prefix, num_tokens, tar_file_ctr)),
                    'w',
                )
                num_files_in_tar = 0

            tar_file_ptr.add(os.path.join(out_dir, '%s-batch-%d.pkl' % (pkl_file_prefix, global_batch_ctr)))
            num_files_in_tar += 1
            os.remove(os.path.join(out_dir, '%s-batch-%d.pkl' % (pkl_file_prefix, global_batch_ctr)))
        return tar_file_ptr, global_batch_ctr, num_files_in_tar, tar_file_ctr

    @staticmethod
    def write_monolingual_batches_to_tarfiles(
        out_dir,
        num_batches_per_tarfile,
        clean,
        max_seq_length,
        min_seq_length,
        fname,
        num_tokens,
        tokenizer,
        num_files_in_tar,
        tar_file_ptr,
        tar_file_ctr,
        global_batch_ctr,
        pkl_file_prefix,
    ):
        """
        Writes current fragment of the overall parallel corpus to tarfiles by:
        (1) Creating a minibatches using a TranslationOneSideDataset object.
        (2) Writing each minibatch to a pickle file.
        (3) Adding pickle files to a tarfile until it reaches num_batches_per_tarfile.
        """

        dataset = TranslationOneSideDataset(
            tokenizer=tokenizer,
            dataset=fname,
            tokens_in_batch=num_tokens,
            clean=clean,
            max_seq_length=max_seq_length,
            min_seq_length=min_seq_length,
            cache_ids=False,
        )

        for batch in dataset.batches:
            global_batch_ctr += 1
            batch = {'src': batch}
            pickle.dump(
                batch, open(os.path.join(out_dir, '%s-batch-%d.pkl' % (pkl_file_prefix, global_batch_ctr)), 'wb')
            )

            if num_files_in_tar == num_batches_per_tarfile:
                tar_file_ctr += 1
                tar_file_ptr.close()
                tar_file_ptr = tarfile.open(
                    os.path.join(out_dir, '%s-batches.tokens.%d.%d.tar' % (pkl_file_prefix, num_tokens, tar_file_ctr)),
                    'w',
                )
                num_files_in_tar = 0

            tar_file_ptr.add(os.path.join(out_dir, '%s-batch-%d.pkl' % (pkl_file_prefix, global_batch_ctr)))
            num_files_in_tar += 1
            os.remove(os.path.join(out_dir, '%s-batch-%d.pkl' % (pkl_file_prefix, global_batch_ctr)))
        return tar_file_ptr, global_batch_ctr, num_files_in_tar, tar_file_ctr

    @property
    def cfg(self):
        return self._cfg
