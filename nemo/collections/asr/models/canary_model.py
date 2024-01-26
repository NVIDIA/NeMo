# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
import itertools
import json
import os
import re
import tempfile
from typing import Dict, List, Optional, Union

import editdistance
import torch
import torch.distributed as dist
from nltk.translate.bleu_score import corpus_bleu
from omegaconf import DictConfig
from tqdm import tqdm

from nemo.collections.asr.data.audio_to_text_canary import CanaryDataset
from nemo.collections.asr.data.audio_to_text_dali import DALIOutputs
from nemo.collections.asr.models.transformer_bpe_models import EncDecTransfModelBPE
from nemo.collections.asr.parts.utils import manifest_utils
from nemo.collections.asr.parts.utils.audio_utils import ChannelSelectorType
from nemo.collections.common.data.lhotse import get_lhotse_dataloader_from_config
from nemo.collections.common.parts.preprocessing.manifest import get_full_path
from nemo.utils import logging

try:
    from nemo.collections.nlp.modules.common.transformer import BeamSearchSequenceGenerator

    NLP_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    NLP_AVAILABLE = False
    logging.warning("Could not import NeMo NLP collection which is required for speech translation model.")


class CanaryModel(EncDecTransfModelBPE):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.context_len_for_AR_decoding = 5

    def change_decoding_strategy(self, cfg: DictConfig):
        logging.info(f"Changing beam search decoding to {cfg}")
        # Beam Search decoding
        self.beam_search = BeamSearchSequenceGenerator(
            embedding=self.transf_decoder.embedding,
            decoder=self.transf_decoder.decoder,
            log_softmax=self.log_softmax,
            max_sequence_length=self.transf_decoder.max_sequence_length,
            beam_size=cfg.beam_size,
            bos=self.tokenizer.bos_id,
            pad=self.tokenizer.pad_id,
            eos=self.tokenizer.eos_id,
            len_pen=cfg.len_pen,
            max_delta_length=cfg.max_generation_delta,
        )

    @torch.no_grad()
    def transcribe(
        self,
        paths2audio_files: Union[List[str], str],
        batch_size: int = 4,
        logprobs: bool = False,
        return_hypotheses: bool = False,
        num_workers: int = 0,
        channel_selector: Optional[ChannelSelectorType] = None,
        augmentor: DictConfig = None,
        verbose: bool = True,
    ) -> List[str]:
        """
        Uses greedy decoding to transcribe audio files. Use this method for debugging and prototyping.
        Args:
            paths2audio_files: (a list) of paths to audio files. \
                Recommended length per file is between 5 and 25 seconds. \
                But it is possible to pass a few hours long file if enough GPU memory is available.
            batch_size: (int) batch size to use during inference.
                Bigger will result in better throughput performance but would use more memory.
            logprobs: (bool) pass True to get log probabilities instead of transcripts.
            return_hypotheses: (bool) Either return hypotheses or text
                With hypotheses can do some postprocessing like getting timestamp or rescoring
            num_workers: (int) number of workers for DataLoader
            channel_selector (int | Iterable[int] | str): select a single channel or a subset of channels from multi-channel audio. If set to `'average'`, it performs averaging across channels. Disabled if set to `None`. Defaults to `None`.
            augmentor: (DictConfig): Augment audio samples during transcription if augmentor is applied.
            verbose: (bool) whether to display tqdm progress bar
        Returns:
            A list of transcriptions (or raw log probabilities if logprobs is True) in the same order as paths2audio_files
        """
        if paths2audio_files is None or len(paths2audio_files) == 0:
            return {}

        if return_hypotheses:
            logging.warning("return_hypotheses=True is currently not supported, returning text instead.")

        if isinstance(paths2audio_files, list):
            logging.info(f"Found paths2audio_files to be a list of {len(paths2audio_files)} items.")
            logging.info(f"Assuming each item in paths2audio_files is a path to audio file.")
            logging.info(f"Transcribing with default Canary setting of English without PnC.")
        elif isinstance(paths2audio_files, str):
            logging.info(f"Found paths2audio_files to be a string. Assuming it is a path to manifest file.")
            assert os.path.exists(paths2audio_files), f"File {paths2audio_files} doesn't exist"
            assert paths2audio_files.endswith('.json') or paths2audio_files.endswith(
                '.jsonl'
            ), f"File {paths2audio_files} must be a json or jsonl file"

            # load json lines
            manifest_path = paths2audio_files  # need to save this as we are overwriting paths2audio_files in nextline
            paths2audio_files = manifest_utils.read_manifest(paths2audio_files)

        def _may_be_make_dict_and_fix_paths(json_items):
            out_json_items = []
            for item in json_items:
                if isinstance(item, str):
                    # assume it is a path to audio file
                    entry = {
                        'audio_filepath': item,
                        'duration': 100000,
                        'source_lang': 'en',
                        'taskname': 'asr',
                        'target_lang': 'en',
                        'pnc': 'no',
                        'answer': 'nothing',
                    }
                elif isinstance(item, dict):
                    entry = item
                    entry['audio_filepath'] = get_full_path(entry['audio_filepath'], manifest_file=manifest_path)
                else:
                    raise ValueError(f"Expected str or dict, got {type(item)}")
                out_json_items.append(entry)
            return out_json_items

        paths2audio_files = _may_be_make_dict_and_fix_paths(paths2audio_files)

        if return_hypotheses and logprobs:
            raise ValueError(
                "Either `return_hypotheses` or `logprobs` can be True at any given time."
                "Returned hypotheses will contain the logprobs."
            )

        if num_workers is None:
            num_workers = min(batch_size, os.cpu_count() - 1)

        # We will store transcriptions here
        hypotheses = []

        # Model's mode and device
        mode = self.training
        device = next(self.parameters()).device
        dither_value = self.preprocessor.featurizer.dither
        pad_to_value = self.preprocessor.featurizer.pad_to

        try:
            self.preprocessor.featurizer.dither = 0.0
            self.preprocessor.featurizer.pad_to = 0
            # Switch model to evaluation mode
            self.eval()
            # Freeze the encoder and decoder modules
            self.encoder.freeze()
            self.transf_decoder.freeze()
            logging_level = logging.get_verbosity()
            logging.set_verbosity(logging.WARNING)
            # Work in tmp directory - will store manifest file there
            with tempfile.TemporaryDirectory() as tmpdir:
                with open(os.path.join(tmpdir, 'manifest.json'), 'w') as fp:
                    for audio_file in paths2audio_files:
                        # _may_be_make_dict_and_fix_paths has already fixed the path and added other fields   if needed
                        fp.write(json.dumps(audio_file) + '\n')

                config = {
                    'paths2audio_files': paths2audio_files,
                    'batch_size': batch_size,
                    'temp_dir': tmpdir,
                    'num_workers': num_workers,
                    'channel_selector': channel_selector,
                }

                if augmentor:
                    config['augmentor'] = augmentor

                temporary_datalayer = self._setup_transcribe_dataloader(config)
                for test_batch in tqdm(temporary_datalayer, desc="Transcribing", disable=not verbose):
                    log_probs, encoded_len, enc_states, enc_mask = self.forward(
                        input_signal=test_batch[0].to(device), input_signal_length=test_batch[1].to(device)
                    )

                    beam_hypotheses = (
                        self.beam_search(
                            encoder_hidden_states=enc_states,
                            encoder_input_mask=enc_mask,
                            return_beam_scores=False,
                            decoder_input_ids=test_batch[2][:, : self.context_len_for_AR_decoding].to(device)
                            if self.context_len_for_AR_decoding > 0
                            else None,
                        )
                        .detach()
                        .cpu()
                        .numpy()
                    )

                    beam_hypotheses = [
                        self._strip_special_tokens(self.tokenizer.ids_to_text(hyp)) for hyp in beam_hypotheses
                    ]

                    # TODO: add support for return_hypotheses=True @AlexGrinch
                    # if return_hypotheses:
                    #     # dump log probs per file
                    #     for idx in range(logits.shape[0]):
                    #         current_hypotheses[idx].y_sequence = logits[idx][: logits_len[idx]]

                    hypotheses += beam_hypotheses

                    del test_batch, log_probs, encoded_len, enc_states, enc_mask
        finally:
            # set mode back to its original value
            self.train(mode=mode)
            self.preprocessor.featurizer.dither = dither_value
            self.preprocessor.featurizer.pad_to = pad_to_value
            if mode is True:
                self.encoder.unfreeze()
                self.transf_decoder.unfreeze()
            logging.set_verbosity(logging_level)

        return hypotheses

    def validation_step(self, batch, batch_idx, dataloader_idx=0, eval_mode="val"):
        signal, signal_len, transcript, transcript_len = batch
        input_ids, labels = transcript[:, :-1], transcript[:, 1:]

        if isinstance(batch, DALIOutputs) and batch.has_processed_signal:
            transf_log_probs, encoded_len, enc_states, enc_mask = self.forward(
                processed_signal=signal,
                processed_signal_length=signal_len,
                transcript=input_ids,
                transcript_length=transcript_len,
            )
        else:
            transf_log_probs, encoded_len, enc_states, enc_mask = self.forward(
                input_signal=signal,
                input_signal_length=signal_len,
                transcript=input_ids,
                transcript_length=transcript_len,
            )

        beam_hypotheses = self.beam_search(
            encoder_hidden_states=enc_states,
            encoder_input_mask=enc_mask,
            return_beam_scores=False,
            decoder_input_ids=input_ids[:, : self.context_len_for_AR_decoding]
            if self.context_len_for_AR_decoding > 0
            else None,
        )
        transf_loss = self.transf_loss(log_probs=transf_log_probs, labels=labels)

        ground_truths = [self.tokenizer.ids_to_text(sent) for sent in transcript.detach().cpu().tolist()]
        translations = [self.tokenizer.ids_to_text(sent) for sent in beam_hypotheses.detach().cpu().tolist()]

        self.val_loss(loss=transf_loss, num_measurements=transf_log_probs.shape[0] * transf_log_probs.shape[1])

        output_dict = {
            f'{eval_mode}_loss': transf_loss,
            'translations': [self._strip_special_tokens(t) for t in translations],
            'ground_truths': [self._strip_special_tokens(g) for g in ground_truths],
        }

        if type(self.trainer.val_dataloaders) == list and len(self.trainer.val_dataloaders) > 1:
            self.validation_step_outputs[dataloader_idx].append(output_dict)
        else:
            self.validation_step_outputs.append(output_dict)

        return output_dict

    def multi_validation_epoch_end(self, outputs, dataloader_idx: int = 0, eval_mode: str = "val"):
        """
        Called at the end of validation to aggregate outputs.
        :param outputs: list of individual outputs of each validation step.
        """
        if not outputs:
            return

        if isinstance(outputs[0], dict):
            outputs = [outputs]

        for output in outputs:
            eval_loss = getattr(self, 'val_loss').compute()
            translations = list(itertools.chain(*[x['translations'] for x in output]))
            ground_truths = list(itertools.chain(*[x['ground_truths'] for x in output]))

            # Gather translations and ground truths from all workers
            tr_and_gt = [None for _ in range(self.world_size)]
            # we also need to drop pairs where ground truth is an empty string
            if self.world_size > 1:
                dist.all_gather_object(
                    tr_and_gt, [(t, g) for (t, g) in zip(translations, ground_truths) if g.strip() != '']
                )
            else:
                tr_and_gt[0] = [(t, g) for (t, g) in zip(translations, ground_truths) if g.strip() != '']

            if self.global_rank == 0:
                _translations = []
                _ground_truths = []
                for rank in range(0, self.world_size):
                    _translations += [t for (t, g) in tr_and_gt[rank]]
                    _ground_truths += [g for (t, g) in tr_and_gt[rank]]

                sacre_bleu = corpus_bleu(_translations, [_ground_truths], tokenize="13a")
                sb_score = sacre_bleu.score * self.world_size

                wer_scores, wer_words = 0, 0
                for h, r in zip(_translations, _ground_truths):
                    wer_words += len(r.split())
                    wer_scores += editdistance.eval(h.split(), r.split())
                wer_score = 1.0 * wer_scores * self.world_size / wer_words

            else:
                sb_score = 0.0
                wer_score = 0.0

            # To log via on_validation_epoch_end in modelPT.py
            # remove  (* self.world_size) if logging via on_validation_epoch_end
            # tensorboard_logs = {}
            # tensorboard_logs.update({f"{eval_mode}_loss": eval_loss})
            # tensorboard_logs.update({f"{eval_mode}_sacreBLEU": sb_score})
            # tensorboard_logs.update({f"{eval_mode}_WER": wer_score})

            # logging here only.
            dataloader_prefix = self.get_validation_dataloader_prefix(dataloader_idx)
            self.log(f"{dataloader_prefix}{eval_mode}_loss", eval_loss, sync_dist=True)
            self.log(f"{dataloader_prefix}{eval_mode}_sacreBLEU", sb_score, sync_dist=True)
            self.log(f"{dataloader_prefix}{eval_mode}_WER", wer_score, sync_dist=True)

            # in multi-validation case, anything after first one will become NaN
            # as we are resetting the metric here.
            # TODO: fix this, (not sure which hook will be ideal for this)
            self.val_loss.reset()

    def _setup_dataloader_from_config(self, config: Optional[Dict]):
        assert config.get("use_lhotse", False), (
            "Canary model requires lhotse dataloading to be enabled "
            "(set model.{train,validation,test}_ds.use_lhotse=True)."
        )

        return get_lhotse_dataloader_from_config(
            config,
            global_rank=self.global_rank,
            world_size=self.world_size,
            dataset=CanaryDataset(tokenizer=self.tokenizer),
        )

    def _setup_transcribe_dataloader(self, config: Dict) -> 'torch.utils.data.DataLoader':
        """
        Setup function for a temporary data loader which wraps the provided audio file.
        Args:
            config: A python dictionary which contains the following keys:
            paths2audio_files: (a list) of paths to audio files. The files should be relatively short fragments. \
                Recommended length per file is between 5 and 25 seconds.
            batch_size: (int) batch size to use during inference. \
                Bigger will result in better throughput performance but would use more memory.
            temp_dir: (str) A temporary directory where the audio manifest is temporarily
                stored.
        Returns:
            A pytorch DataLoader for the given audio file(s).
        """
        batch_size = min(config['batch_size'], len(config['paths2audio_files']))
        dl_config = {
            'manifest_filepath': os.path.join(config['temp_dir'], 'manifest.json'),
            'sample_rate': self.preprocessor._sample_rate,
            'batch_size': batch_size,
            'trim_silence': False,
            'shuffle': False,
            'num_workers': min(batch_size, os.cpu_count() - 1),
            'pin_memory': True,
        }

        # TODO: remove this lhotse hardcoding later (@kpuvvada)
        # currently only works for non-tarred
        lhotse_config = {
            'is_tarred': False,
            'batch_size': 1,
            'use_lhotse': True,
            'lhotse': {
                'use_bucketing': False,
                'max_cuts': batch_size,
                'drop_last': False,
                'text_field': 'answer',
                'lang_field': 'target_lang',
            },
        }

        # update dl_config
        dl_config.update(lhotse_config)

        temporary_datalayer = self._setup_dataloader_from_config(config=DictConfig(dl_config))
        return temporary_datalayer

    def _strip_special_tokens(self, text):
        """
        assuming all special tokens are of format <token>
        Note that if any label/pred is of format <token>, it will be stripped
        """
        assert isinstance(text, str), f"Expected str, got {type(text)}"
        text = re.sub(r'<[^>]+>', '', text)
        # strip spaces at the beginning and end;
        # this is training data artifact, will be fixed in future (@kpuvvada)
        return text.strip()
