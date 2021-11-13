# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

import json
import os
import time
from argparse import ArgumentParser
from typing import List, Tuple

from joblib import Parallel, delayed
from nemo_text_processing.text_normalization.data_loader_utils import post_process_punctuation, pre_process
from nemo_text_processing.text_normalization.normalize import Normalizer
from tqdm import tqdm

try:
    from nemo.collections.asr.metrics.wer import word_error_rate
    from nemo.collections.asr.models import ASRModel

    ASR_AVAILABLE = True
except (ModuleNotFoundError, ImportError):
    ASR_AVAILABLE = False


try:
    import pynini
    from pynini.lib import rewrite

    PYNINI_AVAILABLE = True
except (ModuleNotFoundError, ImportError):
    PYNINI_AVAILABLE = False

"""
The script provides multiple normalization options and chooses the best one that minimizes CER of the ASR output
(most of the semiotic classes use deterministic=False flag).

To run this script with a .json manifest file, the manifest file should contain the following fields:
    "audio_filepath" - path to the audio file
    "text" - raw text
    "pred_text" - ASR model prediction
    
    See https://github.com/NVIDIA/NeMo/blob/main/examples/asr/transcribe_speech.py on how to add ASR predictions
        
    When the manifest is ready, run:
        python normalize_with_audio.py \
               --audio_data PATH/TO/MANIFEST.JSON \
               --language en 
     
        
To run with a single audio file, specify path to audio and text with:
    python normalize_with_audio.py \
           --audio_data PATH/TO/AUDIO.WAV \
           --language en \
           --text raw text OR PATH/TO/.TXT/FILE
           --model QuartzNet15x5Base-En \
           --verbose
    
To see possible normalization options for a text input without an audio file (could be used for debugging), run:
    python python normalize_with_audio.py --text "RAW TEXT"
    
Specify `--cache_dir` to generate .far grammars once and re-used them for faster inference
"""


class NormalizerWithAudio(Normalizer):
    """
    Normalizer class that converts text from written to spoken form. 
    Useful for TTS preprocessing. 

    Args:
        input_case: expected input capitalization
        lang: language
        cache_dir: path to a dir with .far grammar file. Set to None to avoid using cache.
        overwrite_cache: set to True to overwrite .far files
        whitelist: path to a file with whitelist replacements
    """

    def __init__(
        self,
        input_case: str,
        lang: str = 'en',
        cache_dir: str = None,
        overwrite_cache: bool = False,
        whitelist: str = None,
    ):

        super().__init__(
            input_case=input_case,
            lang=lang,
            deterministic=False,
            cache_dir=cache_dir,
            overwrite_cache=overwrite_cache,
            whitelist=whitelist,
        )

    def normalize(
        self,
        text: str,
        n_tagged: int,
        punct_pre_process: bool = True,
        punct_post_process: bool = True,
        verbose: bool = False,
    ) -> str:
        """
        Main function. Normalizes tokens from written to spoken form
            e.g. 12 kg -> twelve kilograms

        Args:
            text: string that may include semiotic classes
            n_tagged: number of tagged options to consider, -1 - to get all possible tagged options
            punct_pre_process: whether to perform punctuation pre-processing, for example, [25] -> [ 25 ]
            punct_post_process: whether to normalize punctuation
            verbose: whether to print intermediate meta information

        Returns:
            normalized text options (usually there are multiple ways of normalizing a given semiotic class)
        """
        if punct_pre_process:
            text = pre_process(text)
        text = text.strip()
        if not text:
            if verbose:
                print(text)
            return text

        text = pynini.escape(text)

        if n_tagged == -1:
            tagged_texts = rewrite.rewrites(text, self.tagger.fst)
        else:
            tagged_texts = rewrite.top_rewrites(text, self.tagger.fst, nshortest=n_tagged)
        # non-deterministic Eng normalization uses tagger composed with verbalizer, no permutation in between
        if self.lang == 'en':
            normalized_texts = tagged_texts
        else:
            normalized_texts = []
            for tagged_text in tagged_texts:
                self._verbalize(tagged_text, normalized_texts)

        if len(normalized_texts) == 0:
            raise ValueError()
        if punct_post_process:
            normalized_texts = [post_process_punctuation(t) for t in normalized_texts]

            # do post-processing based on Moses detokenizer
            if self.processor:
                normalized_texts = [self.processor.detokenize([t]) for t in normalized_texts]
        normalized_texts = set(normalized_texts)
        return normalized_texts

    def _verbalize(self, tagged_text: str, normalized_texts: List[str]):
        """
        Verbalizes tagged text

        Args:
            tagged_text: text with tags
            normalized_texts: list of possible normalization options
        """

        def get_verbalized_text(tagged_text):
            tagged_text = pynini.escape(tagged_text)
            return rewrite.rewrites(tagged_text, self.verbalizer.fst)

        self.parser(tagged_text)
        tokens = self.parser.parse()
        tags_reordered = self.generate_permutations(tokens)
        for tagged_text_reordered in tags_reordered:
            try:
                normalized_texts.extend(get_verbalized_text(tagged_text_reordered))
            except pynini.lib.rewrite.Error:
                continue

    def select_best_match(
        self, normalized_texts: List[str], pred_text: str, verbose: bool = False, remove_punct: bool = False
    ):
        """
        Selects the best normalization option based on the lowest CER

        Args:
            normalized_texts: normalized text options
            pred_text: ASR model transcript of the audio file corresponding to the normalized text
            verbose: whether to print intermediate meta information
            remove_punct: whether to remove punctuation before calculating CER

        Returns:
            normalized text with the lowest CER and CER value
        """
        normalized_texts = calculate_cer(normalized_texts, pred_text, remove_punct)
        normalized_texts = sorted(normalized_texts, key=lambda x: x[1])
        normalized_text, cer = normalized_texts[0]

        if verbose:
            print('-' * 30)
            for option in normalized_texts:
                print(option)
            print('-' * 30)
        return normalized_text, cer


def calculate_cer(normalized_texts: List[str], pred_text: str, remove_punct=False) -> List[Tuple[str, float]]:
    """
    Calculates character error rate (CER)

    Args:
        normalized_texts: normalized text options
        pred_text: ASR model output

    Returns: normalized options with corresponding CER
    """
    normalized_options = []
    for text in normalized_texts:
        text_clean = text.replace('-', ' ').lower()
        if remove_punct:
            for punct in "!?:;,.-()*+-/<=>@^_":
                text_clean = text_clean.replace(punct, "")
        cer = round(word_error_rate([pred_text], [text_clean], use_cer=True) * 100, 2)
        normalized_options.append((text, cer))
    return normalized_options


def get_asr_model(asr_model):
    """
    Returns ASR Model

    Args:
        asr_model: NeMo ASR model
    """
    if os.path.exists(args.model):
        asr_model = ASRModel.restore_from(asr_model)
    elif args.model in ASRModel.get_available_model_names():
        asr_model = ASRModel.from_pretrained(asr_model)
    else:
        raise ValueError(
            f'Provide path to the pretrained checkpoint or choose from {ASRModel.get_available_model_names()}'
        )
    return asr_model


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--text", help="input string or path to a .txt file", default=None, type=str)
    parser.add_argument(
        "--input_case", help="input capitalization", choices=["lower_cased", "cased"], default="cased", type=str
    )
    parser.add_argument("--language", help="Select target language", choices=["en", "ru"], default="en", type=str)
    parser.add_argument("--audio_data", default=None, help="path to an audio file or .json manifest")
    parser.add_argument(
        '--model', type=str, default='QuartzNet15x5Base-En', help='Pre-trained model name or path to model checkpoint'
    )
    parser.add_argument(
        "--n_tagged",
        type=int,
        default=10000,
        help="number of tagged options to consider, -1 - return all possible tagged options",
    )
    parser.add_argument("--verbose", help="print info for debugging", action="store_true")
    parser.add_argument("--remove_punct", help="remove punctuation before calculating cer", action="store_true")
    parser.add_argument(
        "--no_punct_pre_process", help="set to True to disable punctuation pre processing", action="store_true"
    )
    parser.add_argument(
        "--no_punct_post_process", help="set to True to disable punctuation post processing", action="store_true"
    )
    parser.add_argument("--overwrite_cache", help="set to True to re-create .far grammar files", action="store_true")
    parser.add_argument("--whitelist", help="path to a file with with whitelist", default=None, type=str)
    parser.add_argument(
        "--cache_dir",
        help="path to a dir with .far grammar file. Set to None to avoid using cache",
        default=None,
        type=str,
    )
    parser.add_argument("--n_jobs", default=-2, type=int, help="The maximum number of concurrently running jobs")
    return parser.parse_args()


def _normalize_line(normalizer: NormalizerWithAudio, line: str):
    line = json.loads(line)
    pred_text = line['pred_text']

    normalized_texts = normalizer.normalize(
        text=line['text'],
        verbose=args.verbose,
        n_tagged=args.n_tagged,
        punct_pre_process=not args.no_punct_pre_process,
        punct_post_process=not args.no_punct_post_process,
    )
    normalized_text, cer = normalizer.select_best_match(normalized_texts, pred_text, args.verbose, args.remove_punct)
    line['nemo_normalized'] = normalized_text
    line['CER_nemo_normalized'] = cer
    return line


def normalize_manifest(args):
    """
    Args:
        args.audio_data: path to .json manifest file.
    """
    normalizer = NormalizerWithAudio(
        input_case=args.input_case,
        lang=args.language,
        cache_dir=args.cache_dir,
        overwrite_cache=args.overwrite_cache,
        whitelist=args.whitelist,
    )
    manifest_out = args.audio_data.replace('.json', '_normalized.json')
    with open(args.audio_data, 'r') as f:
        lines = f.readlines()

        print(f'Normalizing {len(lines)}lines of {args.audio_data}...')
        with open(manifest_out, 'w') as f_out:
            # to save intermediate results to a file
            batch = max(round(len(lines) / 10), 1000)
            for i in range(0, len(lines), batch):
                print(f'Processing batch {i} out of {round(len(lines)/batch)}.')
                normalized_lines = Parallel(n_jobs=args.n_jobs)(
                    delayed(_normalize_line)(normalizer, line) for line in tqdm(lines[i : i + batch])
                )
                for line in normalized_lines:
                    f_out.write(json.dumps(line, ensure_ascii=False) + '\n')
    print(f'Normalized version saved at {manifest_out}')


if __name__ == "__main__":
    args = parse_args()

    if not ASR_AVAILABLE and args.audio_data:
        raise ValueError("NeMo ASR collection is not installed.")
    start = time.time()
    args.whitelist = os.path.abspath(args.whitelist) if args.whitelist else None
    if args.text:
        normalizer = NormalizerWithAudio(
            input_case=args.input_case,
            lang=args.language,
            cache_dir=args.cache_dir,
            overwrite_cache=args.overwrite_cache,
            whitelist=args.whitelist,
        )

        if os.path.exists(args.text):
            with open(args.text, 'r') as f:
                args.text = f.read().strip()
        normalized_texts = normalizer.normalize(
            text=args.text,
            verbose=args.verbose,
            n_tagged=args.n_tagged,
            punct_pre_process=not args.no_punct_pre_process,
            punct_post_process=not args.no_punct_post_process,
        )
        if args.audio_data:
            asr_model = get_asr_model(args.model)
            pred_text = asr_model.transcribe([args.audio_data])[0]
            normalized_text, cer = normalizer.select_best_match(
                normalized_texts, pred_text, args.verbose, args.remove_punct
            )
            print(f'Transcript: {pred_text}')
            print(f'Normalized: {normalized_text}')
        else:
            print('Normalization options:')
            for norm_text in normalized_texts:
                print(norm_text)
    elif not os.path.exists(args.audio_data):
        raise ValueError(f'{args.audio_data} not found.')
    elif args.audio_data.endswith('.json'):
        normalize_manifest(args)
    else:
        raise ValueError(
            "Provide either path to .json manifest in '--audio_data' OR "
            + "'--audio_data' path to audio file and '--text' path to a text file OR"
            "'--text' string text (for debugging without audio)"
        )
    print(f'Execution time: {round((time.time() - start)/60, 2)} min.')
