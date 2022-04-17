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

import argparse
import os
import re
from pathlib import Path
from typing import List

import regex
from joblib import Parallel, delayed
from normalization_helpers import LATIN_TO_RU, RU_ABBREVIATIONS
from num2words import num2words
from pydub import AudioSegment
from pydub.utils import mediainfo
from tqdm import tqdm

from nemo.collections.asr.models import ASRModel
from nemo.utils import model_utils

try:
    from nemo_text_processing.text_normalization.normalize import Normalizer

    NEMO_NORMALIZATION_AVAILABLE = True
except (ModuleNotFoundError, ImportError):
    NEMO_NORMALIZATION_AVAILABLE = False


parser = argparse.ArgumentParser(description="Prepares text and audio files for segmentation")
parser.add_argument("--in_text", type=str, default=None, help="Path to a text file or a directory with .txt files")
parser.add_argument("--output_dir", type=str, required=True, help="Path to output directory")
parser.add_argument("--audio_dir", type=str, help="Path to folder with .mp3 or .wav audio files")
parser.add_argument("--sample_rate", type=int, default=16000, help="Sampling rate used during ASR model training, Hz")
parser.add_argument("--n_jobs", default=-2, type=int, help="The maximum number of concurrently running jobs")
parser.add_argument(
    "--language",
    type=str,
    default="en",
    choices=["en", "ru", "de", "es", 'other'],
    help='Add target language based on the num2words list of supported languages',
)
parser.add_argument(
    "--cut_prefix", type=int, default=0, help="Number of seconds to cut from the beginning of the audio files.",
)
parser.add_argument(
    "--model", type=str, default="QuartzNet15x5Base-En", help="Pre-trained model name or path to model checkpoint"
)
parser.add_argument(
    "--max_length", type=int, default=40, help="Max number of words of the text segment for alignment."
)
parser.add_argument(
    "--additional_split_symbols",
    type=str,
    default="",
    help="Additional symbols to use for \
    sentence split if eos sentence split resulted in sequence longer than --max_length. "
    "Use '|' as a separator between symbols, for example: ';|:' ",
)
parser.add_argument(
    "--use_nemo_normalization",
    action="store_true",
    help="Set to True to use NeMo Normalization tool to convert numbers from written to spoken format.",
)


def process_audio(in_file: str, wav_file: str = None, cut_prefix: int = 0, sample_rate: int = 16000):
    """Process audio file: .mp3 to .wav conversion and cut a few seconds from the beginning of the audio

    Args:
        in_file: path to the .mp3 or .wav file for processing
        wav_file: path to the output .wav file
        cut_prefix: number of seconds to cut from the beginning of the audio file
        sample_rate: target sampling rate
    """
    try:
        info = mediainfo(in_file)
        sound = AudioSegment.from_file(in_file, start_second=cut_prefix)
        if info["sample_rate"] != str(sample_rate):
            sound = sound.set_frame_rate(sample_rate)
        if info["channels"] != 1:
            sound = sound.set_channels(1)
        sound.export(wav_file, format="wav")
    except Exception as e:
        print(f'{in_file} skipped - {e}')


def split_text(
    in_file: str,
    out_file: str,
    vocabulary: List[str],
    language="en",
    remove_brackets=True,
    do_lower_case=True,
    max_length=100,
    additional_split_symbols=None,
    use_nemo_normalization=False,
):
    """
    Breaks down the in_file roughly into sentences. Each sentence will be on a separate line.
    Written form of the numbers will be converted to its spoken equivalent, OOV punctuation will be removed.

    Args:
        in_file: path to original transcript
        out_file: path to the output file
        vocabulary: ASR model vocabulary
        language: text language
        remove_brackets: Set to True if square [] and curly {} brackets should be removed from text.
            Text in square/curly brackets often contains inaudible fragments like notes or translations
        do_lower_case: flag that determines whether to apply lower case to the in_file text
        max_length: Max number of words of the text segment for alignment
        additional_split_symbols: Additional symbols to use for sentence split if eos sentence split resulted in
            segments longer than --max_length
        use_nemo_normalization: Set to True to use NeMo normalization tool to convert numbers from written to spoken
            format. Normalization using num2words will be applied afterwards to make sure there are no numbers present
            in the text, otherwise they will be replaced with a space and that could deteriorate segmentation results.
    """
    print(f"Splitting text in {in_file} into sentences.")
    with open(in_file, "r") as f:
        transcript = f.read()

    # remove some symbols for better split into sentences
    transcript = (
        transcript.replace("\n", " ")
        .replace("\t", " ")
        .replace("…", "...")
        .replace("\\", " ")
        .replace("--", " -- ")
        .replace(". . .", "...")
    )
    # remove extra space
    transcript = re.sub(r" +", " ", transcript)
    transcript = re.sub(r"(\.+)", ". ", transcript)

    if remove_brackets:
        transcript = re.sub(r'(\[.*?\])', ' ', transcript)
        # remove text in curly brackets
        transcript = re.sub(r'(\{.*?\})', ' ', transcript)

    lower_case_unicode = ''
    upper_case_unicode = ''
    if language == "ru":
        lower_case_unicode = '\u0430-\u04FF'
        upper_case_unicode = '\u0410-\u042F'
    elif language not in ["ru", "en"]:
        print(f"Consider using {language} unicode letters for better sentence split.")

    # remove space in the middle of the lower case abbreviation to avoid splitting into separate sentences
    matches = re.findall(r'[a-z' + lower_case_unicode + ']\.\s[a-z' + lower_case_unicode + ']\.', transcript)
    for match in matches:
        transcript = transcript.replace(match, match.replace('. ', '.'))

    # find phrases in quotes
    with_quotes = re.finditer(r'“[A-Za-z ?]+.*?”', transcript)
    sentences = []
    last_idx = 0
    for m in with_quotes:
        match = m.group()
        match_idx = m.start()
        if last_idx < match_idx:
            sentences.append(transcript[last_idx:match_idx])
        sentences.append(match)
        last_idx = m.end()
    sentences.append(transcript[last_idx:])
    sentences = [s.strip() for s in sentences if s.strip()]

    # Read and split transcript by utterance (roughly, sentences)
    split_pattern = f"(?<!\w\.\w.)(?<![A-Z{upper_case_unicode}][a-z{lower_case_unicode}]\.)(?<![A-Z{upper_case_unicode}]\.)(?<=\.|\?|\!|\.”|\?”\!”)\s"

    new_sentences = []
    for sent in sentences:
        new_sentences.extend(regex.split(split_pattern, sent))
    sentences = [s.strip() for s in new_sentences if s.strip()]

    def additional_split(sentences, split_on_symbols):
        if len(split_on_symbols) == 0:
            return sentences

        split_on_symbols = split_on_symbols.split("|")

        def _split(sentences, delimiter):
            result = []
            for sent in sentences:
                split_sent = sent.split(delimiter)
                # keep the delimiter
                split_sent = [(s + delimiter).strip() for s in split_sent[:-1]] + [split_sent[-1]]

                if "," in delimiter:
                    # split based on comma usually results in too short utterance, combine sentences
                    # that result in a single word split. It's usually not recommended to do that for other delimiters.
                    comb = []
                    for s in split_sent:
                        MIN_LEN = 2
                        # if the previous sentence is too short, combine it with the current sentence
                        if len(comb) > 0 and (len(comb[-1].split()) <= MIN_LEN or len(s.split()) <= MIN_LEN):
                            comb[-1] = comb[-1] + " " + s
                        else:
                            comb.append(s)
                    result.extend(comb)
                else:
                    result.extend(split_sent)
            return result

        another_sent_split = []
        for sent in sentences:
            split_sent = [sent]
            for delimiter in split_on_symbols:
                split_sent = _split(split_sent, delimiter + " ")
            another_sent_split.extend(split_sent)

        sentences = [s.strip() for s in another_sent_split if s.strip()]
        return sentences

    sentences = additional_split(sentences, additional_split_symbols)

    vocabulary_symbols = []
    for x in vocabulary:
        if x != "<unk>":
            # for BPE models
            vocabulary_symbols.extend([x for x in x.replace("##", "").replace("▁", "")])
    vocabulary_symbols = list(set(vocabulary_symbols))
    vocabulary_symbols += [x.upper() for x in vocabulary_symbols]

    # check to make sure there will be no utterances for segmentation with only OOV symbols
    vocab_no_space_with_digits = set(vocabulary_symbols + [str(i) for i in range(10)])
    if " " in vocab_no_space_with_digits:
        vocab_no_space_with_digits.remove(" ")

    sentences = [
        s.strip() for s in sentences if len(vocab_no_space_with_digits.intersection(set(s.lower()))) > 0 and s.strip()
    ]

    # when no punctuation marks present in the input text, split based on max_length
    if len(sentences) == 1:
        sent = sentences[0].split()
        sentences = []
        for i in range(0, len(sent), max_length):
            sentences.append(" ".join(sent[i : i + max_length]))
    sentences = [s.strip() for s in sentences if s.strip()]

    # save split text with original punctuation and case
    out_dir, out_file_name = os.path.split(out_file)
    with open(os.path.join(out_dir, out_file_name[:-4] + "_with_punct.txt"), "w") as f:
        f.write(re.sub(r' +', ' ', "\n".join(sentences)))

    # substitute common abbreviations before applying lower case
    if language == "ru":
        for k, v in RU_ABBREVIATIONS.items():
            sentences = [s.replace(k, v) for s in sentences]
        # replace Latin characters with Russian
        for k, v in LATIN_TO_RU.items():
            sentences = [s.replace(k, v) for s in sentences]

    if language == "en" and use_nemo_normalization:
        if not NEMO_NORMALIZATION_AVAILABLE:
            raise ValueError("NeMo normalization tool is not installed.")

        print("Using NeMo normalization tool...")
        normalizer = Normalizer(input_case="cased", cache_dir=os.path.join(os.path.dirname(out_file), "en_grammars"))
        sentences_norm = normalizer.normalize_list(sentences, verbose=False, punct_post_process=True)
        if len(sentences_norm) != len(sentences):
            raise ValueError("Normalization failed, number of sentences does not match.")
        else:
            sentences = sentences_norm

    sentences = '\n'.join(sentences)

    # replace numbers with num2words
    try:
        p = re.compile("\d+")
        new_text = ""
        match_end = 0
        for i, m in enumerate(p.finditer(sentences)):
            match = m.group()
            match_start = m.start()
            if i == 0:
                new_text = sentences[:match_start]
            else:
                new_text += sentences[match_end:match_start]
            match_end = m.end()
            new_text += sentences[match_start:match_end].replace(match, num2words(match, lang=language))
        new_text += sentences[match_end:]
        sentences = new_text
    except NotImplementedError:
        print(
            f"{language} might be missing in 'num2words' package. Add required language to the choices for the"
            f"--language argument."
        )
        raise

    sentences = re.sub(r' +', ' ', sentences)

    with open(os.path.join(out_dir, out_file_name[:-4] + "_with_punct_normalized.txt"), "w") as f:
        f.write(sentences)

    if do_lower_case:
        sentences = sentences.lower()

    symbols_to_remove = ''.join(set(sentences).difference(set(vocabulary_symbols + ["\n", " "])))
    sentences = sentences.translate(''.maketrans(symbols_to_remove, len(symbols_to_remove) * " "))

    # remove extra space
    sentences = re.sub(r' +', ' ', sentences)
    with open(out_file, "w") as f:
        f.write(sentences)


if __name__ == "__main__":
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    text_files = []
    if args.in_text:
        if args.model is None:
            raise ValueError(f"ASR model must be provided to extract vocabulary for text processing")
        elif os.path.exists(args.model):
            model_cfg = ASRModel.restore_from(restore_path=args.model, return_config=True)
            classpath = model_cfg.target  # original class path
            imported_class = model_utils.import_class_by_path(classpath)  # type: ASRModel
            print(f"Restoring model : {imported_class.__name__}")
            asr_model = imported_class.restore_from(restore_path=args.model)  # type: ASRModel
            model_name = os.path.splitext(os.path.basename(args.model))[0]
        else:
            # restore model by name
            asr_model = ASRModel.from_pretrained(model_name=args.model)  # type: ASRModel
            model_name = args.model

        vocabulary = asr_model.cfg.decoder.vocabulary

        if os.path.isdir(args.in_text):
            text_files = Path(args.in_text).glob(("*.txt"))
        else:
            text_files.append(Path(args.in_text))
        for text in text_files:
            base_name = os.path.basename(text)[:-4]
            out_text_file = os.path.join(args.output_dir, base_name + ".txt")

            split_text(
                text,
                out_text_file,
                vocabulary=vocabulary,
                language=args.language,
                max_length=args.max_length,
                additional_split_symbols=args.additional_split_symbols,
                use_nemo_normalization=args.use_nemo_normalization,
            )
        print(f"Processed text saved at {args.output_dir}")

    if args.audio_dir:
        if not os.path.exists(args.audio_dir):
            raise ValueError(f"{args.audio_dir} not found. '--audio_dir' should contain .mp3 or .wav files.")

        audio_paths = list(Path(args.audio_dir).glob("*"))

        normalized_lines = Parallel(n_jobs=args.n_jobs)(
            delayed(process_audio)(
                audio_paths[i],
                os.path.join(args.output_dir, os.path.splitext(audio_paths[i].name)[0] + ".wav"),
                args.cut_prefix,
                args.sample_rate,
            )
            for i in tqdm(range(len(audio_paths)))
        )

    print("Data preparation is complete.")
