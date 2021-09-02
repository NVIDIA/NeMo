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
import multiprocessing
import os
import re
from pathlib import Path
from typing import List

import regex
import scipy.io.wavfile as wav
from normalization_helpers import LATIN_TO_RU, RU_ABBREVIATIONS
from num2words import num2words

from nemo.collections import asr as nemo_asr

try:
    from nemo_text_processing.text_normalization.normalize import Normalizer

    NEMO_NORMALIZATION_AVAILABLE = True
except (ModuleNotFoundError, ImportError):
    NEMO_NORMALIZATION_AVAILABLE = False


parser = argparse.ArgumentParser(description="Prepares text and audio files for segmentation")
parser.add_argument("--in_text", type=str, default=None, help='Path to a text file or a directory with .txt files')
parser.add_argument("--output_dir", type=str, required=True, help='Path to output directory')
parser.add_argument("--audio_dir", type=str, help='Path to folder with .mp3 or .wav audio files')
parser.add_argument(
    "--audio_format", type=str, default='.mp3', choices=['.mp3', '.wav'], help='Audio files format in --audio_dir'
)
parser.add_argument('--sample_rate', type=int, default=16000, help='Sampling rate used during ASR model training')
parser.add_argument(
    '--language', type=str, default='eng', choices=['eng', 'ru', 'add other languages supported by num2words.']
)
parser.add_argument(
    '--cut_prefix', type=int, default=0, help='Number of seconds to cut from the beginning of the audio files.',
)
parser.add_argument(
    '--model', type=str, default='QuartzNet15x5Base-En', help='Pre-trained model name or path to model checkpoint'
)
parser.add_argument('--min_length', type=int, default=0, help='Min number of chars of the text segment for alignment.')
parser.add_argument(
    '--max_length', type=int, default=100, help='Max number of chars of the text segment for alignment.'
)
parser.add_argument(
    '--additional_split_symbols',
    type=str,
    default='',
    help='Additional symbols to use for \
    sentence split if eos sentence split resulted in sequence longer than --max_length. '
    'Use "|" as a separator between symbols, for example: ";|:" ',
)
parser.add_argument(
    '--use_nemo_normalization',
    action='store_true',
    help='Set to True to use NeMo Normalization tool to convert numbers from written to spoken format.',
)


def convert_audio(in_file: str, wav_file: str = None, sample_rate: int = 16000) -> str:
    """
    Convert .mp3 to .wav and/or change sample rate if needed

    Args:
        in_file: Path to .mp3 or .wav file
        sample_rate: Desired sample rate

    Returns:
        path to .wav file
    """
    print(f"Converting {in_file} to .wav format with sample rate {sample_rate}")
    if not os.path.exists(in_file):
        raise ValueError(f'{in_file} not found')
    if wav_file is None:
        wav_file = in_file.replace(os.path.splitext(in_file)[-1], f"_{sample_rate}.wav")

    os.system(
        f'ffmpeg -i {in_file} -acodec pcm_s16le -ac 1 -af aresample=resampler=soxr -ar {sample_rate} {wav_file} -y'
    )
    return wav_file


def process_audio(in_file: str, wav_file: str = None, cut_prefix: int = 0, sample_rate: int = 16000):
    """Process audio file: .mp3 to .wav conversion and cut a few seconds from the beginning of the audio

    Args:
        in_file: path to the .mp3 or .wav file for processing
        wav_file: path to the output .wav file
        cut_prefix: number of seconds to cut from the beginning of the audio file
        sample_rate: target sampling rate
    """
    wav_audio = convert_audio(str(in_file), wav_file, sample_rate)

    if cut_prefix > 0:
        # cut a few seconds of audio from the beginning
        sample_rate, signal = wav.read(wav_audio)
        wav.write(wav_audio, data=signal[cut_prefix * sample_rate :], rate=sample_rate)


def split_text(
    in_file: str,
    out_file: str,
    vocabulary: List[str] = None,
    language='eng',
    remove_brackets=True,
    do_lower_case=True,
    min_length=0,
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
        min_length: Min number of chars of the text segment for alignment. Short segments will be combined to be
            at least min_length (not recommended for multi speaker data).
        max_length: Max number of chars of the text segment for alignment
        additional_split_symbols: Additional symbols to use for sentence split if eos sentence split resulted in
            segments longer than --max_length
        use_nemo_normalization: Set to True to use NeMo normalization tool to convert numbers from written to spoken
            format. Normalization using num2words will be applied afterwards to make sure there are no numbers present
            in the text, otherwise they will be replaced with a space and that could deteriorate segmentation results.
    """

    print(f'Splitting text in {in_file} into sentences.')
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
        .replace("‘", "’")
    )
    # remove extra space
    transcript = re.sub(r' +', ' ', transcript)
    transcript = re.sub(r'(\.+)', '. ', transcript)

    if remove_brackets:
        transcript = re.sub(r'(\[.*?\])', ' ', transcript)
        # remove text in curly brackets
        transcript = re.sub(r'(\{.*?\})', ' ', transcript)

    lower_case_unicode = ''
    upper_case_unicode = ''
    if language == 'ru':
        lower_case_unicode = '\u0430-\u04FF'
        upper_case_unicode = '\u0410-\u042F'
    elif language not in ['ru', 'eng']:
        print(f'Consider using {language} unicode letters for better sentence split.')

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

    def additional_split(sentences, split_on_symbols, max_length):
        if len(split_on_symbols) == 0:
            return sentences

        split_on_symbols = split_on_symbols.split('|')

        def _split(sentences, delimiter, max_length):
            result = []
            for s in sentences:
                if len(s) <= max_length:
                    result.append(s)
                else:
                    split_sent = s.split(delimiter)
                    result.extend([s + delimiter for s in split_sent[:-1]] + [split_sent[-1]])
            return result

        another_sent_split = []
        for sent in sentences:
            split_sent = [sent]
            for delimiter in split_on_symbols:
                split_sent = _split(split_sent, delimiter + ' ', max_length)
            another_sent_split.extend(split_sent)

        sentences = [s.strip() for s in another_sent_split if s.strip()]
        return sentences

    sentences = additional_split(sentences, additional_split_symbols, max_length)

    # check to make sure there will be no utterances for segmentation with only OOV symbols
    vocab_no_space_with_digits = set(vocabulary + [i for i in range(10)])
    vocab_no_space_with_digits.remove(' ')
    sentences = [s for s in sentences if len(vocab_no_space_with_digits.intersection(set(s))) > 0]

    if min_length > 0:
        sentences_comb = []
        sentences_comb.append(sentences[0])
        # combines short sentence
        for i in range(1, len(sentences)):
            if len(sentences_comb[-1]) < min_length or len(sentences[i]) < min_length:
                sentences_comb[-1] += ' ' + sentences[i].strip()
            else:
                sentences_comb.append(sentences[i].strip())
        sentences = sentences_comb

    sentences = [s.strip() for s in sentences if s.strip()]

    # save split text with original punctuation and case
    out_dir, out_file_name = os.path.split(out_file)
    with open(os.path.join(out_dir, out_file_name[:-4] + '_with_punct.txt'), "w") as f:
        f.write("\n".join(sentences))

    # substitute common abbreviations before applying lower case
    if language == 'ru':
        for k, v in RU_ABBREVIATIONS.items():
            sentences = [s.replace(k, v) for s in sentences]

    if language == 'ru':
        # replace Latin characters with Russian
        for k, v in LATIN_TO_RU.items():
            sentences = [s.replace(k, v) for s in sentences]

    if language == 'eng' and use_nemo_normalization:
        if not NEMO_NORMALIZATION_AVAILABLE:
            raise ValueError(f'NeMo normalization tool is not installed.')

        print('Using NeMo normalization tool...')
        normalizer = Normalizer(input_case='cased')
        sentences_norm = normalizer.normalize_list(sentences, verbose=False)
        if len(sentences_norm) != len(sentences):
            raise ValueError(f'Normalization failed, number of sentences does not match.')

    sentences = '\n'.join(sentences)

    # replace numbers with num2words
    try:
        p = re.compile("\d+")
        new_text = ''
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
            f'{language} might be missing in "num2words" package. Add required language to the choices for the'
            f'--language argument.'
        )
        raise

    sentences = (
        sentences.replace("’", "'")
        .replace("»", '"')
        .replace("«", '"')
        .replace("\\", "")
        .replace("”", '"')
        .replace("„", '"')
        .replace("´", "'")
        .replace("-- --", "--")
        .replace("--", " -- ")
        .replace("’", "'")
        .replace('“', '"')
        .replace('“', '"')
        .replace("‘", "'")
        .replace('—', '-')
        .replace("- -", "--")
        .replace('`', "'")
        .replace(' !', '!')
        .replace(' ?', '?')
        .replace(' ,', ',')
        .replace(' .', '.')
        .replace(' ;', ';')
        .replace(' :', ':')
        .replace('!!', '!')
        .replace('--', '-')
        .replace('“', '"')
        .replace(', , ', ', ')
        .replace('=', '')
    )

    allowed_punct = [',', '.', '?', '!', ':', ';', '-', '"', '(', ')']
    # clean up normalized text and keep only allowed_punct and ASR vocabulary (lower and upper case)
    symbols_to_remove = ''.join(
        set(sentences).difference(set(vocabulary + [s.upper() for s in vocabulary] + ['\n'] + allowed_punct))
    )
    sentences_norm = sentences.translate(''.maketrans(symbols_to_remove, len(symbols_to_remove) * ' '))

    with open(os.path.join(out_dir, out_file_name[:-4] + '_with_punct_normalized.txt'), "w") as f:
        f.write(sentences_norm)

    if do_lower_case:
        sentences = sentences.lower()

    # remove all OOV symbols
    symbols_to_remove = ''.join(set(sentences).difference(set(vocabulary + ['\n'])))
    sentences = sentences.translate(''.maketrans(symbols_to_remove, len(symbols_to_remove) * ' '))

    # remove extra space
    sentences = re.sub(r' +', ' ', sentences)
    with open(out_file, "w") as f:
        f.write(sentences)


if __name__ == '__main__':
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    text_files = []
    if args.in_text:
        vocabulary = None
        if args.model is None:
            print(f"No model provided, vocabulary won't be used")
        elif os.path.exists(args.model):
            asr_model = nemo_asr.models.EncDecCTCModel.restore_from(args.model)
            vocabulary = asr_model.cfg.decoder.vocabulary
        elif args.model in nemo_asr.models.EncDecCTCModel.get_available_model_names():
            asr_model = nemo_asr.models.EncDecCTCModel.from_pretrained(args.model)
            vocabulary = asr_model.cfg.decoder.vocabulary
        else:
            raise ValueError(
                f'Provide path to the pretrained checkpoint or choose from {nemo_asr.models.EncDecCTCModel.get_available_model_names()}'
            )

        if os.path.isdir(args.in_text):
            text_files = Path(args.in_text).glob(("*.txt"))
        else:
            text_files.append(Path(args.in_text))
        for text in text_files:
            base_name = os.path.basename(text)[:-4]
            out_text_file = os.path.join(args.output_dir, base_name + '.txt')

            split_text(
                text,
                out_text_file,
                vocabulary=vocabulary,
                language=args.language,
                min_length=args.min_length,
                max_length=args.max_length,
                additional_split_symbols=args.additional_split_symbols,
                use_nemo_normalization=args.use_nemo_normalization,
            )
        print(f'Processed text saved at {args.output_dir}')

    if args.audio_dir:
        if not os.path.exists(args.audio_dir):
            raise ValueError(f'{args.audio_dir} not found. "--audio_dir" should contain .mp3 or .wav files.')

        audio_paths = list(Path(args.audio_dir).glob(f"*{args.audio_format}"))

        workers = []
        for i in range(len(audio_paths)):
            wav_file = os.path.join(args.output_dir, audio_paths[i].name.replace(args.audio_format, ".wav"))
            worker = multiprocessing.Process(
                target=process_audio, args=(audio_paths[i], wav_file, args.cut_prefix, args.sample_rate),
            )
            workers.append(worker)
            worker.start()
        for w in workers:
            w.join()

    print('Done.')
