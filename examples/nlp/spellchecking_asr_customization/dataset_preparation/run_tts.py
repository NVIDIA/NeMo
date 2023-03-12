# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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


from argparse import ArgumentParser

import soundfile as sf
import torch

from nemo.collections.tts.models.base import SpectrogramGenerator, Vocoder
from nemo.collections.tts.torch.g2ps import EnglishG2p
from nemo.collections.tts.torch.tts_tokenizers import EnglishPhonemesTokenizer

parser = ArgumentParser(description="Run TTS")
parser.add_argument("--input_name", type=str, required=True, help="Input file")
parser.add_argument("--output_dir", type=str, required=True, help="Output dir with .wav files")
parser.add_argument("--output_manifest", type=str, required=True, help="Output manifest file")

args = parser.parse_args()

# Download and load the pretrained fastpitch model
spec_generator = SpectrogramGenerator.from_pretrained(model_name="tts_en_fastpitch").cuda()
spec_generator.eval()

# Download and load the pretrained hifigan model
vocoder = Vocoder.from_pretrained(model_name="tts_hifigan").cuda()

text_tokenizer = EnglishPhonemesTokenizer(
    punct=True, stresses=True, chars=True, space=' ', apostrophe=True, pad_with_space=True, g2p=EnglishG2p(),
)

out_manifest = open(args.output_manifest, "w", encoding="utf-8")

lid = 0
with open(args.input_name, "r", encoding="utf-8") as inp:
    for line in inp:
        line = line.strip()
        raw, inp = line.split("\t")

        # arg: list of phonemes e.g. ["AA1", "M", "AH0"]
        parsed = text_tokenizer.encode_from_g2p(inp.split(","))

        parsed = torch.Tensor(parsed).to(dtype=torch.int64, device=spec_generator.device)
        parsed = torch.unsqueeze(parsed, 0)

        # They then take the tokenized string and produce a spectrogram
        spectrogram = spec_generator.generate_spectrogram(tokens=parsed)

        # Finally, a vocoder converts the spectrogram to audio
        audio = vocoder.convert_spectrogram_to_audio(spec=spectrogram)

        # Save the audio to disk in a file called speech.wav
        # Note vocoder return a batch of audio. In this example, we just take the first and only sample.
        filename = args.output_dir + "/" + str(lid) + ".wav"
        sf.write(filename, audio.to('cpu').detach().numpy()[0], 22050)
        # {"audio_filepath": "tts/1.wav", "text": "ndimbati"}
        out_manifest.write(
            "{\"audio_filepath\": \"" + filename + "\", \"text\": \"" + raw + "\", \"g2p\": \"" + inp + "\"}\n"
        )
        lid += 1

out_manifest.close()
