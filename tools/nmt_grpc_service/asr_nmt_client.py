# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import argparse
import sys
import wave
import os
sys.path.append(os.path.join(os.getcwd(), 'api'))

import api.nmt_pb2 as nmt
import api.nmt_pb2_grpc as nmtsrv
import grpc
import riva_api.audio_pb2 as riva
import riva_api.riva_asr_pb2 as rivaasr
import riva_api.riva_asr_pb2_grpc as rivaasr_srv
import pyaudio


def get_args():
    parser = argparse.ArgumentParser(description="Streaming transcription via Jarvis AI Services")
    parser.add_argument("--jarvis-server", default="localhost:50051", type=str, help="URI to GRPC server endpoint")
    parser.add_argument("--audio-file", required=True, help="path to local file to stream")
    parser.add_argument("--output-device", type=int, default=None, help="output device to use")
    parser.add_argument("--list-devices", action="store_true", help="list output devices indices")
    parser.add_argument("--nmt-server", default="localhost:50052", help="port on which NMT server runs")
    parser.add_argument("--asr_only", action="store_true", help="Whether to skip MT and just display")
    parser.add_argument("--target_language", default="es", help="Target language to translate into.")
    parser.add_argument(
        "--asr_punctuation",
        action="store_true",
        help="Whether to use Riva's punctuation model for ASR transcript postprocessing.",
    )
    return parser.parse_args()


def listen_print_loop(responses, nmt_stub, target_language, asr_only=False):
    num_chars_printed = 0
    prev_utterances = []
    for response in responses:
        if not response.results:
            continue
        result = response.results[0]
        if not result.alternatives:
            continue
        transcript = result.alternatives[0].transcript
        original_transcript = transcript
        if not asr_only:
            req = nmt.TranslateTextRequest(texts=[transcript], source_language='en', target_language=target_language)
            translation = nmt_stub.TranslateText(req).translations[0].translation
            transcript = translation
        overwrite_chars = ' ' * (num_chars_printed - len(transcript))
        if not result.is_final:
            sys.stdout.write(">> " + transcript + overwrite_chars + '\r')
            sys.stdout.flush()
            num_chars_printed = len(transcript) + 3
        else:
            print("## " + transcript + overwrite_chars + "\n")
            num_chars_printed = 0
            prev_utterances.append(original_transcript)


CHUNK = 1024
args = get_args()
wf = wave.open(args.audio_file, 'rb')
channel = grpc.insecure_channel(args.jarvis_server)
client = rivaasr_srv.RivaSpeechRecognitionStub(channel)
nmt_channel = grpc.insecure_channel(args.nmt_server)
nmt_stub = nmtsrv.RivaTranslateStub(nmt_channel)
config = rivaasr.RecognitionConfig(
    encoding=riva.AudioEncoding.LINEAR_PCM,
    sample_rate_hertz=wf.getframerate(),
    language_code="en-US",
    max_alternatives=1,
    enable_automatic_punctuation=args.asr_punctuation,
)
streaming_config = rivaasr.StreamingRecognitionConfig(config=config, interim_results=True)

# instantiate PyAudio (1)
p = pyaudio.PyAudio()
if args.list_devices:
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if info['maxOutputChannels'] < 1:
            continue
        print(f"{info['index']}: {info['name']}")
    sys.exit(0)

# open stream (2)
stream = p.open(
    output_device_index=args.output_device,
    format=p.get_format_from_width(wf.getsampwidth()),
    channels=wf.getnchannels(),
    rate=wf.getframerate(),
    output=True,
)

# read data
def generator(w, s):
    d = w.readframes(CHUNK)
    yield rivaasr.StreamingRecognizeRequest(streaming_config=s)
    while len(d) > 0:
        yield rivaasr.StreamingRecognizeRequest(audio_content=d)
        stream.write(d)
        d = w.readframes(CHUNK)
    return


responses = client.StreamingRecognize(generator(wf, streaming_config))
listen_print_loop(responses, nmt_stub, target_language=args.target_language, asr_only=args.asr_only)
# stop stream (4)
stream.stop_stream()
stream.close()
# close PyAudio (5)
p.terminate()
