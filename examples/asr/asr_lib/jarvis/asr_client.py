#!/usr/bin/python

# Copyright (c) 2018-2019, NVIDIA CORPORATION. All rights reserved.
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

import sys
sys.path.append("../../modules/tensorrt-inference-server/builddir/trtis-clients/install/python/")
import argparse
import numpy as np
import os
from tensorrtserver.api import *
from speech_utils import AudioSegment, SpeechClient, get_speech_features
import soundfile
import pyaudio as pa
import threading
import math
import time
import glob
from multiprocessing.dummy import Pool as ThreadPool
import multiprocessing

FLAGS = None


# read audio chunk from a file
def get_audio_chunk_from_soundfile(sf, chunk_size):

    dtype_options = {'PCM_16': 'int16', 'PCM_32': 'int32', 'FLOAT': 'float32'}
    dtype_file = sf.subtype
    if dtype_file in dtype_options:
        dtype = dtype_options[dtype_file]
    else:
        dtype = 'float32'
    audio_signal = sf.read(chunk_size, dtype=dtype)
    end = False
    # pad to chunk size
    if len(audio_signal) < chunk_size:
        end = True
        audio_signal = np.pad(audio_signal, (0, chunk_size-len(
            audio_signal)), mode='constant')
    return audio_signal, end


# generator that returns chunks of audio data from file
def audio_generator_from_file(input_filename, target_sr, chunk_duration):

    sf = soundfile.SoundFile(input_filename, 'rb')
    chunk_size = int(chunk_duration*sf.samplerate)
    start = True
    end = False

    while not end:

        audio_signal, end = get_audio_chunk_from_soundfile(sf, chunk_size)

        audio_segment = AudioSegment(audio_signal, sf.samplerate, target_sr)

        yield audio_segment.samples, target_sr, start, end
        start = False

    sf.close()


# generator that returns chunks of audio data from file
class AudioGeneratorFromMicrophone:

    def __init__(self,input_device_id, target_sr, chunk_duration):

        self.recording_state = "init"
        self.target_sr  = target_sr
        self.chunk_duration = chunk_duration

        self.p = pa.PyAudio()

        device_info = self.p.get_host_api_info_by_index(0)
        num_devices = device_info.get('deviceCount')
        devices = {}
        for i in range(0, num_devices):
            if (self.p.get_device_info_by_host_api_device_index(0, i).get(
                'maxInputChannels')) > 0:
                devices[i] = self.p.get_device_info_by_host_api_device_index(
                    0, i)

        if (len(devices) == 0):
            raise RuntimeError("Cannot find any valid input devices")

        if input_device_id is None or input_device_id not in \
            devices.keys():
            print("\nInput Devices:")
            for id, info in devices.items():
                print("{}: {}".format(id,info.get("name")))
            input_device_id = int(input("Enter device id to use: "))

        self.input_device_id = input_device_id


    def generate_audio(self, streaming=True):

        chunk_size = int(self.chunk_duration*self.target_sr)
        print('chunk_duration = ' + str(self.chunk_duration))
        print('chunk_size = ' + str(chunk_size))

        self. recording_state = "init"

        def keyboard_listener():
            input("Press Enter to start and end recording...")
            self.recording_state = "capture"
            print("Recording...")

            input("")
            self.recording_state = "release"

        listener = threading.Thread(target=keyboard_listener)
        listener.start()

        start = True

        stream_initialized = False
        step = 0
        audio_signal = 0
        audio_segment = 0
        end = False
        while self.recording_state != "release":
            try:
                if self.recording_state == "capture":

                    if not stream_initialized:
                        stream = self.p.open(
                            format=pa.paInt16,
                            channels=1,
                            rate=self.target_sr,
                            input=True,
                            input_device_index=self.input_device_id,
                            frames_per_buffer=chunk_size)
                        stream_initialized = True

                    # Read audio chunk from microphone
                    audio_signal = stream.read(chunk_size)
                    if self.recording_state == "release":
                      end = True
                    audio_signal = np.frombuffer(audio_signal,dtype=np.int16)
                    audio_segment = AudioSegment(audio_signal,
                                                              self.target_sr,
                                                              self.target_sr)

                    yield audio_segment.samples, self.target_sr, start, end

                    start = False
                    step += 1
            except Exception as e:
                print(e)
                break


        stream.close()
        self.p.terminate()

    def generate_audio_signal(self):
        audio_samples = []
        step = 0
        for audio_sample, sr, start, end in self.generate_audio(False):
            if step == 0:
                audio_samples = audio_sample
            else:
                audio_samples = np.concatenate((audio_samples,
                                                audio_sample))
            step += 1

        return audio_samples

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', action="store_true", required=False,
                        default=False, help='Enable verbose output')
    parser.add_argument('-m', '--model_name', type=str, required=False,
                        default=None,
                        help='Name of model')
    parser.add_argument('--use_client_featurizer', action="store_true",
                        required=False,
                        help='Use feature extraction in TRTIS server')
    parser.add_argument('--use_saved_features', action="store_true",
                        required=False,
                        help='Use pre-computed saved features')
    parser.add_argument('--save_features', action="store_true",
                        required=False, default=False,
                        help='Use feature extraction in TRTIS server')
    parser.add_argument('-x', '--model-version', type=int, required=False,
                        default=None,
                        help='Version of model. Default is to use latest '
                             'version.')
    parser.add_argument('--backend', required=False, default="librosa",
                        help='Backend to use for feature extraction')
    parser.add_argument('--sample_rate', type=int, required=False,
                        default=16000, help='Sample rate.')
    parser.add_argument('--batch_size', type=int, required=False, default=1,
                        help='batch size')
    parser.add_argument('--model_platform', required=False,
                        default='onnx',
                        help='Jasper model platform')
    parser.add_argument('--num_audio_features', type=int, required=False,
                        default=64,
                        help='Number of audio features to extract from signal')
    parser.add_argument('--feature_type', required=False, default="logfbank",
                        help='Type of audio features to extract from signal')
    parser.add_argument('--trim_silence', action='store_true',
                        help='Remove silence at end of audio file')
    parser.add_argument('--norm_per_feature', action='store_true',
                        help='Normalize per feature')
    parser.add_argument('--window_size', type=float, required=False,
                        default=0.02,
                        help='Size of analysis window in milliseconds')
    parser.add_argument('--window_stride', type=float, required=False,
                        default=0.01,
                        help='Stride of analysis window in milliseconds')
    parser.add_argument('--window', required=False, default="hanning",
                        help='windowing function')
    parser.add_argument('--n_fft', required=False, default=None,
                        help='Size of fft window to use if features '
                             'requires fft')
    parser.add_argument('--pad_to', required=False, default=16,
                        help='Padding audio features (max or integer value)')
    parser.add_argument('--dither', type=float, required=False,
                        default=0.00001,
                        help='Weight of Gaussian noise to apply to input '
                             'signal')
    parser.add_argument('--gain', type=float, required=False,
                        default=1.,
                        help='Gain used when normalizing audio signal')
    parser.add_argument('-t', '--num_transcripts', type=int, required=False,
                        default=1,
                        help='Number of candidate transcripts to report. '
                             'Default is 1.')
    parser.add_argument('-u', '--url', type=str, required=False,
                        default='localhost:8000',
                        help='Inference server URL. Default is '
                             'localhost:8000.')
    parser.add_argument('-i', '--protocol', type=str, required=False,
                        default='HTTP',
                        help='Protocol (HTTP/gRPC) used to communicate with '
                             'inference service. Default is HTTP.')
    parser.add_argument('--audio_filename', type=str, required=False,
                        default=None,
                        help='Input audio filename / Input folder.')
    parser.add_argument('--input_device_id', type=int, required=False,
                        default=-1,
                        help='Input device id to use to capture audio')
    parser.add_argument('--mode', type=str, required=False,
                        default="synchronous",
                        help='Type of speech recognition')
    parser.add_argument('--input_method', type=str, required=False,
                        default="file",
                        help='Input method: file or microphone')
    parser.add_argument('--use_vad', action='store_true',
                        help='Use VAD algorithm to detect end of sentences')
    parser.add_argument('--chunk_duration', type=float, required=False,
                        default=2.,
                        help="duration of the audio chunk for streaming "
                                "recognition, in seconds")
    parser.add_argument('--create_buffers', action='store_true',
                        help='Create overlapping buffers for streaming')
    parser.add_argument('--min_duration', type=float, required=False,
                        default=2.560,
                        help="minimum duration of the audio in synchronous "
                                "mode, in seconds")
    FLAGS = parser.parse_args()

    protocol = ProtocolType.from_str(FLAGS.protocol)

    valid_model_platforms = {"tensorflow","pyt","onnx", "trt"}
    valid_modes = {"synchronous","streaming"}
    valid_input_methods = {"file","microphone"}

    if FLAGS.mode not in valid_modes:
        raise ValueError("Invalid mode. Valid choices are {}".format(
            valid_modes))

    if FLAGS.model_platform not in valid_model_platforms:
        raise ValueError("Invalid model_platform {}. Valid choices are {"
                         "}".format(FLAGS.model_platform,
            valid_model_platforms))

    if FLAGS.input_method not in valid_input_methods:
        raise ValueError("Invalid input_method. Valid choices are {}".format(
            valid_input_methods))

    model_name = FLAGS.model_name

    if model_name is None:
        if FLAGS.mode == "synchronous":
            if (FLAGS.model_platform.lower() == "tensorflow"):
                model_name = "jasper-asr-tf-ensemble"
            elif (FLAGS.model_platform.lower() == "pyt"):
                model_name = "jasper-asr-pyt-ensemble"
            elif (FLAGS.model_platform.lower() == "onnx"):
                if (FLAGS.use_vad):
                    model_name = "jasper-asr-onnx-ensemble-vad"
                else:
                    model_name = "jasper-asr-onnx-ensemble"
            elif (FLAGS.model_platform.lower() == "trt"):
                if (FLAGS.use_vad):
                    model_name = "jasper-asr-trt-ensemble-vad"
                else:
                    model_name = "jasper-asr-trt-ensemble"
            else:
                raise ValueError("model_platform not recognized. Valid choices are {}".format(valid_model_platforms))
        elif (FLAGS.mode == "streaming"):
            if (FLAGS.model_platform.lower() == "tensorflow"):
                model_name = "jasper-asr-tf-ensemble-streaming"
            elif (FLAGS.model_platform.lower() == "pyt"):
                model_name = "jasper-asr-pyt-ensemble-streaming"
            elif (FLAGS.model_platform.lower() == "onnx"):
                if (FLAGS.use_vad):
                  model_name = "jasper-asr-onnx-ensemble-vad-streaming"
                else:
                  model_name = "jasper-asr-onnx-ensemble-streaming"
            elif (FLAGS.model_platform.lower() == "trt"):
                if (FLAGS.use_vad):
                  model_name = "jasper-asr-trt-ensemble-vad-streaming"
                else:
                  model_name = "jasper-asr-trt-ensemble-streaming"

            else:
                raise ValueError("model_platform not recognized. Valid choices are {}".format(valid_model_platforms))
        else:
            raise ValueError("Invalid mode. Valid choices are {}".format(
                valid_modes))

    duration = 0

    print("connecting to model => " + model_name)

    from_features_state = FLAGS.use_client_featurizer or FLAGS.use_saved_features
    speech_client = SpeechClient(
      FLAGS.url, protocol, model_name, FLAGS.model_version,
      FLAGS.batch_size, model_platform=FLAGS.model_platform,
      verbose=FLAGS.verbose, mode=FLAGS.mode,
      from_features=from_features_state
    )

    speech_features_params = {"num_audio_features": FLAGS.num_audio_features,
                              "input_type": FLAGS.feature_type,
                              "norm_per_feature": FLAGS.norm_per_feature,
                              "window_size": FLAGS.window_size,
                              "window_stride": FLAGS.window_stride,
                              "window": FLAGS.window,
                              "sample_freq": FLAGS.sample_rate,
                              "pad_to": FLAGS.pad_to,
                              "dither": FLAGS.dither,
                              "backend": FLAGS.backend,
                              "num_fft": FLAGS.n_fft,
                              "gain": FLAGS.gain}


    if FLAGS.input_method == "microphone":

        if FLAGS.mode not in ["streaming","synchronous"]:
            raise ValueError("mode must be 'streaming' or 'synchronous' when "
                             "capturing audio from microphone")

        if "SSH_CONNECTION" in os.environ:
            raise EnvironmentError("Microphone not supported via SSH")

        audio_generator = AudioGeneratorFromMicrophone(
            FLAGS.input_device_id,
            target_sr=FLAGS.sample_rate,
            chunk_duration=FLAGS.chunk_duration)

        if FLAGS.mode == 'streaming':
          partial_transcript, partial_wordtimes = speech_client.streaming_recognize(audio_generator.generate_audio(),"mic")
          ti = 1
          for transcript, score in zip(speech_client.base_transcripts,speech_client.base_scores):
            print("Final transcript {0}: {1} ".format(ti, transcript + partial_transcript))
            print("Final score {0}: {1:.5f} ".format(ti, score))
            ti += 1

          words = base_transcripts[0].split();
          print("\nTimestamps:");
          print("word:start(ms):end(ms)");
          print("---");
          for (word, times) in zip(words, speech_client.base_wordtimes):
              print("{0}:{1}:{2}".format(word, times[0], times[1]))
          words = partial_transcript.split();
          for (word, times) in zip(words, partial_wordtimes):
              print("{0}:{1}:{2}".format(word, times[0], times[1]))
          print("---");
          print("Audio processed:", speech_client.audio_processed[0]);

        elif FLAGS.mode == 'synchronous':
            audio_signal = audio_generator.generate_audio_signal()
            speech_client.recognize([audio_signal],[FLAGS.sample_rate],
                                    ["microphone"])

    elif FLAGS.input_method == "file":


        if FLAGS.audio_filename is None:
            raise ValueError("Audio filename not specified. Please use "
                             "--audio_filename <filename> arguments ")

        filenames = []
        if FLAGS.use_saved_features:
           #TODO :- add support for npz, hdf5.
           file_tag = "*.npy"
        else:
           file_tag = "*.wav"
        if os.path.isdir(FLAGS.audio_filename):
            filenames = glob.glob(os.path.join(os.path.abspath(FLAGS.audio_filename), "**", file_tag),
                          recursive=True)
        else:
            filenames = [FLAGS.audio_filename, ]

        filenames.sort()

        if FLAGS.batch_size > 1 and FLAGS.mode != "synchronous":
            raise ValueError("batch_size > 1 currently only supported with "
                             "synchronous mode")

        # Read the audio files
        if FLAGS.mode == "synchronous":

            if FLAGS.use_client_featurizer:
                if (FLAGS.model_platform != "onnx"):
                    raise ValueError("use_client_featurizer only supported "
                                     "with onnx model")
                audio_features = []
                for filename in filenames:
                    print("Reading audio file: ", filename)
                    audio = AudioSegment.from_file(
                        filename,
                        target_sr=FLAGS.sample_rate,
                        offset=0, duration=duration,
                        min_duration=FLAGS.min_duration,
                        trim=FLAGS.trim_silence)

                    audio_feature, out_duration = get_speech_features(
                        audio.samples, audio.sample_rate,
                        speech_features_params)

                    if (FLAGS.save_features):
                        np.save(filename+"." + str(out_duration) + '.features', audio_features)
                    audio_features.append(audio_feature)

                speech_client.recognize_from_features(audio_features,
                                                      filenames)
            elif FLAGS.use_saved_features:
                if (FLAGS.model_platform != "onnx"):
                    raise ValueError("use_saved_features only supported "
                                     "with onnx model")
                audio_features = []
                for filename in filenames:
                    print("Reading feature file: ", filename)
                    #TODO :- add support for npz, hdf5.
                    feat = np.load(filename, allow_pickle=True)
                    audio_feature = feat[0]
                    audio_features.append(audio_feature)

                speech_client.recognize_from_features(audio_features,
                                                      filenames)
            else:
                # Group requests in batches
                audio_idx = 0
                last_request = False

                while not last_request:
                    batch_audio_samples = []
                    batch_audio_sample_rates = []
                    batch_filenames = []

                    for idx in range(FLAGS.batch_size):
                        filename = filenames[audio_idx]
                        print("Reading audio file: ", filename)
                        audio = AudioSegment.from_file(
                            filename,
                            target_sr=FLAGS.sample_rate,
                            offset=0, duration=duration,
                            min_duration=FLAGS.min_duration,
                            trim=FLAGS.trim_silence)

                        audio_idx = (audio_idx + 1) % len(filenames)
                        if audio_idx == 0:
                            last_request = True

                        batch_audio_samples.append(audio.samples)
                        batch_audio_sample_rates.append(audio.sample_rate)
                        batch_filenames.append(filename)

                    speech_client.recognize(
                        batch_audio_samples,
                        batch_audio_sample_rates,
                        batch_filenames)

        else: # if not synchronous

            for filename in filenames:

                speech_client = SpeechClient(
                  FLAGS.url, protocol, model_name, FLAGS.model_version,
                  FLAGS.batch_size, model_platform=FLAGS.model_platform,
                  verbose=FLAGS.verbose, mode=FLAGS.mode,
                  from_features=from_features_state
                )

                if FLAGS.mode == "streaming":
                    if FLAGS.use_client_featurizer:
                        raise ValueError("Streaming mode not supported with "
                                         "client use_client_featurizer option")
                    else:
                        partial_transcript, partial_wordtimes = speech_client.streaming_recognize(
                            audio_generator_from_file(filename,
                                            target_sr=FLAGS.sample_rate,
                                            chunk_duration=FLAGS.chunk_duration),
                            filename)

                print("\nFile: ", filename)
                #transcripts, scores = nbest_transcript_score;
                ti = 1
                for transcript, score in zip(speech_client.base_transcripts,speech_client.base_scores):
                  print("Final transcript {0}: {1} ".format(ti, transcript + partial_transcript))
                  print("Final score {0}: {1:.5f} ".format(ti, score))
                  ti += 1
                print("\nTimestamps:");
                print("word:start(ms):end(ms)");
                print("---");
                words = speech_client.base_transcripts[0].split();
                for (word, times) in zip(words, speech_client.base_wordtimes):
                    print("{0}:{1}:{2}".format(word, int(times[0]), int(times[1])))
                words = partial_transcript.split();
                for (word, times) in zip(words, partial_wordtimes):
                    print("{0}:{1}:{2}".format(word, int(times[0]), int(times[1])))
                print('---')
                print("Audio processed:", speech_client.audio_processed[0]);

    else:
        raise ValueError("Invalid input method. Valid choices are: {"
                         "}".format(valid_input_methods))
