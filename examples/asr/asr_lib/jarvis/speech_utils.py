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

import librosa
import python_speech_features as psf
import soundfile as sf
import math
from os import system
import numpy as np
from tensorrtserver.api import *
import tensorrtserver.api.model_config_pb2 as model_config
import grpc
import random
from tensorrtserver.api import api_pb2
from tensorrtserver.api import grpc_service_pb2
from tensorrtserver.api import grpc_service_pb2_grpc

WINDOWS_FNS = {"hanning": np.hanning, "hamming": np.hamming, "none": None}


def model_dtype_to_np(model_dtype):
    if model_dtype == model_config.TYPE_BOOL:
        return np.bool
    elif model_dtype == model_config.TYPE_INT8:
        return np.int8
    elif model_dtype == model_config.TYPE_INT16:
        return np.int16
    elif model_dtype == model_config.TYPE_INT32:
        return np.int32
    elif model_dtype == model_config.TYPE_INT64:
        return np.int64
    elif model_dtype == model_config.TYPE_UINT8:
        return np.uint8
    elif model_dtype == model_config.TYPE_UINT16:
        return np.uint16
    elif model_dtype == model_config.TYPE_UINT32:
        return np.uint32
    elif model_dtype == model_config.TYPE_FP16:
        return np.float16
    elif model_dtype == model_config.TYPE_FP32:
        return np.float32
    elif model_dtype == model_config.TYPE_FP64:
        return np.float64
    elif model_dtype == model_config.TYPE_STRING:
        return np.dtype(object)
    return None


class SpeechClient(object):

    def __init__(self, url, protocol, model_name, model_version, batch_size,
                 model_platform=None, verbose=False,
                 mode="batch",
                 from_features=True):

        self.model_name = model_name
        self.model_version = model_version
        self.verbose = verbose
        self.batch_size = batch_size
        self.transpose_audio_features = False
        self.grpc_stub = None
        self.ctx = None
        self.correlation_id = 0
        self.first_run = True
        self.base_transcripts=[]
        self.base_scores=[]
        self.base_wordtimes=None
        self.audio_processed = 0.0
        self.from_features = from_features

        if mode == "streaming":
            self.correlation_id = random.randint(1,2**31-1)

        self.buffer = []

        if mode == "streaming":
            # Create gRPC stub for communicating with the server
            print('opening GRPC channel ' + str(url))
            channel = grpc.insecure_channel(url,
                             options=[('grpc.keepalive_time_ms', 10000),
                                      ('grpc.keepalive_timeout_ms', 10000),
                                      ('grpc.keepalive_permit_without_calls', True)])

            self.grpc_stub = grpc_service_pb2_grpc.GRPCServiceStub(channel)
            request = grpc_service_pb2.StatusRequest(model_name=model_name)
            response = self.grpc_stub.Status(request)
            server_status = response.server_status
        else:
            self.ctx = InferContext(url, protocol, model_name, model_version,
                                    verbose, self.correlation_id, False)
            server_ctx = ServerStatusContext(url, protocol, model_name,
                                             verbose)
            server_status = server_ctx.get_server_status()

        # Make sure the model matches our requirements, and get some
        # properties of the model that we need for preprocessing
        if from_features:
            self.audio_features_name,  \
            self.final_transcripts_name, self.final_transcripts_scores_name, \
            self.final_wordtimes_name, \
            self.partial_transcript_name, \
            self.partial_wordtimes_name, \
            self.audio_features_type, \
            self.final_transcripts_type, self.final_transcripts_scores_type, \
            self.final_wordtimes_type, self.partial_transcript_type, \
            self.partial_wordtimes_type = \
                self.parse_model_from_features(server_status, model_name,
                                               batch_size, model_platform,
                                               verbose)
        else:
            self.audio_signals_name, self.sample_rate_name,\
            self.end_flag_name,\
            self.final_transcripts_name, self.final_transcripts_scores_name, \
            self.final_wordtimes_name, \
            self.partial_transcript_name, \
            self.partial_wordtimes_name, \
            self.audio_processed_name, \
            self.audio_signals_type, self.sample_rate_type, \
            self.end_flag_type, \
            self.final_transcripts_type, self.final_transcripts_scores_type, \
            self.final_wordtimes_type, self.partial_transcript_type, \
            self.partial_wordtimes_type, self.audio_processed_type = \
                self.parse_model(server_status, model_name,
                                     batch_size, model_platform, verbose)


    def postprocess(self, results, labels):

        if self.from_features and len(results) != 5:
            raise Exception("expected 5 result, got {}".format(len(results)))
        if not self.from_features and len(results) != 6:
            raise Exception("expected 6 result, got {}".format(len(results)))

        final_transcript_values = results['FINAL_TRANSCRIPTS']
        final_transcript_scores = results['FINAL_TRANSCRIPTS_SCORE']
        final_wordtimes = results['FINAL_WORDS_START_END']
        partial_transcripts = results['PARTIAL_TRANSCRIPT']
        partial_wordtimes = results['PARTIAL_WORDS_START_END']
        if not self.from_features:
          audio_processed = results['AUDIO_PROCESSED']

        for i in range(0, len(final_transcript_values)):
            partial_transcript = partial_transcripts[i][0]
            filename = labels[i]
            num_transcript = len(final_transcript_values[i])

            print('For batch {0} '.format(i))
            print('File: ', filename)
            ti = 1
            for (transcript, score) in zip(final_transcript_values[i], final_transcript_scores[i]):
                print("Final transcript {0}: {1}".format(ti, transcript.decode("utf-8") + partial_transcript.decode("utf-8")))
                print("Final score {0}: {1:.5f}".format(ti, score))
                ti += 1

            words = final_transcript_values[i][0].decode("utf-8").split();
            timestamps = final_wordtimes[i]
            print("\nTimestamps:");
            print("word:start(ms):end(ms)");
            print("---");
            for (word, times) in zip(words, timestamps):
                print("{0}:{1}:{2}".format(word, int(times[0]), int(times[1])))

            words = partial_transcript.decode("utf-8").split();
            timestamps = partial_wordtimes[i]
            for (word, times) in zip(words, timestamps):
                print("{0}:{1}:{2}".format(word, int(times[0]), int(times[1])))

            print('---')
            if not self.from_features:
              print("Audio processed:", audio_processed[i][0]);

    def postprocess_streaming(self, response, transcript_callback=None):

        if (self.first_run):
            self.first_run = False
            #clear_screen()
            #print("ASR - capturing audio, press Escape or Ctrl+C to "
            #      "stop recording\nTranscript:\n")

        if len(response.meta_data.output) != 6:
            raise Exception("expected 6 result, got {}".format(
                len(response.meta_data.output)))

        for i in range(0, len(response.meta_data.output)):
          if response.meta_data.output[i].name == "FINAL_TRANSCRIPTS":
              final_transcript_id = i
          elif response.meta_data.output[i].name == "FINAL_TRANSCRIPTS_SCORE":
              final_score_id = i
          elif response.meta_data.output[i].name == "FINAL_WORDS_START_END":
              final_wordtimes_id = i
          elif response.meta_data.output[i].name == "PARTIAL_TRANSCRIPT":
              partial_transcript_id = i
          elif response.meta_data.output[i].name == "PARTIAL_WORDS_START_END":
              partial_wordtimes_id = i
          elif response.meta_data.output[i].name == "AUDIO_PROCESSED":
              audio_processed_id = i

        final_transcripts = response.raw_output[final_transcript_id]
        partial_transcripts = response.raw_output[partial_transcript_id]

        final_score = np.frombuffer(response.raw_output[final_score_id],
                            dtype=self.final_transcripts_scores_type)

        final_wordtimes = np.frombuffer(response.raw_output[final_wordtimes_id],
                            dtype=self.final_wordtimes_type)
        final_wordtimes.shape = (len(final_wordtimes)//2, 2)

        partial_wordtimes = np.frombuffer(response.raw_output[partial_wordtimes_id],
                            dtype=self.partial_wordtimes_type)
        partial_wordtimes.shape = (len(partial_wordtimes)//2, 2)

        audio_processed = np.frombuffer(response.raw_output[audio_processed_id],
                            dtype=self.audio_processed_type)

        # Extract final_transcripts
        total_len = len(final_transcripts)
        bytes_used_for_size = 4
        str_index = 0
        ts = 0
        transcripts=[]
        scores=[]
        while ( str_index < total_len ):
            size = int.from_bytes(final_transcripts[str_index:str_index + bytes_used_for_size], byteorder='little')
            str = final_transcripts[str_index + bytes_used_for_size:str_index + bytes_used_for_size + size]
            transcripts.append(str.decode("utf-8"))
            scores.append(final_score[ts])
            ts += 1
            str_index = str_index + size + bytes_used_for_size

        # Store in memory
        if (len(self.base_transcripts) == 0):
            for transcript in transcripts:
                self.base_transcripts.append("")
            for score in scores:
                self.base_scores.append(0.) 
            self.base_wordtimes = np.zeros(final_wordtimes.shape)

        for i, transcript in enumerate(transcripts):
            self.base_transcripts[i] += transcript

        if transcripts[0] != "":
            for i, score in enumerate(scores):
                self.base_scores[i] += score

        self.base_wordtimes = np.concatenate((self.base_wordtimes, final_wordtimes))
        self.audio_processed = audio_processed

        size = int.from_bytes(partial_transcripts[0:bytes_used_for_size], byteorder='little')
        partial_transcript = partial_transcripts[bytes_used_for_size:bytes_used_for_size + size]

        #print(partial_transcript)
        #print(self.base_transcripts)
        #print("base_scores")
        #print(self.base_scores)
        #print("base_wordtimes")
        #print(self.base_wordtimes)

        #print(transcripts)
        #print(scores)
        #print(final_wordtimes)
        #print(partial_transcript)
        #print(partial_wordtimes)

        partial_transcript = partial_transcript.decode("utf-8")
        #print("first_partial_transcript=")
        #print(first_partial_transcript)
        #print("first_final_transcript=")
        #print(first_final_transcript)

        #print(self.base_transcripts[0] + partial_transcript)

        if transcript_callback is not None:
            class TranscriptInfo:
                def __init__(self, transcript, score, word_times):
                    self.transcript = transcript
                    self.score = score
                    self.word_times = word_times

                def has_transcript(self):
                    if self.transcript != "":
                        return True
                    else:
                        return False

                def print(self, name='Transcript Info', end= '\n'):
                    print('{:s}: {:s} ({:f})'.format(name, self.transcript, self.score), end = end, flush = True)
                    #print(name)
                    #print('   transcript: ' + self.transcript)
                    #if self.scores is not None:
                        #print('   scores:     {:f}'.format(self.scores))
                        ##print('   scores:     ' + str(self.scores))
                    #print('   word_times: ' + str(self.word_times))

            transcript_callback( TranscriptInfo(self.base_transcripts[0], self.base_scores[0], self.base_wordtimes),
                                 TranscriptInfo(transcripts[0], scores[0], final_wordtimes),
                                 TranscriptInfo(partial_transcript, 1.0, partial_wordtimes) )

        else:
            out_str = "\033[5;0H" + self.base_transcripts[0] + partial_transcript
            print(out_str, end='\r', flush=True)

        #print("out_str")
        #print(out_str)
        #print("final_score")
        #print(final_score)
        #print("final_wordtimes")
        #print(final_wordtimes)
        #print("partial_transcript")
        #print(partial_transcript)
        #print("partial_wordtimes")
        #print(partial_wordtimes)

        return partial_transcript, partial_wordtimes

    def parse_model(self, server_status,
                    model_name, batch_size,
                    model_platform=None, verbose=False):
        """
        Check the configuration of the ensemble model
        """

        print("Server status:")
        print(server_status)

        if model_name not in server_status.model_status:
            raise Exception("unable to get status for '" + model_name + "'")

        status = server_status.model_status[model_name]
        config = status.config

        self.model_platform = model_platform

        # Inputs are:
        #   1) audio_signal: raw audio samples [num_samples]
        #   2) sample_rate: sample rate of audio
        #   3) end_flag: set to 1 if end of streaming sequence

        if len(config.input) < 2:
            raise Exception(
                "expecting 2 to 3 inputs, got {}".format(len(config.input)))

        # Outputs are:
        #   1) final_transcripts:        candidate transcripts
        #   2) final_transcripts_score: score of each transcript
        #   3) final_wordtimes:      start and end times of words
        #   4) partial_transcript:   partial transcript 
        #   5) partial_wordtimes:    start and end times of words
        #   6) audio_processed:    audio processed so far

        if len(config.output) != 6:
            raise Exception(
                "expecting 6 outputs, got {}".format(len(config.output)))

        audio_signal = config.input[0]
        sample_rate = config.input[1]
        end_flag    = config.input[1]
        if len(config.input) > 2:
            end_flag = config.input[2]

        final_transcripts = config.output[0]
        final_transcripts_scores = config.output[1]
        final_wordtimes = config.output[2]
        partial_transcript = config.output[3]
        partial_wordtimes = config.output[4]
        audio_processed = config.output[5]

        expected_audio_signal_dim = 1
        expected_audio_signal_type = model_config.TYPE_FP32

        if audio_signal.data_type != expected_audio_signal_type:
            raise Exception("expecting audio_signal datatype to be " +
                            model_config.DataType.Name(
                                expected_audio_signal_type) +
                            "model '" + model_name + "' output type is " +
                            model_config.DataType.Name(audio_signal.data_type))

        if sample_rate.data_type != model_config.TYPE_UINT32:
            raise Exception(
                "expecting sample_rate datatype to be TYPE_UINT32, "
                "model '" + model_name + "' output type is " +
                model_config.DataType.Name(sample_rate.data_type))


        # Model specifying maximum batch size of 0 indicates that batching
        # is not supported and so the input tensors do not expect an "N"
        # dimension (and 'batch_size' should be 1 so that only a single
        # image instance is inferred at a time).
        max_batch_size = config.max_batch_size
        if max_batch_size == 0:
            if batch_size != 1:
                raise Exception(
                    "batching not supported for model '" + model_name + "'")
        else:  # max_batch_size > 0
            if batch_size > max_batch_size:
                raise Exception(
                    "expecting batch size <= {} for model {}".format(
                        max_batch_size, model_name))

        if len(audio_signal.dims) != expected_audio_signal_dim:
            raise Exception("Expecting audio signal to have {} dimensions, "
                            "model '{}' audio_signal has {}".format(
                expected_audio_signal_dim,
                model_name,
                len(audio_signal.dims)))

        if len(sample_rate.dims) != 1:
            raise Exception("Expecting sample_rate to have 1 dimension, "
                            "model '{}' sample_rate has {}".format(
                model_name,len(sample_rate.dims)))


        if len(final_transcripts.dims) != 1:
            raise Exception("Expecting final_transcripts to have 1 dimension, "
                            "model '{}' final_transcripts has {}".format(
                model_name,len(final_transcripts.dims)))

        if len(partial_transcript.dims) != 1:
            raise Exception("Expecting partial_transcript to have 1 dimension, "
                            "model '{}' partial_transcript has {}".format(
                model_name,len(partial_transcript.dims)))

        if len(final_transcripts_scores.dims) != 1:
            raise Exception(
                "Expecting transcripts_scores to have 1 dimension, "
                "model '{}' transcripts_scores has {}".format(
                    model_name,len(final_transcripts_scores.dims)))

        if len(final_wordtimes.dims) != 2:
            raise Exception(
                "Expecting final_wordtimes to have 2 dimension, "
                "model '{}' final_wordtimes has {}".format(
                    model_name,len(final_wordtimes.dims)))

        if len(partial_wordtimes.dims) != 2:
            raise Exception(
                "Expecting partial_wordtimes to have 2 dimension, "
                "model '{}' partial_wordtimes has {}".format(
                    model_name,len(partial_wordtimes.dims)))

        if len(audio_processed.dims) != 1:
            raise Exception(
                "Expecting audio_processed to have 1 dimension, "
                "model '{}' audio_processed has {}".format(
                    model_name,len(audio_processed.dims)))

        return (audio_signal.name, sample_rate.name, end_flag.name,
                final_transcripts.name, final_transcripts_scores.name, 
                final_wordtimes.name, partial_transcript.name,
                partial_wordtimes.name, audio_processed.name,
                model_dtype_to_np(audio_signal.data_type),
                model_dtype_to_np(sample_rate.data_type),
                model_dtype_to_np(end_flag.data_type),
                model_dtype_to_np(final_transcripts.data_type),
                model_dtype_to_np(final_transcripts_scores.data_type),
                model_dtype_to_np(final_wordtimes.data_type),
                model_dtype_to_np(partial_transcript.data_type),
                model_dtype_to_np(partial_wordtimes.data_type),
                model_dtype_to_np(audio_processed.data_type)
                )

    def parse_model_from_features(self, server_status,
                                  model_name, batch_size,
                                  model_platform=None, verbose=False):
        """
        Check the configuration of the ensemble model
        """
        print("Server status:")
        print(server_status)

        if model_name not in server_status.model_status:
            raise Exception("unable to get status for '" + model_name + "'")

        status = server_status.model_status[model_name]
        config = status.config

        self.model_platform = model_platform

        # Inputs are:
        #   1) audio_features: set of audio features [num_time_steps,
        #   num_features]
        #   2) num_time_steps: length of audio features

        if len(config.input) != 1:
            raise Exception(
                "expecting 1 input, got {}".format(len(config.input)))

        # Outputs are:
        #   1) transcripts:        candidate transcripts
        #   2) transcripts_scores: score of each transcript
        #   3) wordtimes: start and end times of words
        #   4) partial_transcript:   partial transcript 
        #   5) partial_wordtimes:    start and end times of words

        if len(config.output) != 5:
            raise Exception(
                "expecting 5 output, got {}".format(len(config.output)))

        audio_features = config.input[0]

        final_transcripts = config.output[0]
        final_transcripts_scores = config.output[1]
        final_wordtimes = config.output[2]
        partial_transcript = config.output[3]
        partial_wordtimes = config.output[4]

        expected_audio_features_dim = 2
        expected_audio_features_type = model_config.TYPE_FP32

        if audio_features.data_type != expected_audio_features_type:
            raise Exception("expecting audio_features datatype to be " +
                            model_config.DataType.Name(
                                expected_audio_features_type) +
                            "model '" + model_name + "' output type is " +
                            model_config.DataType.Name(
                                audio_features.data_type))

        max_batch_size = config.max_batch_size
        if max_batch_size == 0:
            if batch_size != 1:
                raise Exception(
                    "batching not supported for model '" + model_name + "'")
        else:  # max_batch_size > 0
            if batch_size > max_batch_size:
                raise Exception(
                    "expecting batch size <= {} for model {}".format(
                        max_batch_size, model_name))

        if len(audio_features.dims) != expected_audio_features_dim:
            raise Exception("Expecting audio features to have {} dimensions, "
                            "model '{}' audio_features has {}".format(
                expected_audio_features_dim,
                model_name,
                len(audio_features.dims)))

        if len(final_transcripts.dims) != 1:
            raise Exception("Expecting final_transcripts to have 1 dimension, "
                            "model '{}' final_transcripts has {}".format(
                model_name,len(final_transcripts.dims)))

        if len(partial_transcript.dims) != 1:
            raise Exception("Expecting partial_transcript to have 1 dimension, "
                            "model '{}' partial_transcript has {}".format(
                model_name,len(partial_transcript.dims)))

        if len(final_transcripts_scores.dims) != 1:
            raise Exception(
                "Expecting transcripts_scores to have 1 dimension, "
                "model '{}' transcripts_scores has {}".format(
                    model_name,len(final_transcripts_scores.dims)))

        if len(final_wordtimes.dims) != 2:
            raise Exception(
                "Expecting final_wordtimes to have 2 dimension, "
                "model '{}' final_wordtimes has {}".format(
                    model_name,len(final_wordtimes.dims)))

        if len(partial_wordtimes.dims) != 2:
            raise Exception(
                "Expecting partial_wordtimes to have 2 dimension, "
                "model '{}' partial_wordtimes has {}".format(
                    model_name,len(partial_wordtimes.dims)))

        return (audio_features.name,
                final_transcripts.name, final_transcripts_scores.name, 
                final_wordtimes.name, partial_transcript.name,
                partial_wordtimes.name,
                model_dtype_to_np(audio_features.data_type),
                model_dtype_to_np(final_transcripts.data_type),
                model_dtype_to_np(final_transcripts_scores.data_type),
                model_dtype_to_np(final_wordtimes.data_type),
                model_dtype_to_np(partial_transcript.data_type),
                model_dtype_to_np(partial_wordtimes.data_type),
                )

    def update_audio_request(self, request, audio_generator):

        for audio_signal, sample_rate, start, end in audio_generator:
            # Delete the current inputs

            input_batch = [audio_signal.astype(self.audio_signals_type)]

            end_flag_batch = end
            end_flag_batch = [np.asarray([end_flag_batch],
                              dtype=self.end_flag_type)]

            sample_rates_batch = sample_rate
            sample_rates_batch = [np.asarray([sample_rates_batch],
                                             dtype=self.sample_rate_type)]

            flags = InferRequestHeader.FLAG_NONE
            input_batch[0] = np.expand_dims(input_batch[0], axis=0)

            audio_bytes = input_batch[0].tobytes()
            sample_rates_bytes = sample_rates_batch[0].tobytes()
            end_flag_bytes = end_flag_batch[0].tobytes()

            request.meta_data.input[0].dims[0] = audio_signal.shape[0]
            request.meta_data.input[0].batch_byte_size = len(audio_bytes)

            request.meta_data.input[1].dims[0] = 1
            request.meta_data.input[1].batch_byte_size = len(
                sample_rates_bytes)

            request.meta_data.input[2].dims[0] = 1
            request.meta_data.input[2].batch_byte_size = len(end_flag_bytes)

            if start:
                flags = flags | InferRequestHeader.FLAG_SEQUENCE_START

            if end:
                flags = flags | InferRequestHeader.FLAG_SEQUENCE_END

            request.meta_data.flags = flags;
#
#            # Write bytes to file
#            try:
#              if start:
#                audio_signal_file = open("AUDIO_SIGNAL","wb")
#                audio_signal_file.write(audio_bytes)
#                audio_signal_file.close()
#
#                sample_rates_file = open("SAMPLE_RATE","wb")
#                sample_rates_file.write(sample_rates_bytes)
#                sample_rates_file.close()
#
#                end_flag_file = open("END_FLAG","wb")
#                end_flag_file.write(end_flag_bytes)
#                end_flag_file.close()
#            except Exception as e:
#              print(str(e))
#
            # Send request with audio signal
            del request.raw_input[:]
            request.raw_input.extend([audio_bytes])
            request.raw_input.extend([sample_rates_bytes])
            request.raw_input.extend([end_flag_bytes])

            yield request

#        print('ASR - audio generator has finished')



    def recognize(self, audio_signal, sample_rate, filenames):
        # Send requests of FLAGS.batch_size audio signals. If the number of
        # audios isn't an exact multiple of FLAGS.batch_size then just
        # start over with the first audio until the batch is filled.

        flags = InferRequestHeader.FLAG_NONE
        flags = flags | InferRequestHeader.FLAG_SEQUENCE_START

        input_batch = []
        input_sample_rates = []
        input_filenames = []
        max_num_samples_batch = 30720

        for idx in range(self.batch_size):
            input_batch.append(audio_signal[idx].astype(
                self.audio_signals_type))
            input_sample_rates.append(
                np.asarray([sample_rate[idx]],
                           dtype=self.sample_rate_type))
            input_filenames.append(filenames[idx])
            num_samples = audio_signal[idx].shape[0]

            if (num_samples > max_num_samples_batch):
                max_num_samples_batch = num_samples

        for idx in range(self.batch_size):
            num_samples = input_batch[idx].shape[0]

            mean = np.mean(input_batch[idx])
            std_var = np.std(input_batch[idx])
            gauss_noise = np.random.normal(
                mean,std_var,
                max_num_samples_batch-num_samples)

            input_batch[idx]= np.concatenate(
                (input_batch[idx], gauss_noise.astype(
                    self.audio_signals_type)))


#
#       # Write bytes to file
#        try:
#          audio_signal_file = open("AUDIO_SIGNAL","wb")
#          audio_signal_file.write(input_batch[0].tobytes())
#          audio_signal_file.close()
#
#          sample_rates_file = open("SAMPLE_RATE","wb")
#          sample_rates_file.write(input_sample_rates[0].tobytes())
#          sample_rates_file.close()
#        except Exception as e:
#          print(str(e))
#

        # Send request
        print("Sending request to transcribe file(s):", ",".join(
            input_filenames))

        result = self.ctx.run(
            {self.audio_signals_name: input_batch,
             self.sample_rate_name: input_sample_rates},
            {self.final_transcripts_name: InferContext.ResultFormat.RAW,
             self.final_transcripts_scores_name: InferContext.ResultFormat.RAW,
             self.final_wordtimes_name: InferContext.ResultFormat.RAW,
             self.partial_transcript_name: InferContext.ResultFormat.RAW,
             self.partial_wordtimes_name: InferContext.ResultFormat.RAW,
             self.audio_processed_name: InferContext.ResultFormat.RAW,
             },
            self.batch_size, flags)

        

        self.postprocess(result, input_filenames)

        return

    def recognize_from_features(self, audio_features, filenames):

        # Send requests of FLAGS.batch_size audio signals. If the number of
        # audios isn't an exact multiple of FLAGS.batch_size then just
        # start over with the first audio until the batch is filled.

        flags = InferRequestHeader.FLAG_NONE
        flags = flags | InferRequestHeader.FLAG_SEQUENCE_START

        results = []
        result_filenames = []
        request_ids = []
        audio_idx = 0
        last_request = False
        while not last_request:
            input_batch = []
            input_filenames = []
            max_time_steps_batch = 0
            for idx in range(self.batch_size):
                input_batch.append(audio_features[audio_idx].astype(
                    self.audio_features_type))
                input_filenames.append(filenames[audio_idx])
                num_time_steps = audio_features[audio_idx].shape[0]

                audio_idx = (audio_idx + 1) % len(audio_features)
                if audio_idx == 0:
                    last_request = True

                if (num_time_steps > max_time_steps_batch):
                    max_time_steps_batch = num_time_steps

            result_filenames.append(input_filenames)

            for idx in range(self.batch_size):
                num_time_steps = input_batch[idx].shape[0]
                input_batch[idx] = np.pad(input_batch[idx],
                                          ((0,
                                            max_time_steps_batch -
                                            num_time_steps),
                                           (0, 0)), mode='constant')

            #input_batch[0] = np.expand_dims(input_batch[0], axis=0)


            # Send request
            print("Sending request to transcribe file(s):", ",".join(
                input_filenames))
            result = self.ctx.run(
                {self.audio_features_name: input_batch},
                {self.final_transcripts_name: InferContext.ResultFormat.RAW,
                 self.final_transcripts_scores_name: InferContext.ResultFormat.RAW,
                 self.final_wordtimes_name: InferContext.ResultFormat.RAW,
                 self.partial_transcript_name: InferContext.ResultFormat.RAW,
                 self.partial_wordtimes_name: InferContext.ResultFormat.RAW
                 },
                self.batch_size, flags)

            self.postprocess(result, input_filenames)

            results.append(result)

        # return results, result_filenames
        return

    def streaming_recognize(self, audio_generator, filename, transcript_callback=None):

        request = grpc_service_pb2.InferRequest()
        request.model_name = self.model_name
        if self.model_version is None:
            request.model_version = -1
        else:
            request.model_version = self.model_version
        request.meta_data.batch_size = self.batch_size
        request.meta_data.correlation_id = self.correlation_id

        # Prepare outputs
        output_final_transcript = api_pb2.InferRequestHeader.Output()
        output_final_transcript.name = self.final_transcripts_name
        request.meta_data.output.extend([output_final_transcript])

        output_final_transcript_score = api_pb2.InferRequestHeader.Output()
        output_final_transcript_score.name = self.final_transcripts_scores_name
        request.meta_data.output.extend([output_final_transcript_score])

        output_final_wordtimes = api_pb2.InferRequestHeader.Output()
        output_final_wordtimes.name = self.final_wordtimes_name
        request.meta_data.output.extend([output_final_wordtimes])

        output_partial_transcript = api_pb2.InferRequestHeader.Output()
        output_partial_transcript.name = self.partial_transcript_name
        request.meta_data.output.extend([output_partial_transcript])

        output_partial_wordtimes = api_pb2.InferRequestHeader.Output()
        output_partial_wordtimes.name = self.partial_wordtimes_name
        request.meta_data.output.extend([output_partial_wordtimes])

        output_audio_processed = api_pb2.InferRequestHeader.Output()
        output_audio_processed.name = self.audio_processed_name
        request.meta_data.output.extend([output_audio_processed])

        input_audio_signals = api_pb2.InferRequestHeader.Input()
        input_audio_signals.name = self.audio_signals_name
        # These will need to be set at every inference call
        input_audio_signals.dims.extend([1])
        input_audio_signals.batch_byte_size = 0
        request.meta_data.input.extend([input_audio_signals])

        input_sample_rates = api_pb2.InferRequestHeader.Input()
        input_sample_rates.name = self.sample_rate_name
        # These will need to be set at every inference call
        input_sample_rates.dims.extend([1])
        input_sample_rates.batch_byte_size = 0
        request.meta_data.input.extend([input_sample_rates])

        input_end_flag = api_pb2.InferRequestHeader.Input()
        input_end_flag.name = self.end_flag_name
        # These will need to be set at every inference call
        input_end_flag.dims.extend([1])
        input_end_flag.batch_byte_size = 0
        request.meta_data.input.extend([input_end_flag])

        requestGenerator = self.update_audio_request(request, audio_generator.generate_audio())
        responses = self.grpc_stub.StreamInfer(requestGenerator)


        #while (1):
        #    try:
        #        responses.next()
        #    except:
        #        time.sleep(0.1)

        for response in responses:
            partial_transcript, partial_wordtimes = \
            self.postprocess_streaming(response, transcript_callback)
            #if audio_generator.recording_state == "destroy":
            #    break

        print('ASR - engine has been shutdown')
        return partial_transcript, partial_wordtimes

def preemphasis(signal, coeff=0.97):
    return np.append(signal[0], signal[1:] - coeff * signal[:-1])


def normalize_signal(signal, gain=None):
    """
    Normalize float32 signal to [-1, 1] range
    """
    if gain is None or gain < 0.:
        gain = 1.0 / (np.max(np.abs(signal)) + 1e-5)
    return signal * gain


def get_speech_features(signal, sample_freq, params):
    """
    Get speech features using either librosa (recommended) or
    python_speech_features
    Args:
      signal (np.array): np.array containing raw audio signal
      sample_freq (float): sample rate of the signal
      params (dict): parameters of pre-processing
    Returns:
      np.array: np.array of audio features with shape=[num_time_steps,
      num_features].
      audio_duration (float): duration of the signal in seconds
    """

    backend = params.get('backend', 'psf')

    features_type = params.get('input_type', 'spectrogram')
    num_features = params['num_audio_features']
    window_size = params.get('window_size', 20e-3)
    window_stride = params.get('window_stride', 10e-3)
    augmentation = params.get('augmentation', None)

    if backend == 'librosa':
        window_fn = WINDOWS_FNS[params.get('window', "hanning")]
        dither = params.get('dither', 0.0)
        num_fft = params.get('num_fft', None)
        norm_per_feature = params.get('norm_per_feature', False)
        mel_basis = params.get('mel_basis', None)
        gain = params.get('gain')
        mean = params.get('features_mean')
        std_dev = params.get('features_std_dev')
        if mel_basis is not None and sample_freq != params["sample_freq"]:
            raise ValueError(
                ("The sampling frequency set in params {} does not match the "
                 "frequency {} read from file ").format(
                    params["sample_freq"],
                    sample_freq)
            )
        features, duration = get_speech_features_librosa(
            signal, sample_freq, num_features, features_type,
            window_size, window_stride, augmentation, window_fn=window_fn,
            dither=dither, norm_per_feature=norm_per_feature, num_fft=num_fft,
            mel_basis=mel_basis, gain=gain, mean=mean, std_dev=std_dev
        )
    else:
        pad_to = params.get('pad_to', 8)
        features, duration = get_speech_features_psf(
            signal, sample_freq, num_features, pad_to, features_type,
            window_size, window_stride, augmentation
        )
    features = np.transpose(features)
    return features, duration


def get_speech_features_librosa(signal, sample_freq, num_features,
                                features_type='spectrogram',
                                window_size=20e-3,
                                window_stride=10e-3,
                                augmentation=None,
                                window_fn=np.hanning,
                                num_fft=None,
                                dither=0.0,
                                norm_per_feature=False,
                                mel_basis=None,
                                gain=None,
                                mean=None,
                                std_dev=None):
    """Function to convert raw audio signal to numpy array of features.
    Backend: librosa
    Args:
      signal (np.array): np.array containing raw audio signal.
      sample_freq (float): frames per second.
      num_features (int): number of speech features in frequency domain.
      features_type (string): 'mfcc' or 'spectrogram'.
      window_size (float): size of analysis window in milli-seconds.
      window_stride (float): stride of analysis window in milli-seconds.
      augmentation (dict, optional): dictionary of augmentation parameters. See
          :func:`augment_audio_signal` for specification and example.

    Returns:
      np.array: np.array of audio features with shape=[num_time_steps,
      num_features].
      audio_duration (float): duration of the signal in seconds
    """
    signal = normalize_signal(signal.astype(np.float32), gain)
    #if augmentation:
    #    signal = augment_audio_signal(signal, sample_freq, augmentation)

    audio_duration = len(signal) * 1.0 / sample_freq

    n_window_size = int(sample_freq * window_size)
    n_window_stride = int(sample_freq * window_stride)
    num_fft = num_fft or 2 ** math.ceil(math.log2(window_size * sample_freq))

    if dither > 0:
        signal += dither * np.random.randn(*signal.shape)

    if features_type == 'spectrogram':
        # ignore 1/n_fft multiplier, since there is a post-normalization
        powspec = np.square(np.abs(librosa.core.stft(
            signal, n_fft=n_window_size,
            hop_length=n_window_stride, win_length=n_window_size, center=True,
            window=window_fn)))
        # remove small bins
        powspec[powspec <= 1e-30] = 1e-30
        features = 10 * np.log10(powspec.T)

        assert num_features <= n_window_size // 2 + 1, \
            "num_features for spectrogram should be <= (sample_freq * " \
            "window_size // 2 + 1)"

        # cut high frequency part
        features = features[:, :num_features]

    elif features_type == 'mfcc':
        signal = preemphasis(signal, coeff=0.97)
        S = np.square(
            np.abs(
                librosa.core.stft(signal, n_fft=num_fft,
                                  hop_length=int(window_stride * sample_freq),
                                  win_length=int(window_size * sample_freq),
                                  center=True, window=window_fn
                                  )
            )
        )
        features = librosa.feature.mfcc(sr=sample_freq, S=S,
                                        n_mfcc=num_features,
                                        n_mels=2 * num_features).T
    elif features_type == 'logfbank':
        signal = preemphasis(signal, coeff=0.97)
        S = np.abs(librosa.core.stft(signal, n_fft=num_fft,
                                     hop_length=int(
                                         window_stride * sample_freq),
                                     win_length=int(window_size * sample_freq),
                                     center=True, window=window_fn)) ** 2.0
        if mel_basis is None:
            # Build a Mel filter
            mel_basis = librosa.filters.mel(sample_freq, num_fft,
                                            n_mels=num_features,
                                            fmin=0, fmax=int(sample_freq / 2))
        features = np.log(np.dot(mel_basis, S) + 1e-20).T
    else:
        raise ValueError('Unknown features type: {}'.format(features_type))

    norm_axis = 0 if norm_per_feature else None
    if mean is None:
        mean = np.mean(features, axis=norm_axis)
    if std_dev is None:
        std_dev = np.std(features, axis=norm_axis)

    features = (features - mean) / std_dev
    #np.savetxt("normalized_features_librosa.dat",features)

    # now it is safe to pad
    # if pad_to > 0:
    #   if features.shape[0] % pad_to != 0:
    #     pad_size = pad_to - features.shape[0] % pad_to
    #     if pad_size != 0:
    #         features = np.pad(features, ((0,pad_size), (0,0)),
    #         mode='constant')
    return features, audio_duration


def get_speech_features_psf(signal, sample_freq, num_features,
                            pad_to=8,
                            features_type='spectrogram',
                            window_size=20e-3,
                            window_stride=10e-3,
                            augmentation=None):
    """Function to convert raw audio signal to numpy array of features.
    Backend: python_speech_features
    Args:
      signal (np.array): np.array containing raw audio signal.
      sample_freq (float): frames per second.
      num_features (int): number of speech features in frequency domain.
      pad_to (int): if specified, the length will be padded to become divisible
          by ``pad_to`` parameter.
      features_type (string): 'mfcc' or 'spectrogram'.
      window_size (float): size of analysis window in milli-seconds.
      window_stride (float): stride of analysis window in milli-seconds.
      augmentation (dict, optional): dictionary of augmentation parameters. See
          :func:`augment_audio_signal` for specification and example.
    Returns:
      np.array: np.array of audio features with shape=[num_time_steps,
      num_features].
      audio_duration (float): duration of the signal in seconds
    """
    #if augmentation is not None:
    #    signal = augment_audio_signal(signal, sample_freq, augmentation)
    #else:
    signal = (
            normalize_signal(signal.astype(np.float32)) * 32767.0).astype(
        np.int16)

    audio_duration = len(signal) * 1.0 / sample_freq

    n_window_size = int(sample_freq * window_size)
    n_window_stride = int(sample_freq * window_stride)

    # making sure length of the audio is divisible by 8 (fp16 optimization)
    length = 1 + int(math.ceil(
        (1.0 * signal.shape[0] - n_window_size) / n_window_stride
    ))
    if pad_to > 0:
        if length % pad_to != 0:
            pad_size = (pad_to - length % pad_to) * n_window_stride
            signal = np.pad(signal, (0, pad_size), mode='constant')

    if features_type == 'spectrogram':
        frames = psf.sigproc.framesig(sig=signal,
                                      frame_len=n_window_size,
                                      frame_step=n_window_stride,
                                      winfunc=np.hanning)

        # features = np.log1p(psf.sigproc.powspec(frames, NFFT=N_window_size))
        features = psf.sigproc.logpowspec(frames, NFFT=n_window_size)
        assert num_features <= n_window_size // 2 + 1, \
            "num_features for spectrogram should be <= (sample_freq * window_size // 2 + 1)"

        # cut high frequency part
        features = features[:, :num_features]

    elif features_type == 'mfcc':
        features = psf.mfcc(signal=signal,
                            samplerate=sample_freq,
                            winlen=window_size,
                            winstep=window_stride,
                            numcep=num_features,
                            nfilt=2 * num_features,
                            nfft=512,
                            lowfreq=0, highfreq=None,
                            preemph=0.97,
                            ceplifter=2 * num_features,
                            appendEnergy=False)

    elif features_type == 'logfbank':
        features = psf.logfbank(signal=signal,
                                samplerate=sample_freq,
                                winlen=window_size,
                                winstep=window_stride,
                                nfilt=num_features,
                                nfft=512,
                                lowfreq=0, highfreq=sample_freq / 2,
                                preemph=0.97)
    else:
        raise ValueError('Unknown features type: {}'.format(features_type))

    if pad_to > 0:
        assert features.shape[0] % pad_to == 0
    mean = np.mean(features)
    std_dev = np.std(features)
    features = (features - mean) / std_dev

    return features, audio_duration


class AudioSegment(object):
    """Monaural audio segment abstraction.
    :param samples: Audio samples [num_samples x num_channels].
    :type samples: ndarray.float32
    :param sample_rate: Audio sample rate.
    :type sample_rate: int
    :raises TypeError: If the sample data type is not float or int.
    """

    def __init__(self, samples, sample_rate, target_sr=16000, trim=False,
                 trim_db=60):
        """Create audio segment from samples.
        Samples are convert float32 internally, with int scaled to [-1, 1].
        """
        samples = self._convert_samples_to_float32(samples)
        if target_sr is not None and target_sr != sample_rate:
            samples = librosa.core.resample(samples, sample_rate, target_sr)
            sample_rate = target_sr
        if trim:
            samples, _ = librosa.effects.trim(samples, trim_db)
        self._samples = samples
        self._sample_rate = sample_rate
        if self._samples.ndim >= 2:
            self._samples = np.mean(self._samples, 1)

    def zero(self):
        self._samples = np.zeros_like(self._samples)

    @staticmethod
    def _convert_samples_to_float32(samples):
        """Convert sample type to float32.
        Audio sample type is usually integer or float-point.
        Integers will be scaled to [-1, 1] in float32.
        """
        float32_samples = samples.astype('float32')
        if samples.dtype in np.sctypes['int']:
            bits = np.iinfo(samples.dtype).bits
            float32_samples *= (1. / ((2 ** (bits - 1)) - 1))
        elif samples.dtype in np.sctypes['float']:
            pass
        else:
            raise TypeError("Unsupported sample type: %s." % samples.dtype)
        return float32_samples

    @classmethod
    def from_file(cls, filename, target_sr=16000, offset=0, duration=0,
                 min_duration=0, trim=False):
        """
        Load a file supported by librosa and return as an AudioSegment.
        :param filename: path of file to load
        :param target_sr: the desired sample rate
        :param int_values: if true, load samples as 32-bit integers
        :param offset: offset in seconds when loading audio
        :param duration: duration in seconds when loading audio
        :return: numpy array of samples
        """
        with sf.SoundFile(filename, 'r') as f:
            dtype_options = {'PCM_16': 'int16', 'PCM_32': 'int32', 'FLOAT': 'float32'}
            dtype_file = f.subtype
            if dtype_file in dtype_options:
                dtype = dtype_options[dtype_file]
            else:
                dtype = 'float32'
            sample_rate = f.samplerate
            if offset > 0:
                f.seek(int(offset * sample_rate))
            if duration > 0:
                samples = f.read(int(duration * sample_rate), dtype=dtype)
            else:
                samples = f.read(dtype=dtype)

        num_zero_pad = int(target_sr * min_duration - samples.shape[0])
        if num_zero_pad > 0:
            samples = np.pad(samples, [0, num_zero_pad], mode='constant')

        samples = samples.transpose()
        return cls(samples, sample_rate, target_sr=target_sr, trim=trim)

    @property
    def samples(self):
        return self._samples.copy()

    @property
    def sample_rate(self):
        return self._sample_rate

# define our clear function
def clear_screen():
    _ = system('clear')
