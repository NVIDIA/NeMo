
import sys
import numpy as np
import os
import soundfile
import pyaudio as pa
import threading
import math
import time
import glob

from multiprocessing.dummy import Pool as ThreadPool
import multiprocessing

from .speech_utils import AudioSegment


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
class AudioGeneratorFromFile:

    def __init__(self, input_filename, target_sr, chunk_duration, playback_factor, default_capture_state='capture'):
        
        self.sf = soundfile.SoundFile(input_filename, 'rb')
        self.target_sr = target_sr
        self.chunk_duration = chunk_duration
        self.playback_factor = playback_factor
        self.audio_callback = None
        self.recording_state = default_capture_state

    def set_capture_state(self, state):
        self.recording_state = state

    def generate_audio(self, streaming=True):
    
        chunk_size = int(self.chunk_duration * self.sf.samplerate)
        chunk_playback = self.chunk_duration / self.playback_factor
        
        last_time = None
        start = True
        end = False
        
        while not end:

            # read next audio chunk
            audio_signal, end = get_audio_chunk_from_soundfile(self.sf, chunk_size)
            
            # convert to segment format
            audio_segment = AudioSegment(audio_signal, self.sf.samplerate, self.target_sr)

            if self.audio_callback:
                self.audio_callback(audio_segment.samples, audio_segment.sample_rate, self.recording_state)

            # rate-limit to playback factor
            curr_time = time.time()
            
            if last_time is not None and (curr_time - last_time) < chunk_playback:
                time.sleep(chunk_playback - (curr_time - last_time))

            curr_time = time.time()
            
            # check for user exit
            if self.recording_state == "release":
                end = True
            
            # return the samples 
            yield audio_segment.samples, self.target_sr, start, end
            
            start = False
            last_time = curr_time
            
        self.sf.close()
    
class AudioGeneratorFromMicrophone:

    def __init__(self,input_device_id, target_sr, chunk_duration, default_capture_state='capture'):

        self.recording_state = "init"
        self.start           = True
        self.target_sr       = target_sr
        self.chunk_duration  = chunk_duration
        self.audio_callback  = None

        self.default_capture_state   = default_capture_state
        self.mute_max_silence_frames = 10   # ~1 second
        self.mute_silence_frames     = self.mute_max_silence_frames

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


    def set_capture_state(self, state):
        if state == 'mute':
            self.mute_silence_frames = 0
        elif state == 'live' and self.recording_state != 'live':
            self.start = True

        self.recording_state = state

    def generate_audio(self, streaming=True):

        chunk_size = int(self.chunk_duration*self.target_sr)

        print('ASR - chunk_duration = ' + str(self.chunk_duration))
        print('ASR - chunk_size = ' + str(chunk_size))

        #self.recording_state = "init"

        #def keyboard_listener():
            #input("Press Enter to start and end recording...")
            #self.recording_state = "capture"
            #print("Recording...")

            #input("")
            #self.recording_state = "release"

        #listener = threading.Thread(target=keyboard_listener)
        #listener.start()

        # set initial capture state
        self.recording_state = self.default_capture_state
        
        print('ASR - initial capture state:  ' + self.recording_state)

        if self.recording_state == "mute":
            print('ASR - input is muted by default (hold the Push-to-Talk button to speak)')

        #start = True
        stream_initialized = False
        step = 0
        audio_signal = 0
        audio_segment = 0
        end = False

        while not end: #self.recording_state != "release":
            try:
                #if self.recording_state != "mute" or self.mute_silence_frames < self.mute_max_silence_frames: #or self.recording_state == "mute":

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
                    audio_signal = stream.read(chunk_size, exception_on_overflow=False) 
                    audio_signal = np.frombuffer(audio_signal,dtype=np.int16)

                    audio_segment = AudioSegment(audio_signal,
                                                 self.target_sr,
                                                 self.target_sr)

                    if self.audio_callback:
                        self.audio_callback(audio_segment.samples, audio_segment.sample_rate, self.recording_state)

                    emit_samples = True

                    if self.recording_state == "release":
                        end = True
                    elif self.recording_state == "mute":
                        if self.mute_silence_frames < self.mute_max_silence_frames:
                            #audio_signal = np.zeros_like(audio_signal)
                            audio_segment.zero()
                            self.mute_silence_frames = self.mute_silence_frames + 1
                        else:
                            emit_samples = False
                    
                    if emit_samples:
                        yield audio_segment.samples, self.target_sr, self.start, end

                        self.start = False
                        step += 1
            except Exception as e:
                print(e)
                break

        stream.close()
        self.p.terminate()
        print('ASR - shutdown microphone audio device')
        self.recording_state = "destroy"

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
