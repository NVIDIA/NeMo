
from .speech_utils import AudioSegment, SpeechClient, get_speech_features
from .audio_utils import AudioGeneratorFromMicrophone, AudioGeneratorFromFile


class JarvisAsrEngine():
    def __init__(self, wav, microphone, url="localhost:8001", 
                 model="jasper-asr-trt-ensemble-vad-streaming",
                 default_capture_state="capture",
                 chunk_duration = .4,
                 verbose=True): 
                 
        if (wav is None and microphone is None) or (wav is not None and microphone is not None):
            raise ValueError('ASR - must provide either wav filename or microphone device ID')

        self.sample_rate = 16000
        self.chunk_duration = chunk_duration

        self.speech_client = SpeechClient(
            url=url, 
            protocol=None, #ProtocolType.from_str("gRPC"),
            model_name=model, 
            model_version=None,
            batch_size=1,
            model_platform="trt",
            verbose=verbose,
            mode="streaming",
            from_features=False)

        if microphone is not None:
            self.audio_generator = AudioGeneratorFromMicrophone(
                input_device_id=microphone,
                target_sr=self.sample_rate,
                chunk_duration=self.chunk_duration,
                default_capture_state=default_capture_state)
        else:
            self.audio_generator = AudioGeneratorFromFile(
                input_filename=wav,
                target_sr=self.sample_rate,
                chunk_duration=self.chunk_duration,
                playback_factor=1000,
                default_capture_state=default_capture_state)
    
    @property
    def capture_state(self):
        return self.audio_generator.recording_state

    def set_capture_state(self, state):
        # valid values for state string:
        #   'lvie'    -> active recording
        #   'mute'    -> temporarily mute
        #   'release' -> shutdown
        if state != 'live':
            print("ASR - setting capture state '{:s}'".format(state))
        self.audio_generator.set_capture_state(state)
            
    def start(self, transcript_callback=None, audio_callback=None):
        self.audio_generator.audio_callback = audio_callback
        return self.speech_client.streaming_recognize(
            self.audio_generator, "mic", transcript_callback)

    def stop(self):
        print('ASR - stopping ASR engine')
        self.set_capture_state('release')

