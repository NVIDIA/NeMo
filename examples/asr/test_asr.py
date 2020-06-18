
import argparse
import asr_lib as asr


# 
# parse arguments
#
parser = argparse.ArgumentParser(description='Interactive chatbot.')

parser.add_argument('--mic', type=int, default=None,
                    help='Input microphone device ID to use to capture audio '
                         '(see scripts/list_microphones.sh for device IDs)')
parser.add_argument('--wav', type=str, default=None, help='Input WAV filename to process')
parser.add_argument('--asr-backend', type=str, default="jarvis-jasper",
                    help='ASR engine backends: ' + asr.backends_str() +
                         ' (default: jarvis-jasper)')
parser.add_argument('--url', type=str, default='localhost:8001',
                    help='TensorRT Inference Server URL. (default: localhost:8001)')
parser.add_argument('--verbose', action="store_true", help='Enable verbose output')
parser.add_argument('--chunk_duration', type=float, required=False,
                        default=.4,
                        help="duration of the audio chunk for streaming "
                                "recognition, in seconds")

args = parser.parse_args()
print(args)


#
# create engines
#
asr_engine = asr.create_engine(
                backend=args.asr_backend,
                wav=args.wav,
                microphone=args.mic,
                url=args.url,
                chunk_duration=args.chunk_duration,
                verbose=args.verbose)

#
# callback that recieves ASR transcript updates (i.e. words spoken by the user)
# 
#   base - previous conversation so far
#   phrase - when a sentence is complete
#   partial - mid-sentence transcript
#
# each TranscriptInfo object has the following members:
#
#   info.transcript (string)
#   info.score      (float)
#   info.word_times (array)
#
# it is possible for a TranscriptInfo to be empty, for example
# the 'phrase' transcript only occurs at the end of a sentence.
# use the info.has_transcript() to see if it currently has one.
#
def on_asr_transcript(base, phrase, partial):
    
    #base.print('Base')
    if partial.has_transcript():
        partial.print('\rPartial', '')

    if phrase.has_transcript():
        phrase.print('\rPhrase')


#
# start processing
#
asr_engine.start(on_asr_transcript)

