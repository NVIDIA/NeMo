import pyaudio
import wave
import numpy as np

p = pyaudio.PyAudio()

# print input device info
info = p.get_host_api_info_by_index(0)
numdevices = info.get('deviceCount')
for i in range(0, numdevices):
    dev_info = p.get_device_info_by_host_api_device_index(0, i)
    if (dev_info.get('maxInputChannels')) > 0:
        print("Input Device ID {:d} - {:s} (inputs={:.0f}) (sample_rate={:.0f})".format(i,
              dev_info.get('name'), dev_info.get('maxInputChannels'), dev_info.get('defaultSampleRate')))

