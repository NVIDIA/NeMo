# NeMo Voice Agent

A [Pipecat](https://github.com/pipecat-ai/pipecat) example demonstrating the simplest way to create a voice agent using NVIDIA NeMo STT/TTS service and HuggingFace LLM. Everthing is deployed locally so you can have your own voice agent. 



## ✨ Key Features

- Open-source, local deployment, and flexible customization.
- Talk to most LLMs from HuggingFace, use different prompts to configure the agent. 
- Speaker diarization up to 4 speakers.
- Streaming speech recognition.
- FastPitch-HiFiGAN TTS.
- WebSocket server for easy deployment.


## 💡 Upcoming Next
- More accurate and noise-robust streaming ASR and diarization models.
- Faster EOU detection and backchannel handling (e.g., bot will not stop speaking when user is saying something like "uhuh", "wow", "i see").
- Better streaming ASR and diarization pipeline.
- Better TTS model with more natural voice.
- Joint ASR and diarization model.





## 🚀 Quick Start

### Hardware requirements

- A computer with at least one GPU. At least 18GB VRAM is recommended for using 8B LLMs, and 10GB VRAM for 4B LLMs.
- A microphone connected to the computer.
- A speaker connected to the computer.

### Install dependencies

Create a new conda environment with the dependencies:
```bash
conda env create -f environment.yml
```

Activate the environment via `conda activate nemo-voice`

Alternatively, you can install the dependencies manually in an existing environment:
```bash
pip install -r requirements.txt
```
The incompatabilities errors from pip can be ignored.

### Configure the server

Edit the `server/server_config.yaml` file to configure the server, for example:
- Changing the LLM and prompt you want to use, by either putting a local path to a text file or the whole prompt string. See `example_prompts/` for examples to start with. 
- Configure the LLM parameters, such as temperature, max tokens, etc.
- Distribute different components to different GPUs if you have more than one.
- Adjust VAD parameters for sensitivity and end-of-turn detection timeout.


### Start the server

Open a terminal and run the server via:

```bash
NEMO_PATH=???  # Use your local NeMo path for the latest version
export PYTHONPATH=$NEMO_PATH:$PYTHONPATH

export HF_TOKEN=???  # Use your own HuggingFace token if needed
export WEBSOCKET_SERVER=websocket_server  # currently only support websocket_server
python ./server/server.py
```

### Launch the client
In another terminal, run the client via:

```bash
cd client
npm install
npm run dev
```

### Connect to the client via browser

Open the client via browser: `http://[YOUR MACHINE IP ADDRESS]:5173/`. You can mute/unmute your microphone via the "Mute" button, and reset the LLM context history and speaker cache by clicking the "Reset" button.


## 📑 Supported Models

### 🤖 LLM

Most LLMs from HuggingFace are supported. A few examples are:
- [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B)
- [meta-llama/Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)
- [nvidia/Nemotron-Mini-4B-Instruct](https://huggingface.co/nvidia/Nemotron-Mini-4B-Instruct)

### 🎤 ASR 

We use [cache-aware streaming FastConformer](https://arxiv.org/abs/2312.17279) to transcribe the user's speech. While new models are to be released, we use the existing Englishmodels for now:
- [nvidia/stt_en_fastconformer_hybrid_large_streaming_multi](https://huggingface.co/nvidia/stt_en_fastconformer_hybrid_large_streaming_multi)
- [stt_en_fastconformer_hybrid_large_streaming_80ms](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/stt_en_fastconformer_hybrid_large_streaming_80ms)

### 💬 Diarization

We use [streaming Sortformer](https://arxiv.org/abs/2409.06656) to detect the speaker for each user turn. As of now, we only support detecting 1 speaker for a single user turn, but different turns can be from different speakers, with a maximum of 4 speakers in the whole conversation.

### 🔉 TTS

We use [FastPitch-HiFiGAN](https://huggingface.co/nvidia/tts_en_fastpitch) to generate the speech for the LLM response, more TTS models will be supported in the future.


## 📝 Notes
- If directly loading from HuggingFace and got I/O erros, you can set `llm.model=<local_path>`, where the model is downloaded via somehing like `huggingface-cli download Qwen/Qwen3-8B --local-dir <local_path>`.
- The current ASR and diarization models are not noise-robust, you might need to use a noise-cancelling microphone or a quiet environment. But we will release better models soon.
- The diarization model works best with speakers that have much more different voices from each other, while it might not work well on some accents due to the limited training data.
- If using chrome browser, you might need to allow microphone access in the browser settings and add the ip address of the machine to the allow list via `chrome://flags/#unsafely-treat-insecure-origin-as-secure`.


## ☁️ NVIDIA NIM Services

You can also modify the `server/bot_websocket_server.py` to use NVIDIA NIM services for better LLM,ASR and TTS performance, by refering to these Pipecat services:
- [NVIDIA NIM LLM Service](https://github.com/pipecat-ai/pipecat/blob/main/src/pipecat/services/nim/llm.py)
- [NVIDIA Riva ASR Service](https://github.com/pipecat-ai/pipecat/blob/main/src/pipecat/services/riva/stt.py)
- [NVIDIA Riva TTS Service](https://github.com/pipecat-ai/pipecat/blob/main/src/pipecat/services/riva/tts.py)

For details of available NVIDIA NIM services, please refer to:
- [NVIDIA NIM LLM Service](https://docs.nvidia.com/nim/large-language-models/latest/introduction.html)
- [NVIDIA Riva ASR NIM Service](https://docs.nvidia.com/nim/riva/asr/latest/overview.html)
- [NVIDIA Riva TTS NIM Service](https://docs.nvidia.com/nim/riva/tts/latest/overview.html)


