# NeMo Voice Agent

A [Pipecat](https://github.com/pipecat-ai/pipecat) example demonstrating the simplest way to create a voice agent using NVIDIA NeMo STT/TTS service and HuggingFace LLM. Everything is open-source and deployed locally so you can have your own voice agent. Feel free to explore the code and see how different speech technologies can be integrated with LLMs to create a seamless conversation experience.



## ✨ Key Features

- Open-source, local deployment, and flexible customization.
- Talk to most LLMs from HuggingFace, use different prompts to configure the agent. 
- Streaming speech recognition.
- FastPitch-HiFiGAN TTS.
- Speaker diarization up to 4 speakers (checkpoint will be released very soon).
- WebSocket server for easy deployment.


## 💡 Upcoming Next
- More accurate and noise-robust streaming ASR and diarization models.
- Faster EOU detection and backchannel handling (e.g., bot will not stop speaking when user is saying something like "uhuh", "wow", "i see").
- Better streaming ASR and diarization pipeline.
- Better TTS model with more natural voice.
- Joint ASR and diarization model.
- Function calling, RAG, etc.



## 🚀 Quick Start

### Hardware requirements

- A computer with at least one GPU. At least 18GB VRAM is recommended for using 8B LLMs, and 10GB VRAM for 4B LLMs.
- A microphone connected to the computer.
- A speaker connected to the computer.

### Install dependencies

First, install or update the npm and node.js to the latest version, for example:

```bash
sudo apt-get update
sudo apt-get install -y npm nodejs
```

or:

```bash
curl -fsSL https://fnm.vercel.app/install | bash
. ~/.bashrc
fnm use --install-if-missing 20
```

Second, create a new conda environment with the dependencies:

```bash
conda env create -f environment.yml
```

Then you can activate the environment via `conda activate nemo-voice`.

Alternatively, you can install the dependencies manually in an existing environment via:
```bash
pip install -r requirements.txt
```
The incompatability errors from pip can be ignored, if any.

### Configure the server

Edit the `server/server_config.yaml` file to configure the server, for example:
- Changing the LLM and prompt you want to use, by either putting a local path to a text file or the whole prompt string. See `example_prompts/` for examples to start with. 
- Configure the LLM parameters, such as temperature, max tokens, etc.
- Distribute different components to different GPUs if you have more than one.
- Adjust VAD parameters for sensitivity and end-of-turn detection timeout.

**If you want to access the server from a different machine, you need to change the `baseUrl` in `client/src/app.ts` to the actual ip address of the server machine.**



### Start the server

Open a terminal and run the server via:

```bash
NEMO_PATH=???  # Use your local NeMo path for the latest version
export PYTHONPATH=$NEMO_PATH:$PYTHONPATH

# export HF_TOKEN="hf_..."  # Use your own HuggingFace API token if needed, as some models may require.
# export HF_HUB_CACHE="/path/to/your/huggingface/cache"  # change where HF cache is stored if you don't want to use the default cache
# export SERVER_CONFIG_PATH="/path/to/your/server_config.yaml"  # change where the server config is stored if you have a couple of different configs
python ./server/server.py
```

### Launch the client
In another terminal on the server machine, start the client via:

```bash
cd client
npm install
npm run dev
```

### Connect to the client via browser

Open the client via browser: `http://[YOUR MACHINE IP ADDRESS]:5173/`. You can mute/unmute your microphone via the "Mute" button, and reset the LLM context history and speaker cache by clicking the "Reset" button. 

**If using chrome browser, you need to add `http://[YOUR MACHINE IP ADDRESS]:5173/` to the allow list via `chrome://flags/#unsafely-treat-insecure-origin-as-secure`.**


## 📑 Supported Models

### 🤖 LLM

Most LLMs from HuggingFace are supported. A few examples are:
- [nvidia/Llama-3.1-Nemotron-Nano-8B-v1](https://huggingface.co/nvidia/Llama-3.1-Nemotron-Nano-8B-v1) (default)
- [meta-llama/Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)
- [Qwen/Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
- [nvidia/Nemotron-Mini-4B-Instruct](https://huggingface.co/nvidia/Nemotron-Mini-4B-Instruct)

Please refer to the HuggingFace webpage of each model to configure the model parameters `llm.generation_kwargs` and `llm.apply_chat_template_kwargs` in `server/server_config.yaml` as needed.

### 🎤 ASR 

We use [cache-aware streaming FastConformer](https://arxiv.org/abs/2312.17279) to transcribe the user's speech. While new models are to be released, we use the existing English models for now:
- [stt_en_fastconformer_hybrid_large_streaming_80ms](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/stt_en_fastconformer_hybrid_large_streaming_80ms)  (default)
- [nvidia/stt_en_fastconformer_hybrid_large_streaming_multi](https://huggingface.co/nvidia/stt_en_fastconformer_hybrid_large_streaming_multi)

### 💬 Diarization

We use [streaming Sortformer](http://arxiv.org/abs/2507.18446) to detect the speaker for each user turn. As of now, we only support detecting 1 speaker for a single user turn, but different turns can be from different speakers, with a maximum of 4 speakers in the whole conversation.

### 🔉 TTS

We use [FastPitch-HiFiGAN](https://huggingface.co/nvidia/tts_en_fastpitch) to generate the speech for the LLM response, and it only supports English output. More TTS models will be supported in the future.


## 📝 Notes & FAQ
- Only one connection to the server is supported at a time, a new connection will disconnect the previous one, but the context will be preserved.
- If directly loading from HuggingFace and got I/O erros, you can set `llm.model=<local_path>`, where the model is downloaded via somehing like `huggingface-cli download Qwen/Qwen3-8B --local-dir <local_path>`. Same for TTS models.
- The current ASR and diarization models are not noise-robust, you might need to use a noise-cancelling microphone or a quiet environment. But we will release better models soon.
- The diarization model works best with speakers that have much more different voices from each other, while it might not work well on some accents due to the limited training data.
- If you see errors like `SyntaxError: Unexpected reserved word` when running `npm run dev`, please update the Node.js version.
- If you see the error `Error connecting: Cannot read properties of undefined (reading 'enumerateDevices')`, it usually means the browser is not allowed to access the microphone. Please check the browser settings and add `http://[YOUR MACHINE IP ADDRESS]:5173/` to the allow list, e.g., via `chrome://flags/#unsafely-treat-insecure-origin-as-secure` for chrome browser.



## ☁️ NVIDIA NIM Services

NVIDIA also provides a variety of [NIM](https://developer.nvidia.com/nim?sortBy=developer_learning_library%2Fsort%2Ffeatured_in.nim%3Adesc%2Ctitle%3Aasc&hitsPerPage=12) services for better ASR, TTS and LLM performance with more efficient deployment on either cloud or local servers.

You can also modify the `server/bot_websocket_server.py` to use NVIDIA NIM services for better LLM,ASR and TTS performance, by refering to these Pipecat services:
- [NVIDIA NIM LLM Service](https://github.com/pipecat-ai/pipecat/blob/main/src/pipecat/services/nim/llm.py)
- [NVIDIA Riva ASR Service](https://github.com/pipecat-ai/pipecat/blob/main/src/pipecat/services/riva/stt.py)
- [NVIDIA Riva TTS Service](https://github.com/pipecat-ai/pipecat/blob/main/src/pipecat/services/riva/tts.py)

For details of available NVIDIA NIM services, please refer to:
- [NVIDIA NIM LLM Service](https://docs.nvidia.com/nim/large-language-models/latest/introduction.html)
- [NVIDIA Riva ASR NIM Service](https://docs.nvidia.com/nim/riva/asr/latest/overview.html)
- [NVIDIA Riva TTS NIM Service](https://docs.nvidia.com/nim/riva/tts/latest/overview.html)


