# NeMo Voice Agent

A [Pipecat](https://github.com/pipecat-ai/pipecat) example demonstrating the simplest way to create a voice agent using NVIDIA NeMo STT/TTS service and HuggingFace LLM. Everything is open-source and deployed locally so you can have your own voice agent. Feel free to explore the code and see how different speech technologies can be integrated with LLMs to create a seamless conversation experience. 

As of now, we only support English input and output, but more languages will be supported in the future.



## ✨ Key Features

- Open-source, local deployment, and flexible customization.
- Allow users to talk to most LLMs from HuggingFace with configurable prompts. 
- Streaming speech recognition with low latency.
- FastPitch-HiFiGAN TTS for fast audio response generation.
- Speaker diarization up to 4 speakers in different userturns.
- WebSocket server for easy deployment.


## 💡 Upcoming Next
- More accurate and noise-robust streaming ASR models.
- Faster EOU detection and handling backchannel phrases.
- Better streaming ASR and speaker diarization pipeline.
- Better TTS model with more natural voice.
- Joint ASR and speaker diarization model.
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
The incompatibility errors from pip can be ignored.

### Configure the server

Edit the `server/server_config.yaml` file to configure the server, for example:
- Changing the LLM and system prompt you want to use in `llm.model` and `llm.system_prompt`, by either putting a local path to a text file or the whole prompt string. See `example_prompts/` for examples to start with. 
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

There should be a message in terminal showing the address and port of the client.

### Connect to the client via browser

Open the client via browser: `http://[YOUR MACHINE IP ADDRESS]:5173/` (or whatever address and port is shown in the terminal where the client was launched). 

You can mute/unmute your microphone via the "Mute" button, and reset the LLM context history and speaker cache by clicking the "Reset" button. 

**If using chrome browser, you need to add `http://[YOUR MACHINE IP ADDRESS]:5173/` to the allow list via `chrome://flags/#unsafely-treat-insecure-origin-as-secure`.**

If you want to use a different port for client connection, you can modify `client/vite.config.js` to change the `port` variable.

## 📑 Supported Models

### 🤖 LLM

Most LLMs from HuggingFace are supported. A few examples are:
- [Qwen/Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) (default)
- [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B)
- [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)
- [nvidia/Llama-3.1-Nemotron-Nano-8B-v1](https://huggingface.co/nvidia/Llama-3.1-Nemotron-Nano-8B-v1) 
- [nvidia/Nemotron-Mini-4B-Instruct](https://huggingface.co/nvidia/Nemotron-Mini-4B-Instruct)

Please refer to the HuggingFace webpage of each model to configure the model parameters `llm.generation_kwargs` and `llm.apply_chat_template_kwargs` in `server/server_config.yaml` as needed.

You can change the `llm.system_prompt` in `server/server_config.yaml` to configure the behavior of the LLM, by either putting a local path to a text file or the whole prompt string. See `example_prompts/` for examples to start with.


### 🎤 ASR 

We use [cache-aware streaming FastConformer](https://arxiv.org/abs/2312.17279) to transcribe the user's speech into text. While new models will be released soon, we use the existing English models for now:
- [stt_en_fastconformer_hybrid_large_streaming_80ms](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/stt_en_fastconformer_hybrid_large_streaming_80ms)  (default)
- [nvidia/stt_en_fastconformer_hybrid_large_streaming_multi](https://huggingface.co/nvidia/stt_en_fastconformer_hybrid_large_streaming_multi)

### 💬 Speaker Diarization

Speaker diarization aims to distinguish different speakers in the input speech audio. We use [streaming Sortformer](http://arxiv.org/abs/2507.18446) to detect the speaker for each user turn. 

As of now, we only support detecting 1 speaker per user turn, but different turns come from different speakers, with a maximum of 4 speakers in the whole conversation. 

Currently supported models are:
 - [nvidia/diar_streaming_sortformer_4spk-v2](https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2) (default)


Please note that in some circumstances, the diarization model might not work well in noisy environments, or it may confuse the speakers. In this case, you can disable the diarization by setting `diar.enabled` to `false` in `server/server_config.yaml`.

### 🔉 TTS

We use [FastPitch-HiFiGAN](https://huggingface.co/nvidia/tts_en_fastpitch) to generate the speech for the LLM response, and it only supports English output. More TTS models will be supported in the future.


### Turn-taking

As the new turn-taking prediction model is not yet released, we use the VAD-based turn-taking prediction for now. You can set the `vad.stop_secs` to the desired value in `server/server_config.yaml` to control the amount of silence needed to indicate the end of a user's turn.

Additionally, the voice agent support ignoring back-channel phrases while the bot is talking, which it means phrases such as "uh-huh", "yeah", "okay"  will not interrupt the bot while it's talking. To control the backchannel phrases to be used, you can set the `turn_taking.backchannel_phrases` to the desired list of phrases or a file path to a yaml file containing the list of phrases in `server/server_config.yaml`. Setting it to `null` will disable detecting the backchannel phrases, and that the VAD will interrupt the bot immediately when the user starts speaking.


## 📝 Notes & FAQ
- Only one connection to the server is supported at a time, a new connection will disconnect the previous one, but the context will be preserved.
- If directly loading from HuggingFace and got I/O erros, you can set `llm.model=<local_path>`, where the model is downloaded using a command like `huggingface-cli download Qwen/Qwen2.5-7B-Instruct --local-dir <local_path>`. Same for TTS models.
- The current ASR and diarization models are not noise-robust, you might need to use a noise-cancelling microphone or a quiet environment. But we will release better models soon.
- The diarization model works best with speakers that have much more different voices from each other, while it might not work well on some accents due to the limited training data.
- If you see errors like `SyntaxError: Unexpected reserved word` when running `npm run dev`, please update the Node.js version.
- If you see the error `Error connecting: Cannot read properties of undefined (reading 'enumerateDevices')`, it usually means the browser is not allowed to access the microphone. Please check the browser settings and add `http://[YOUR MACHINE IP ADDRESS]:5173/` to the allow list, e.g., via `chrome://flags/#unsafely-treat-insecure-origin-as-secure` for chrome browser.
- If you see something like `node:internal/errors:496` when running `npm run dev`, remove the `client/node_modules` folder and run `npm install` again, then run `npm run dev` again.



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


