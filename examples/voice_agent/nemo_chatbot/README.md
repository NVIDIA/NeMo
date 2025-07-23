# NeMo Voice Agent

A Pipecat example demonstrating the simplest way to create a voice agent using `WebsocketTransport`, NeMo STT/TTS service, and HuggingFace LLM. Evertying is deployed locally so you can own your own agent.

## 🚀 Quick Start

### Install dependencies

```bash
conda env create -f environment.yml
```

Activate the environment via `conda activate nemo-pipecat`

### Run the server

```bash
NEMO_PATH=???  # Use your own NeMo path
export PYTHONPATH=$NEMO_PATH:$PYTHONPATH
export HF_TOKEN=???  # Use your own HuggingFace token
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

Open the client via browser: `http://[YOUR SERVER IP ADDRESS]:5173/`
