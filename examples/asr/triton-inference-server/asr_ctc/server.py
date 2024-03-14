import logging
import os
import soundfile
import tempfile
import uuid

import numpy as np
import torch  # pytype: disable=import-error
import nemo.collections.asr as nemo_asr

from pytriton.decorators import batch
from pytriton.model_config import ModelConfig, Tensor
from pytriton.triton import Triton, TritonConfig


MAX_BATCH_SIZE = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# MODEL = nemo_asr.models.ASRModel.from_pretrained("nvidia/parakeet-rnnt-1.1b", map_location=DEVICE)
MODEL = nemo_asr.models.ASRModel.restore_from(f"{os.environ['HOME']}/.cache/huggingface/hub/models--nvidia--parakeet-ctc-1.1b/snapshots/085a3de63c7598065b072cd8f2182e6a5fa593eb/parakeet-ctc-1.1b.nemo", map_location=DEVICE)


@batch
def _infer_fn(**inputs):
    (input1_batch, input1_lengths) = inputs.values()

    import ipdb; ipdb.set_trace()

    with tempfile.TemporaryDirectory() as tmpdir:
        # Write audio files to temp disk
        audio_paths = []
        for sample_id in range(input1_batch.shape[0]):
            audio_path = os.path.join(tmpdir, f'audio_{uuid.uuid4()}.wav')
            audio_len = input1_lengths[sample_id][0]
            soundfile.write(audio_path, input1_batch[sample_id, :audio_len], samplerate=16000)
            audio_paths.append(audio_path)

        transcriptions = MODEL.transcribe(audio_paths, batch_size=MAX_BATCH_SIZE)

        # if transcriptions form a tuple (from RNNT), extract just "best" hypothesis
        if type(transcriptions) == tuple and len(transcriptions) == 2:
            transcriptions = transcriptions[0]

    # TODO: Question - Is it possible to return a list of unpacked tensors (as a list?) instead of a packed tensor?
    # Ans: No. Tensors only.

    # pack the transcription strs into a numpy tensors
    transcript_t_list = []
    transcrip_t_len_list = []
    for transcription in transcriptions:
        # TODO: Question - is it possible to avoid this encoding/decoding into np.object_?
        # Ans: New Backend - solved. Some minor restrictions

        transcript_t = np.array([str(x).encode("utf-8") for x in transcription], dtype=np.object_)
        transcript_t = transcript_t.reshape([1, -1])
        transcript_t_list.append(transcript_t)
        transcrip_t_len_list.append(transcript_t.shape[1])

    # pack the transcriptions into a single tensor
    transcript_t_packed = np.zeros([len(transcriptions), max(transcrip_t_len_list)], dtype=np.object_)
    for i, transcript_t in enumerate(transcript_t_list):
        transcript_t_packed[i, :transcript_t.shape[1]] = transcript_t

    transcrip_t_len_list = np.array(transcrip_t_len_list, dtype=np.int64).reshape([-1, 1])

    return [transcript_t_packed, transcrip_t_len_list]


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s: %(message)s")
logger = logging.getLogger("examples.nemo_asr.server")

triton_config = TritonConfig(
    http_port=5000
)

with Triton(config=triton_config) as triton:
    logger.info("Loading ASR model.")

    # TODO: Question - How to do you enqueue samples by sample into this engine? Similar to vLLM / FastAPI?
    # TRTLLM - Additional adapter.
    triton.bind(
        model_name="NeMoASR",
        infer_func=_infer_fn,
        inputs=[
            Tensor(dtype=np.float32, shape=(-1,)),
            Tensor(dtype=np.int64, shape=(-1,)),
        ],
        outputs=[
            Tensor(dtype=np.object_, shape=(-1,)),
            Tensor(dtype=np.int64, shape=(-1,)),
        ],
        config=ModelConfig(max_batch_size=MAX_BATCH_SIZE),
        strict=True,
    )
    logger.info("Serving models")
    triton.serve()
