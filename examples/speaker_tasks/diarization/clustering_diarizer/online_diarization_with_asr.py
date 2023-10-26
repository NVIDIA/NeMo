from scipy.io import wavfile
from nemo.collections.asr.parts.utils.diarization_utils import OnlineDiarWithASR, write_txt
from nemo.collections.asr.models import OnlineClusteringDiarizer
from pytorch_lightning import seed_everything
from nemo.core.config import hydra_runner
import torch
from nemo.utils import logging
import gradio as gr
import numpy as np

seed_everything(42)

"""
This script demonstrates how to run a simulated online speaker diarization with asr.
Usage:

python online_diar_with_asr_infer.py \
    diarizer.manifest_filepath=<path to manifest file> \
    diarizer.simulation_uniq_id='en_0638' \
    diarizer.out_dir=<path to the output directory> \
    diarizer.speaker_embeddings.model_path=<pretrained modelname or path to .nemo> \
    diarizer.asr.model_path=<pretrained modelname or path to .nemo> \
    diarizer.asr.parameters.asr_based_vad=True \
    diarizer.speaker_embeddings.parameters.save_embeddings=False \
    diarizer.asr.parameters.streaming_simulation=True \
    diarizer.clustering.parameters.history_buffer_size=100 \
    diarizer.clustering.parameters.current_buffer_size=200 \


After run the above python script, run the following at the directory you specified for `diarizer.out_dir`.
$ watch  -n 0.1 --color "script -q -c 'cat print_script.sh' /dev/null"

This will display online transcript of the input audio signal with color coded speakers.

Currently, the following ASR models on NGC are supported:

    stt_en_conformer_ctc*
"""


@hydra_runner(config_path="../conf/inference", config_name="online_diar_infer_general.yaml")
def main(cfg):
    online_diar_asr = OnlineDiarWithASR(cfg=cfg)
    diar = online_diar_asr.diar

    if cfg.diarizer.asr.parameters.streaming_simulation:
        diar.uniq_id = cfg.diarizer.simulation_uniq_id
        online_diar_asr.get_audio_rttm_map(diar.uniq_id)
        diar.single_audio_file_path = diar.AUDIO_RTTM_MAP[diar.uniq_id]['audio_filepath']
        online_diar_asr.rttm_file_path = diar.AUDIO_RTTM_MAP[diar.uniq_id]['rttm_filepath']
    else:
        online_diar_asr.rttm_file_path = None

    diar._init_segment_variables()
    diar.device = online_diar_asr.device
    write_txt(f"{diar._out_dir}/print_script.sh", "")
    
    if cfg.diarizer.asr.parameters.streaming_simulation:
        samplerate, sdata = wavfile.read(diar.single_audio_file_path)
        if  diar.AUDIO_RTTM_MAP[diar.uniq_id]['offset'] and diar.AUDIO_RTTM_MAP[diar.uniq_id]['duration']:
            
            offset = samplerate*diar.AUDIO_RTTM_MAP[diar.uniq_id]['offset']
            duration = samplerate*diar.AUDIO_RTTM_MAP[diar.uniq_id]['duration']
            stt, end = int(offset), int(offset + duration)
            sdata = sdata[stt:end]

        for index in range(int(np.floor(sdata.shape[0]/online_diar_asr.n_frame_len))):
            shift = online_diar_asr.CHUNK_SIZE
            sample_audio = sdata[shift*index:shift*(index+1)]
            online_diar_asr.buffer_counter = index
            online_diar_asr.streaming_step(sample_audio)
    else:
        isTorch = torch.cuda.is_available()
        iface = gr.Interface(
            fn=online_diar_asr.audio_queue_launcher,
            inputs=[
                gr.Audio(source="microphone", type="numpy", streaming=True), 
                "state",
            ],
            outputs=[
                "textbox",
                "state",
            ],
            layout="horizontal",
            theme="huggingface",
            title=f"NeMo Streaming Conformer CTC Large - English, CUDA:{isTorch}",
            description="Demo for English speech recognition using Conformer Transducers",
            allow_flagging='never',
            live=True,
        )
        iface.launch(share=False)

if __name__ == "__main__":
    main()
