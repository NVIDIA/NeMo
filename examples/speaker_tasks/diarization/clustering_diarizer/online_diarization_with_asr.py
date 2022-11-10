#!/usr/bin/env python
# coding: utf-8
from scipy.io import wavfile
from nemo.collections.asr.parts.utils.diarization_utils import OnlineDiarWithASR, write_txt
from nemo.collections.asr.models import OnlineDiarizer
from pytorch_lightning import seed_everything
from nemo.core.config import hydra_runner
import torch
from nemo.utils import logging
import gradio as gr
import numpy as np

seed_everything(42)

@hydra_runner(config_path="../conf/inference", config_name="online_diar_infer_general.yaml")
def main(cfg):
    diar = OnlineDiarizer(cfg)
    asr_diar = OnlineDiarWithASR(cfg=cfg)

    if cfg.diarizer.asr.parameters.streaming_simulation:
        diar.uniq_id = cfg.diarizer.simulation_uniq_id
        asr_diar.get_audio_rttm_map(diar.uniq_id)
        diar.single_audio_file_path = diar.AUDIO_RTTM_MAP[diar.uniq_id]['audio_filepath']
        diar.rttm_file_path = diar.AUDIO_RTTM_MAP[diar.uniq_id]['rttm_filepath']
        asr_diar.rttm_file_path = diar.rttm_file_path
    else:
        diar.rttm_file_path = None

    diar._init_segment_variables()
    diar.device = asr_diar.device
    write_txt(f"{diar._out_dir}/print_script.sh", "")
    
    if cfg.diarizer.asr.parameters.streaming_simulation:
        samplerate, sdata = wavfile.read(diar.single_audio_file_path)
        if  diar.AUDIO_RTTM_MAP[diar.uniq_id]['offset'] and diar.AUDIO_RTTM_MAP[diar.uniq_id]['duration']:
            
            offset = samplerate*diar.AUDIO_RTTM_MAP[diar.uniq_id]['offset']
            duration = samplerate*diar.AUDIO_RTTM_MAP[diar.uniq_id]['duration']
            stt = int(offset)
            end = int(offset + duration)
            sdata = sdata[stt:end]

        for index in range(int(np.floor(sdata.shape[0]/asr_diar.n_frame_len))):
            asr_diar.buffer_counter = index
            sample_audio = sdata[asr_diar.CHUNK_SIZE*(asr_diar.buffer_counter):asr_diar.CHUNK_SIZE*(asr_diar.buffer_counter+1)]
            asr_diar.streaming_step(sample_audio)
    else:
        isTorch = torch.cuda.is_available()
        iface = gr.Interface(
            fn=asr_diar.audio_queue_launcher,
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
