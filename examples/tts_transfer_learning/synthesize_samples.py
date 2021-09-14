from nemo.collections.tts.models.base import SpectrogramGenerator, Vocoder, TextToWaveform
from nemo.collections.tts.models import FastPitchModel
from nemo.collections.tts.modules.hifigan_modules import Generator as Hifigan_generator
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import torch
import json
import librosa
import os
from nemo.collections.tts.models import TwoStagesModel
from nemo.collections.asr.modules import AudioToMelSpectrogramPreprocessor
from nemo.collections.asr.parts.preprocessing.features import WaveformFeaturizer
from nemo.collections.tts.models import HifiGanModel
import argparse
import torchaudio

def infer(spec_gen_model, vocoder_model, str_input, speaker = None):
    parser_model = spec_gen_model
    with torch.no_grad():
        parsed = parser_model.parse(str_input)
        if speaker is not None:
            speaker = torch.tensor([speaker]).long().cuda()
        spectrogram = spec_gen_model.generate_spectrogram(tokens=parsed, speaker = speaker)
        if isinstance(vocoder_model, Hifigan_generator):
            audio = vocoder_model(x=spectrogram.half()).squeeze(1)
        else:
            audio = vocoder_model.convert_spectrogram_to_audio(spec=spectrogram)
        
    if spectrogram is not None:
        if isinstance(spectrogram, torch.Tensor):
            spectrogram = spectrogram.to('cpu').numpy()
        if len(spectrogram.shape) == 3:
            spectrogram = spectrogram[0]
    if isinstance(audio, torch.Tensor):
        audio = audio.to('cpu').numpy()
    return spectrogram, audio

def get_best_ckpt(experiment_base_dir, new_speaker_id, duration_mins, mixing_enabled, original_speaker_id):
    if not mixing_enabled:
        exp_dir = "{}/{}_to_{}_no_mixing_{}_mins".format(experiment_base_dir, original_speaker_id, new_speaker_id, duration_mins)
    else:
        exp_dir = "{}/{}_to_{}_mixing_{}_mins".format(experiment_base_dir, original_speaker_id, new_speaker_id, duration_mins)
    
    ckpt_candidates = []
    last_ckpt = None
    for root, dirs, files in os.walk(exp_dir):
        for file in files:
            if file.endswith(".ckpt"):
                val_error = float(file.split("v_loss=")[1].split("-epoch")[0])
                if "last" in file:
                    last_ckpt = os.path.join(root, file)
                ckpt_candidates.append( (val_error, os.path.join(root, file)))
    ckpt_candidates.sort()
    
    return ckpt_candidates, last_ckpt

wav_featurizer = WaveformFeaturizer(sample_rate=44100, int_values=False, augmentor=None)
mel_processor = AudioToMelSpectrogramPreprocessor(
        window_size = None,
        window_stride = None,
        sample_rate=44100,
        n_window_size=2048,
        n_window_stride=512,
        window="hann",
        normalize=None,
        n_fft=None,
        preemph=None,
        features=80,
        lowfreq=0,
        highfreq=None,
        log=True,
        log_zero_guard_type="add",
        log_zero_guard_value=1e-05,
        dither=0.0,
        pad_to=1,
        frame_splicing=1,
        exact_pad=False,
        stft_exact_pad=False,
        stft_conv=False,
        pad_value=0,
        mag_power=1.0
)


def main():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--data_dir', type=str, default="/home/pneekhara/Datasets/78419/Hi_Fi_TTS_v_0_backup")
    parser.add_argument('--experiment_base_dir', type=str, default="/home/pneekhara/ExperimentsAutomatedResetPitch/")
    parser.add_argument('--num_val', type=int, default=50)
    parser.add_argument('--out_dir', type=str, default="/home/pneekhara/synthesized_samples3/")
    args = parser.parse_args()

    cfg = {'linvocoder':  {'_target_': 'nemo.collections.tts.models.two_stages.GriffinLimModel',
                        'cfg': {'n_iters': 64, 'n_fft': 2048, 'l_hop': 512}},
        'mel2spec': {'_target_': 'nemo.collections.tts.models.two_stages.MelPsuedoInverseModel',
                    'cfg': {'sampling_rate': 44100, 'n_fft': 2048, 
                            'mel_fmin': 0, 'mel_fmax': None, 'mel_freq': 80}}}
    
    vocoder_gl = TwoStagesModel(cfg).eval().cuda()
    vocoder = HifiGanModel.load_from_checkpoint("/home/pneekhara/PreTrainedModels/HifiGan--val_loss=0.08-epoch=899.ckpt")
    vocoder.eval().cuda()

    data_dir = args.data_dir
    experiment_base_dir = args.experiment_base_dir

    clean_other_mapping = {
        92 : 'clean',
        6097 : 'clean',
        92 : 'clean',
        9017 : 'clean',
        6670 : 'other',
        6671 : 'other',
        8051 : 'clean',
        9136 : 'other',
        11614 : 'other',
        11697 : 'other',
    }

    full_data_ckpts = {
        92 : '/home/pneekhara/Checkpoints/FastPitchSpeaker92Epoch999.ckpt',
        6097 : '/home/pneekhara/Checkpoints/FastPitch--v_loss=0.75-epoch=999-last.ckpt'
    }

    num_val = args.num_val

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    for speaker in clean_other_mapping:
        manifest_path = os.path.join(data_dir, "{}_manifest_{}_{}.json".format(speaker, clean_other_mapping[speaker], "dev"))
        val_records = []
        with open(manifest_path, "r") as f:
            for i, line in enumerate(f):
                val_records.append( json.loads(line) )
                if len(val_records) >= num_val:
                    break
        print ("**** REAL VALIDATION *****")
        for vidx, val_record in enumerate(val_records):
            print("Audio path:", val_record['audio_filepath'] )
            audio_path = os.path.join( data_dir, val_record['audio_filepath'] )
            
            real_wav = wav_featurizer.process(audio_path)
            real_mel, _ = mel_processor.get_features(real_wav[None], torch.tensor([[real_wav.shape[0]]]).long() )
            real_mel = real_mel[0].cuda()
            with torch.no_grad():
                # vocoded_audio_real = vocoder(x=real_mel.half()).squeeze(1)
                vocoded_audio_real = vocoder.convert_spectrogram_to_audio(spec=real_mel).cpu().numpy()

            # vocoded_audio_real = vocoded_audio_real.to('cpu').numpy()
            # audio, sr = librosa.load(audio_path, sr=None)
            print (vidx, val_record['text'])
            
            fname = os.path.join(args.out_dir, "real_actual_{}_{}.wav".format(speaker, vidx))
            torchaudio.save(fname, torch.tensor(real_wav.flatten(), dtype=torch.float)[None], 44100)

            fname = os.path.join(args.out_dir, "real_vocoded_{}_{}.wav".format(speaker, vidx))
            torchaudio.save(fname, torch.tensor(vocoded_audio_real.flatten(), dtype=torch.float)[None], 44100)
            # print("Vocoded (GL) from real spectrogram:", speaker)
        print ("************************")
        print ("********Generated*********")
        for duration_mins in ["All", 60, 30, 5]:
            for mixing in [False, True]:
                last_ckpt = None
                if duration_mins == "All" and speaker in full_data_ckpts:
                    if mixing:
                        continue
                    last_ckpt = full_data_ckpts[speaker]
                else:
                    _, last_ckpt = get_best_ckpt(experiment_base_dir, speaker, duration_mins, mixing, 8051)
                if last_ckpt is None:
                    print ("Checkpoint not found for:", "Speaker: {} | Dataset size: {} mins | Mixing:{}".format(speaker, duration_mins, mixing)) 
                    continue
                    
                # print(last_ckpt)
                spec_model = FastPitchModel.load_from_checkpoint(last_ckpt)
                spec_model.eval().cuda()
                _speaker=None

                mix_str = "nomix"
                if mixing:
                    mix_str = "mix"
                    _speaker = 1
                for vidx, val_record in enumerate(val_records):
                    print ("SYNTHESIZED FOR -- Speaker: {} | Dataset size: {} mins | Mixing:{} | Text: {}".format(speaker, duration_mins, mixing, val_record['text']))
                    _, audio = infer(spec_model, vocoder, val_record['text'], speaker = _speaker)
                    fname = os.path.join(args.out_dir, "synthesized_{}-{}_{}_{}.wav".format(duration_mins, mix_str, speaker, vidx))
                    torchaudio.save(fname, torch.tensor(audio.flatten(), dtype=torch.float)[None], 44100)

if __name__ == '__main__':
    main()