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
from nemo.collections.asr.data import audio_to_text
from omegaconf import OmegaConf
from hydra.utils import instantiate

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
                if "v_loss" in file:
                    val_error = float(file.split("v_loss=")[1].split("-epoch")[0])
                elif "val_loss" in file:
                    val_error = float(file.split("val_loss=")[1].split("-epoch")[0])
                else:
                    val_error = 0.0
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
    parser.add_argument('--filelist_dir', type=str, default="/home/pneekhara/filelists")
    parser.add_argument('--original_spec_model_ckpt', type=str, default="/home/pneekhara/PreTrainedModels/FastPitch.nemo")
    parser.add_argument('--experiment_base_dir', type=str, default="/home/pneekhara/ExperimentsAutomatedResetPitch/")
    parser.add_argument('--experiment_base_dir_hifi', type=str, default="/home/pneekhara/ExperimentsHiFiFinetuning/")
    parser.add_argument('--num_val', type=int, default=50)
    parser.add_argument('--use_finetuned_vocoder', type=int, default=0)
    parser.add_argument('--out_dir', type=str, default="/home/pneekhara/synthesized_samples_universalvoc/")
    parser.add_argument('--fastpitch_cfg_path', type=str, default="/home/pneekhara/NeMo/examples/tts/conf/fastpitch_align_44100.yaml")
    args = parser.parse_args()

    fastpitch_cfg = OmegaConf.load(args.fastpitch_cfg_path)

    cfg = {'linvocoder':  {'_target_': 'nemo.collections.tts.models.two_stages.GriffinLimModel',
                        'cfg': {'n_iters': 64, 'n_fft': 2048, 'l_hop': 512}},
        'mel2spec': {'_target_': 'nemo.collections.tts.models.two_stages.MelPsuedoInverseModel',
                    'cfg': {'sampling_rate': 44100, 'n_fft': 2048, 
                            'mel_fmin': 0, 'mel_fmax': None, 'mel_freq': 80}}}
    
    vocoder_gl = TwoStagesModel(cfg).eval().cuda()
    # vocoder_universal = HifiGanModel.load_from_checkpoint("/home/pneekhara/PreTrainedModels/HifiGan--val_loss=0.08-epoch=899.ckpt")
    vocoder_universal = HifiGanModel.load_from_checkpoint("/home/pneekhara/PreTrainedModels/HiFiMix_No_92_6097.ckpt")
    vocoder_universal.eval().cuda()

    data_dir = args.data_dir
    experiment_base_dir = args.experiment_base_dir

    clean_other_mapping = {
        92 : 'clean',
        6097 : 'clean',
        8051 : 'clean',
        11697 : 'other',
        6670 : 'other',
        6671 : 'other',
        9017 : 'clean',
        11614 : 'other',
        9136 : 'other',
        12787 : 'other'
        # 92 : 'clean',
        # 9017 : 'clean',
        # 6670 : 'other',
        # 6671 : 'other',
        # 8051 : 'clean',
        # 9136 : 'other',
        # 11614 : 'other',
        # 11697 : 'other',
    }

    full_data_ckpts = {
        92 : '/home/pneekhara/Checkpoints/FastPitchSpeaker92Epoch999.ckpt',
        6097 : '/home/pneekhara/Checkpoints/FastPitch--v_loss=0.75-epoch=999-last.ckpt'
    }

    num_val = args.num_val

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    speaking_rate_stats = {}

    spec_model_original = FastPitchModel.restore_from(args.original_spec_model_ckpt)
    spec_model_original.eval().cuda()

    for speaker in clean_other_mapping:
        manifest_path = os.path.join(args.filelist_dir, "{}_mainifest_{}_ns_all_local.json".format(speaker, "dev"))
        val_records = []
        with open(manifest_path, "r") as f:
            for i, line in enumerate(f):
                val_records.append( json.loads(line) )
                if len(val_records) >= num_val:
                    break
        
        fastpitch_cfg.validation_datasets = manifest_path
        fastpitch_cfg.prior_folder = "/home/pneekhara/priors/{}".format(speaker)
        val_dataset = instantiate(fastpitch_cfg.model.validation_ds.dataset)
        
        fastpitch_cfg.model.validation_ds.dataloader_params.batch_size = 1
        fastpitch_cfg.model.validation_ds.dataloader_params.num_workers = 1
        fastpitch_cfg.model.validation_ds.dataloader_params.shuffle = False
        val_loader = torch.utils.data.DataLoader(val_dataset, collate_fn=val_dataset.collate_fn, **fastpitch_cfg.model.validation_ds.dataloader_params)


        print ("**** REAL VALIDATION *****")
        original_model_audio_durations = []
        for vidx, val_record in enumerate(val_records):
            print("Audio path:", val_record['audio_filepath'] )
            audio_path = val_record['audio_filepath']
            
            real_wav = wav_featurizer.process(audio_path)
            real_mel, _ = mel_processor.get_features(real_wav[None], torch.tensor([[real_wav.shape[0]]]).long() )
            real_mel = real_mel[0].cuda()


            _, audio_from_text = infer(spec_model_original, vocoder_universal, val_record['text'], speaker = None)
            fname = os.path.join(args.out_dir, "baseline_originalModel_{}_{}.wav".format(speaker, vidx))
            torchaudio.save(fname, torch.tensor(audio_from_text.flatten(), dtype=torch.float)[None], 44100)
            original_model_audio_durations.append(len(audio_from_text.flatten()))
            with torch.no_grad():
                # vocoded_audio_real = vocoder(x=real_mel.half()).squeeze(1)
                vocoded_audio_real = vocoder_universal.convert_spectrogram_to_audio(spec=real_mel).cpu().numpy()

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


        for duration_mins in ["All", 60, 30, 5, 1]:
            for mixing in [False, True]:
                last_ckpt = None
                if duration_mins == "All" and speaker in full_data_ckpts:
                    if mixing:
                        continue
                    last_ckpt = full_data_ckpts[speaker]
                    vocoder = vocoder_universal
                else:
                    _, last_ckpt = get_best_ckpt(experiment_base_dir, speaker, duration_mins, mixing, 8051)
                    _, last_ckpt_vocoder = get_best_ckpt(args.experiment_base_dir_hifi, speaker, duration_mins, mixing, 8051)
                    if last_ckpt_vocoder is not None and args.use_finetuned_vocoder == 1:
                        print ("Loading finetuned vocoder")
                        vocoder = HifiGanModel.load_from_checkpoint(last_ckpt_vocoder)
                        vocoder.eval().cuda()
                    else:
                        vocoder = vocoder_universal
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

                gt_phonmenes_per_second_list = []
                synthFA_phonmenes_per_second_list = []
                synth_phonmenes_per_second_list = []
                synthOriginalModel_phonmenes_per_second_list = []
                for vidx, val_batch in enumerate(val_loader):
                    val_record = val_records[vidx]

                    with torch.no_grad():
                        audio, audio_lens, text, text_lens, attn_prior, pitch, speakers = val_batch
                        mels, spec_len = spec_model.preprocessor(input_signal=audio.cuda(), length=audio_lens.cuda())

                        if mixing:
                            _speaker_tensor = torch.tensor([1]).long().cuda()
                        else:
                            _speaker_tensor = None
                        mels_pred, *_ = spec_model(
                            text=text.cuda(),
                            durs=None,
                            pitch=None,
                            speaker=_speaker_tensor,
                            pace=1.0,
                            spec=mels.cuda(),
                            attn_prior=attn_prior.cuda(),
                            mel_lens=spec_len.cuda(),
                            input_lens=text_lens.cuda()
                        )

                        audio_from_text_dur = vocoder.convert_spectrogram_to_audio(spec=mels_pred)
                        audio_from_text_dur = audio_from_text_dur.to('cpu').numpy().flatten()

                        fname = os.path.join(args.out_dir, "synthesizedForceAlignment_{}-{}_{}_{}.wav".format(duration_mins, mix_str, speaker, vidx))
                        torchaudio.save(fname, torch.tensor(audio_from_text_dur, dtype=torch.float)[None], 44100)

                        text_length = text_lens[0].item()
                        gt_audio_length = audio_lens[0].item()
                        synth_audio_length = len(audio_from_text_dur)
                        
                        gt_phonmenes_per_second = text_length/ (gt_audio_length/44100.0)
                        synthFA_phonemes_per_second = text_length/ (synth_audio_length/44100.0)
                        gt_phonmenes_per_second_list.append(gt_phonmenes_per_second)
                        synthFA_phonmenes_per_second_list.append(synthFA_phonemes_per_second)

                        synthOriginalModel_phonmenes_per_second_list.append(text_length/ (original_model_audio_durations[vidx]/44100.0))

                        if duration_mins == "All" and not mixing:
                            mels_pred, *_ = spec_model_original(
                                text=text.cuda(),
                                durs=None,
                                pitch=None,
                                speaker=None,
                                pace=1.0,
                                spec=mels.cuda(),
                                attn_prior=attn_prior.cuda(),
                                mel_lens=spec_len.cuda(),
                                input_lens=text_lens.cuda()
                            )

                            audio_from_text_dur = vocoder.convert_spectrogram_to_audio(spec=mels_pred)
                            audio_from_text_dur = audio_from_text_dur.to('cpu').numpy().flatten()

                            fname = os.path.join(args.out_dir, "originalModel_ForceAlignment_{}_{}.wav".format(speaker, vidx))
                            torchaudio.save(fname, torch.tensor(audio_from_text_dur, dtype=torch.float)[None], 44100)

                        
                        _, audio_from_text = infer(spec_model, vocoder, val_record['text'], speaker = _speaker)
                        fname = os.path.join(args.out_dir, "synthesized_{}-{}_{}_{}.wav".format(duration_mins, mix_str, speaker, vidx))
                        torchaudio.save(fname, torch.tensor(audio_from_text.flatten(), dtype=torch.float)[None], 44100)
                    
                    synth_phonemes_per_second = text_length/ (len(audio_from_text.flatten())/44100.0)
                    synth_phonmenes_per_second_list.append(synth_phonemes_per_second)
                    
                    print ("SYNTHESIZED FOR -- Speaker: {} | Dataset size: {} mins | Mixing:{} | Text: {}".format(speaker, duration_mins, mixing, val_record['text']))

                    if vidx+1 >= num_val:
                        break
                
                spk_rate_key = "{}_{}_{}".format(speaker, duration_mins, mix_str)
                speaking_rate_stats[spk_rate_key] = {
                    'gt_phonemes_per_sec' : float(np.mean(gt_phonmenes_per_second_list)),
                    'synthFA_phonemes_per_sec' : float(np.mean(synthFA_phonmenes_per_second_list)),
                    'synth_phonemes_per_sec' : float(np.mean(synth_phonmenes_per_second_list)),
                    'originalmodel_phonemes_per_sec' : float(np.mean(synthOriginalModel_phonmenes_per_second_list)),
                    'error' : float(np.mean(gt_phonmenes_per_second_list)) - float(np.mean(synth_phonmenes_per_second_list))
                }
                print ("Speaking rate stats")
                print(speaking_rate_stats)
                with open(os.path.join(args.out_dir, "speaking_rate.json"), 'w') as f:
                    f.write(json.dumps(speaking_rate_stats))


if __name__ == '__main__':
    main()