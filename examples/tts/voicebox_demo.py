import os
import soundfile as sf
import librosa
import numpy as np
from textgrid import TextGrid
from tqdm import tqdm, trange

import torch
from nemo.collections.tts.models.voicebox import VoiceboxModel


# get audio files from Meta's Voicebox demo page
# match sampling rate w/ our Voicebox vocoder
sampling_rate = 24000


# '4': "the" pronunced differently
ori_word_masks = {
    '0': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    '1': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    '2': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
    '3': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    '4': [1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    '5': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
}

tgt_word_masks = {
    '0': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    '1': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    '2': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    '3': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    '4': [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    '5': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
}

def get_audio_data(audio_path, device):
    # audio_data, orig_sr = sf.read(audio_path)
    audio_data, orig_sr = librosa.load(audio_path, sr=sampling_rate)
    audio_data = audio_data / max(np.abs(audio_data))
    audio = torch.tensor(audio_data, dtype=torch.float, device=device).unsqueeze(0)
    audio_len = torch.tensor(audio.shape[1], device=device).unsqueeze(0)
    return audio, audio_len, orig_sr

def get_textgrid_data(textgrid_path):
    tg = TextGrid()
    tg.read(textgrid_path)
    phn_dur = []
    word_dur = []
    for tier in tg.tiers:
        if tier.name == "words":
            for interval in tier.intervals:
                minTime = interval.minTime
                maxTime = interval.maxTime
                word = interval.mark
                word_dur.append((word, minTime, maxTime))
        if tier.name == "phones":
            for interval in tier.intervals:
                minTime = interval.minTime
                maxTime = interval.maxTime
                phoneme = interval.mark
                if phoneme == "":
                    phoneme = "sil"
                phn_dur.append((phoneme, minTime, maxTime))
    text = ' '.join([w for w, *_ in word_dur]).strip()
    return phn_dur, word_dur, text

def find_overlap(ori_word_dur, tgt_word_dur):
    ori_st, ori_ed = 0, len(ori_word_dur) - 1
    tgt_st, tgt_ed = 0, len(tgt_word_dur) - 1
    while True:
        if ori_st == len(ori_word_dur) - 1 or tgt_st == len(tgt_word_dur) - 1:
            break
        if ori_word_dur[ori_st][0] == "":
            ori_st += 1
            continue
        if tgt_word_dur[tgt_st][0] == "":
            tgt_st += 1
            continue
        if ori_word_dur[ori_st][0] == tgt_word_dur[tgt_st][0]:
            ori_st += 1
            tgt_st += 1
            continue
        else:
            break
    while True:
        if ori_ed == 0 or tgt_ed == 0:
            break
        if ori_word_dur[ori_ed][0] == "":
            ori_ed -= 1
            continue
        if tgt_word_dur[tgt_ed][0] == "":
            tgt_ed -= 1
            continue
        if ori_word_dur[ori_ed][0] == tgt_word_dur[tgt_ed][0]:
            ori_ed -= 1
            tgt_ed -= 1
            continue
        else:
            break
    ori_word_mask = [1 for _ in ori_word_dur]
    for i in range(len(ori_word_mask)):
        if i >= ori_st and i <= ori_ed:
            ori_word_mask[i] = 0
    print([(i, w) for i, (w, *_) in enumerate(ori_word_dur)])
    print([(i, w) for i, (w, *_) in enumerate(tgt_word_dur)])
    print([(i, w) for i, w in enumerate(ori_word_mask)])
    print(ori_word_mask)
    tgt_word_mask = [0 for _ in tgt_word_dur]
    for i in range(len(tgt_word_mask)):
        if i >= tgt_st and i <= tgt_ed:
            tgt_word_mask[i] = 1
    print([(i, w) for i, w in enumerate(tgt_word_mask)])
    print(tgt_word_mask)
    return ori_word_mask, tgt_word_mask

def word_mask_to_phn_mask(word_dur, phone_dur, word_mask):
    phn_mask = []
    p_id = 0
    for i, (w, ws, we) in enumerate(word_dur):
        p, ps, pe = phone_dur[p_id]
        while ps >= ws and pe <= we:
            phn_mask.append(word_mask[i])
            p_id += 1
            if p_id >= len(phone_dur):
                break
            p, ps, pe = phone_dur[p_id]
    return phn_mask

def combine_masked_dur(ori_word_dur, tgt_word_dur, ori_word_mask, tgt_word_mask):
    """can also use with phn dur/mask"""
    ori_dur = [ed - st for w, st, ed in ori_word_dur]
    tgt_dur = [ed - st for w, st, ed in tgt_word_dur]
    new_dur = []
    new_mask = []
    new_word_dur = []
    last_time = 0

    # find start of edit span
    ocut_st = ori_word_mask.index(0)
    tcut_st = tgt_word_mask.index(1)
    # find end of edit span
    ocut_ed = ori_word_mask.index(1, ocut_st)
    tcut_ed = tgt_word_mask.index(0, tcut_st)

    # pre-edit
    new_dur += ori_dur[:ocut_st]
    new_mask += [1] * ocut_st
    new_word_dur += ori_word_dur[:ocut_st]

    # mid-edit
    _time_shift_ = tgt_word_dur[tcut_st][1] - ori_word_dur[ocut_st][1]  # time_shift of start of edit
    new_dur += tgt_dur[tcut_st: tcut_ed]
    new_mask += [0] * (tcut_ed - tcut_st)
    new_word_dur += [(w, st-_time_shift_, ed-_time_shift_) for w, st, ed in tgt_word_dur[tcut_st: tcut_ed]]

    # post-edit
    _time_shift = ori_word_dur[ocut_ed-1][-1] - new_word_dur[-1][-1]    # time_shift of end of edit
    new_dur += ori_dur[ocut_ed:]
    new_mask += [1] * (len(ori_word_mask) - ocut_ed)
    new_word_dur += [(w, st-_time_shift, ed-_time_shift) for w, st, ed in ori_word_dur[ocut_ed:]]

    return new_word_dur, new_mask, new_dur

def audio_frame_mask_by_phn(ori_audio, model, ori_phn_dur, ori_phn_mask, new_phn_dur):
    """phn_dur: cumulated, (phn, st_time, ed_time)"""
    audio_enc_dec = model.voicebox.audio_enc_dec
    ori_mel = audio_enc_dec.encode(ori_audio)
    # mel_len = audio_len // downsample_factor
    ori_mel_len = ori_mel.shape[1]
    dur_ratio = ori_mel_len / ori_phn_dur[-1][-1]
    ori_st = ori_phn_mask.index(0)
    ori_ed = ori_phn_mask.index(1, ori_st)
    ori_st_pos = round(dur_ratio * ori_phn_dur[ori_st][-2])
    ori_ed_pos = round(dur_ratio * ori_phn_dur[ori_ed][-2])
    ori_ed_len = ori_mel_len - ori_ed_pos

    new_mel_len = round(dur_ratio * new_phn_dur[-1][-1])

    new_cond = torch.ones((1, new_mel_len, ori_mel.shape[2]), device=ori_mel.device) * 0
    new_cond[:, :ori_st_pos] = ori_mel[:, :ori_st_pos]
    new_cond[:, -ori_ed_len:] = ori_mel[:, -ori_ed_len:]
    new_frame_mask = torch.zeros((1, new_mel_len), device=ori_mel.device).int()
    new_frame_mask[:, :ori_st_pos] = 1
    new_frame_mask[:, -ori_ed_len:] = 1

    aligned_phonemes = []
    new_dur = []
    new_phoneme_ids = []
    for p, st, ed in new_phn_dur:
        aligned_phonemes += [p] * (round(dur_ratio*ed) - round(dur_ratio*st))
        new_dur.append(round(dur_ratio*ed) - round(dur_ratio*st))
        new_phoneme_ids.append(p)
    new_phoneme_ids = model.tokenizer.text_to_ids(new_phoneme_ids)[0]
    aligned_phoneme_ids = model.tokenizer.text_to_ids(aligned_phonemes)[0]
    return ori_mel, new_cond, new_frame_mask, new_phoneme_ids, aligned_phoneme_ids, new_dur

def get_edit_data(model, device, ori_audio_path, ori_textgrid_path, tgt_textgrid_path, ori_word_mask=None, tgt_word_mask=None):
    ori_audio, ori_audio_len, orig_sr = get_audio_data(ori_audio_path, device)
    ori_phn_dur, ori_word_dur, ori_text = get_textgrid_data(ori_textgrid_path)
    tgt_phn_dur, tgt_word_dur, tgt_text = get_textgrid_data(tgt_textgrid_path)
    if not ori_word_mask or not tgt_word_mask:
        ori_word_mask, tgt_word_mask = find_overlap(ori_word_dur=ori_word_dur, tgt_word_dur=tgt_word_dur)
        # new_word_dur, new_word_mask, _ = combine_masked_dur(ori_word_dur=ori_word_dur, tgt_word_dur=tgt_word_dur, ori_word_mask=ori_word_mask, tgt_word_mask=tgt_word_mask)
    ori_phn_mask = word_mask_to_phn_mask(ori_word_dur, ori_phn_dur, ori_word_mask)
    tgt_phn_mask = word_mask_to_phn_mask(tgt_word_dur, tgt_phn_dur, tgt_word_mask)
    new_phn_dur, new_phn_mask, _ = combine_masked_dur(ori_word_dur=ori_phn_dur, tgt_word_dur=tgt_phn_dur, ori_word_mask=ori_phn_mask, tgt_word_mask=tgt_phn_mask)
    ori_mel, new_cond, new_frame_mask, new_phoneme_ids, aligned_phoneme_ids, new_dur = audio_frame_mask_by_phn(ori_audio=ori_audio, model=model, ori_phn_dur=ori_phn_dur, ori_phn_mask=ori_phn_mask, new_phn_dur=new_phn_dur)
    return ori_audio, ori_mel, new_cond, new_frame_mask, torch.tensor(aligned_phoneme_ids).to(new_cond.device).unsqueeze(0), new_phn_dur, new_phn_mask

def get_data(model, corpus_dir, textgrid_dir, device):
    outputs = {}
    for spk in sorted(os.listdir(corpus_dir)):
        print(spk)
        ori_audio_path = f"{corpus_dir}/{spk}/{spk}.wav"
        ori_textgrid_path = f"{textgrid_dir}/{spk}/{spk}.TextGrid"
        tgt_textgrid_path = f"{textgrid_dir}/{spk}/{spk}_0.TextGrid"
        ori_word_mask = ori_word_masks[spk]
        tgt_word_mask = tgt_word_masks[spk]
        outputs[spk] = get_edit_data(model, device, ori_audio_path, ori_textgrid_path, tgt_textgrid_path, ori_word_mask, tgt_word_mask)
    return outputs



if __name__ == "__main__":
    # ckpt_path = "nemo_experiments/local-1/2023-12-14_03-54-42/checkpoints/local-1--val_loss_total=9.0312-epoch=20-last.ckpt"
    # ckpt_path = "nemo_experiments/ngc/2023-12-15_16-41-45/checkpoints/ngc--val_loss_total=3.7990-epoch=20.ckpt"
    ckpt_path = "nemo_experiments/ngc/2023-12-15_16-41-45/checkpoints/ngc--val_loss_total=3.7464-epoch=26.ckpt"
    corpus_dir = "/datasets/LibriLight_aligned/raw_data_cuts/demo"
    textgrid_dir = "/datasets/LibriLight_aligned/textgrids/demo"
    out_dir = "nemo_experiments/edit_demo"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VoiceboxModel.load_from_checkpoint(ckpt_path, map_location='cpu')
    model = model.to(device)

    outputs = get_data(model, corpus_dir, textgrid_dir, device)

    for spk in tqdm(outputs):
        _ori_audio, ori_mel, new_cond, new_frame_mask, aligned_phoneme_ids, *_ = outputs[spk]
        print(spk)
        print(new_cond.shape, new_frame_mask.shape, aligned_phoneme_ids.shape)

        # masked_audio = model.voicebox.audio_enc_dec.decode(new_cond)
        # masked_audio = masked_audio[0].cpu().numpy()
        # masked_audio = masked_audio / max(np.abs(masked_audio))
        # sf.write(f"{out_dir}/{spk}_0_masked.wav", masked_audio, sampling_rate, format='WAV')

        # ori_audio = model.voicebox.audio_enc_dec.decode(ori_mel)
        ori_audio = _ori_audio
        ori_audio = ori_audio[0].cpu().numpy()
        sf.write(f"{out_dir}/{spk}_0.wav", ori_audio, sampling_rate, format='WAV')

        for i in trange(100):
            output_audio = model.cfm_wrapper.sample(
                cond=new_cond,
                # phoneme_ids=torch.tensor(new_phoneme_ids).to(new_cond.device),
                aligned_phoneme_ids=aligned_phoneme_ids,
                cond_mask=None,
                steps=20,
            )
            # print(output_audio.shape)
            output_audio = output_audio[0].cpu().numpy()
            output_audio = output_audio / max(np.abs(output_audio)) / 2
            sf.write(f"{out_dir}/{spk}_1.wav", output_audio, sampling_rate, format='WAV')
