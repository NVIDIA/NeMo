# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Example Run Command: python self_vc.py --ssl_model_ckpt_path <PATH TO CKPT> --hifi_ckpt_path <PATH TO CKPT> \
# --fastpitch_ckpt_path <PATH TO CKPT> --source_audio_path <SOURCE CONTENT WAV PATH> --target_audio_path \
# <TARGET SPEAKER WAV PATH> --out_path <PATH TO OUTPUT WAV>

import argparse
import os

import librosa
import soundfile
import torch

import nemo.collections.asr as nemo_asr
from nemo.collections.asr.models import label_models
from nemo.collections.asr.parts.preprocessing.features import WaveformFeaturizer
from nemo.collections.tts.models import fastpitch_ssl, hifigan, ssl_tts
from nemo.collections.tts.parts.utils.tts_dataset_utils import get_base_dir


def load_wav(wav_path, wav_featurizer, pad_multiple=1024):
    wav = wav_featurizer.process(wav_path)
    # if wav is multi-channel, take the mean
    if wav.ndim > 1:
        wav = wav.mean(axis=1)

    if (wav.shape[0] % pad_multiple) != 0:
        wav = torch.cat([wav, torch.zeros(pad_multiple - wav.shape[0] % pad_multiple, dtype=torch.float)])
    wav = wav[:-1]

    return wav


def get_pitch_contour(wav, pitch_mean=None, pitch_std=None, compute_mean_std=False, sample_rate=22050):
    f0, _, _ = librosa.pyin(
        wav.numpy(),
        fmin=librosa.note_to_hz('C2'),
        fmax=librosa.note_to_hz('C7'),
        frame_length=1024,
        hop_length=256,
        sr=sample_rate,
        center=True,
        fill_na=0.0,
    )
    pitch_contour = torch.tensor(f0, dtype=torch.float32)
    _pitch_mean = pitch_contour.mean().item()
    _pitch_std = pitch_contour.std().item()
    if compute_mean_std:
        pitch_mean = _pitch_mean
        pitch_std = _pitch_std
    if (pitch_mean is not None) and (pitch_std is not None):
        pitch_contour = pitch_contour - pitch_mean
        pitch_contour[pitch_contour == -pitch_mean] = 0.0
        pitch_contour = pitch_contour / pitch_std

    return pitch_contour


def segment_wav(wav, segment_length=32000, hop_size=16000, min_segment_size=16000):
    if len(wav) < segment_length:
        pad = torch.zeros(segment_length - len(wav))
        segment = torch.cat([wav, pad])
        return [segment]
    else:
        si = 0
        segments = []
        while si < len(wav) - min_segment_size:
            segment = wav[si : si + segment_length]
            if len(segment) < segment_length:
                pad = torch.zeros(segment_length - len(segment))
                segment = torch.cat([segment, pad])

            segments.append(segment)
            si += hop_size
        return segments


def get_speaker_embedding(nemo_sv_model, wav_featurizer, audio_paths, duration=None, device="cpu"):
    all_segments = []
    all_wavs = []
    for audio_path in audio_paths:
        wav = load_wav(audio_path, wav_featurizer)
        segments = segment_wav(wav)
        all_segments += segments
        all_wavs.append(wav)
        if duration is not None and len(all_segments) >= duration:
            # each segment is 2 seconds with one second overlap.
            # so 10 segments would mean 0 to 2, 1 to 3.. 9 to 11 (11 seconds.)
            all_segments = all_segments[: int(duration)]
            break

    signal_batch = torch.stack(all_segments)
    signal_length_batch = torch.stack([torch.tensor(signal_batch.shape[1]) for _ in range(len(all_segments))])
    signal_batch = signal_batch.to(device)
    signal_length_batch = signal_length_batch.to(device)

    _, speaker_embeddings = nemo_sv_model(input_signal=signal_batch, input_signal_length=signal_length_batch)
    speaker_embedding = torch.mean(speaker_embeddings, dim=0)
    l2_norm = torch.norm(speaker_embedding, p=2)
    speaker_embedding = speaker_embedding / l2_norm

    return speaker_embedding[None]


def group_content_embeddings(content_embedding, duration, emb_similarity_threshold=0.925):
    # content_embedding: (256, n_timesteps)
    grouped_content_embeddings = [ content_embedding[:, 0] ]
    grouped_durations = [ duration[0] ]
    group_size = 1
    for _tidx in range(1, content_embedding.shape[1]):
        prev_embedding = grouped_content_embeddings[-1]
        curr_embedding = content_embedding[:, _tidx]
        emb_similarity = torch.cosine_similarity(prev_embedding, curr_embedding, dim=0)
        if emb_similarity < emb_similarity_threshold:
            grouped_content_embeddings.append(curr_embedding)
            grouped_durations.append(duration[_tidx])
        else:
            # group with previous embedding
            grouped_content_embeddings[-1] = (grouped_content_embeddings[-1] * group_size + curr_embedding) / (group_size + 1)
            grouped_durations[-1] += duration[_tidx]
    
    grouped_content_embeddings = torch.stack(grouped_content_embeddings, dim=1)
    grouped_durations = torch.stack(grouped_durations, dim=0)

    return grouped_content_embeddings, grouped_durations

def get_ssl_features_disentangled(ssl_model, wav_featurizer, audio_path, use_unique_tokens=False, device="cpu"):
    """
    Extracts content embedding, speaker embedding and duration tokens to be used as inputs for FastPitchModel_SSL 
    synthesizer. Content embedding and speaker embedding extracted using SSLDisentangler model.
    Args:
        ssl_model: SSLDisentangler model
        wav_featurizer: WaveformFeaturizer object
        audio_path: path to audio file
        device: device to run the model on
    Returns:
        content_embedding, speaker_embedding, duration
    """
    wav = load_wav(audio_path, wav_featurizer)
    audio_signal = wav[None]
    audio_signal_length = torch.tensor([wav.shape[0]])
    audio_signal = audio_signal.to(device)
    audio_signal_length = audio_signal_length.to(device)

    processed_signal, processed_signal_length = ssl_model.preprocessor(
        input_signal=audio_signal, length=audio_signal_length,
    )

    batch_content_embedding, batch_encoded_len = ssl_model.encoder(
        audio_signal=processed_signal, length=processed_signal_length
    )
    if ssl_model._cfg.get("normalize_content_encoding", False):
        batch_content_embedding = ssl_model._normalize_encoding(batch_content_embedding)

    content_embedding = batch_content_embedding[0, :, : batch_encoded_len[0]]
    ssl_downsampling_factor = ssl_model._cfg.encoder.subsampling_factor
    duration = torch.ones(content_embedding.shape[1]) * ssl_downsampling_factor
    
    if use_unique_tokens:
        print("Grouping..")
        emb_similarity_threshold = ssl_model._cfg.get("emb_similarity_threshold", 0.925)
        final_content_embedding, final_duration = group_content_embeddings(content_embedding, duration, emb_similarity_threshold)
        print("Grouped duration", final_duration)
    else:
        final_content_embedding, final_duration = content_embedding, duration
        
    final_content_embedding = final_content_embedding.to(device)
    final_duration = final_duration.to(device)

    return final_content_embedding[None], final_duration[None]


def main():
    parser = argparse.ArgumentParser(description='Evaluate the model')
    parser.add_argument('--ssl_model_ckpt_path', type=str)
    parser.add_argument('--hifi_ckpt_path', type=str)
    parser.add_argument('--fastpitch_ckpt_path', type=str)
    parser.add_argument('--source_audio_path', type=str)
    parser.add_argument('--target_audio_path', type=str)  # can be a list seperated by comma
    parser.add_argument('--out_path', type=str)
    parser.add_argument('--source_target_out_pairs', type=str)
    parser.add_argument('--compute_pitch', type=int, default=1)
    parser.add_argument('--compute_duration', type=int, default=1)
    parser.add_argument('--max_input_length_sec', type=int, default=20)
    parser.add_argument('--segment_length_seconds', type=int, default=16)
    parser.add_argument('--use_unique_tokens', type=int, default=0)
    parser.add_argument('--duration', type=float, default=None)
    parser.add_argument('--sample_rate', type=float, default=22050)
    args = parser.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    combine_outpaths = False
    if args.source_target_out_pairs is not None:
        assert args.source_audio_path is None, "source_audio_path and source_target_out_pairs are mutually exclusive"
        assert args.target_audio_path is None, "target_audio_path and source_target_out_pairs are mutually exclusive"
        assert args.out_path is None, "out_path and source_target_out_pairs are mutually exclusive"
        with open(args.source_target_out_pairs, "r") as f:
            lines = f.readlines()
            source_target_out_pairs = [line.strip().split(";") for line in lines]
            source_target_out_pairs = [[r.strip() for r in pair] for pair in source_target_out_pairs]
    else:
        assert args.source_audio_path is not None, "source_audio_path is required"
        assert args.target_audio_path is not None, "target_audio_path is required"
        if args.out_path is None:
            source_name = os.path.basename(args.source_audio_path).split(".")[0]
            target_name = os.path.basename(args.target_audio_path).split(".")[0]
            args.out_path = "swapped_{}_{}.wav".format(source_name, target_name)
        
        _wav_featurizer = WaveformFeaturizer(sample_rate=args.sample_rate, int_values=False, augmentor=None)
        source_audio_wav = _wav_featurizer.process(args.source_audio_path)
        source_audio_length = source_audio_wav.shape[0]
        if source_audio_length > args.max_input_length_sec * args.sample_rate:
            print("Segmenting the long source audio into chunks")
            # break audio into segments
            source_audio_basedir = os.path.dirname(args.source_audio_path)
            temp_dir = os.path.join(source_audio_basedir, "temp")
            if not os.path.exists(temp_dir):
                os.makedirs(temp_dir)
            
            # segment_length is nearest multiple of 1024
            segment_length = args.segment_length_seconds * int(args.sample_rate / 1024) * 1024
            si = 0
            seg_num = 0
            source_target_out_pairs = []
            combine_outpaths = True
            while si < source_audio_length:
                segment = source_audio_wav[si : si + segment_length]
                segment_path = os.path.join(temp_dir, "source_{}.wav".format(seg_num))
                segment_outpath = os.path.join(temp_dir, "out_{}.wav".format(seg_num))
                soundfile.write(segment_path, segment, 22050)
                si += segment_length
                source_target_out_pairs.append((segment_path, args.target_audio_path, segment_outpath))
                seg_num += 1
        else:
            source_target_out_pairs = [(args.source_audio_path, args.target_audio_path, args.out_path)]

    out_paths = [r[2] for r in source_target_out_pairs]
    out_dir = get_base_dir(out_paths)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    ssl_model = nemo_asr.models.ssl_models.SpeechEncDecSelfSupervisedModel.load_from_checkpoint(
        args.ssl_model_ckpt_path,
        map_location=device,
    )
    ssl_model.eval()
    ssl_model.to(device)

    nemo_sv_model = label_models.EncDecSpeakerLabelModel.from_pretrained("titanet_large")
    nemo_sv_model = nemo_sv_model.to(device)
    nemo_sv_model.eval()
    sv_sample_rate = nemo_sv_model._cfg.preprocessor.sample_rate

    vocoder = hifigan.HifiGanModel.load_from_checkpoint(args.hifi_ckpt_path, map_location=device).to(device)
    vocoder.eval()

    fastpitch_model = fastpitch_ssl.FastPitchModel_SSL.load_from_checkpoint(args.fastpitch_ckpt_path, strict=False, map_location=device)
    fastpitch_model = fastpitch_model.to(device)
    fastpitch_model.eval()
    fastpitch_model.non_trainable_models = {'vocoder': vocoder}
    fpssl_sample_rate = fastpitch_model._cfg.sample_rate

    wav_featurizer = WaveformFeaturizer(sample_rate=fpssl_sample_rate, int_values=False, augmentor=None)
    wav_featurizer_sv = WaveformFeaturizer(sample_rate=sv_sample_rate, int_values=False, augmentor=None)

    use_unique_tokens = args.use_unique_tokens == 1
    compute_pitch = args.compute_pitch == 1
    compute_duration = args.compute_duration == 1


    for pidx, source_target_out in enumerate(source_target_out_pairs):
        print("Processing pair {}/{}".format(pidx + 1, len(source_target_out_pairs)))
        source_audio_path = source_target_out[0]
        source_audio_length = wav_featurizer.process(source_audio_path).shape[0]
        target_audio_paths = source_target_out[1].split(",")
        out_path = source_target_out[2]

        with torch.no_grad():
            content_embedding1, duration1 = get_ssl_features_disentangled(
                ssl_model, wav_featurizer, source_audio_path, use_unique_tokens, device=device,
            )

            speaker_embedding2 = get_speaker_embedding(
                nemo_sv_model, wav_featurizer_sv, target_audio_paths, duration=args.duration, device=device
            )

            pitch_contour1 = None
            if not compute_pitch:
                pitch_contour1 = get_pitch_contour(
                    load_wav(source_audio_path, wav_featurizer), compute_mean_std=True, sample_rate=fpssl_sample_rate
                )[None]
                pitch_contour1 = pitch_contour1.to(device)

            wav_generated = fastpitch_model.generate_wav(
                content_embedding1,
                speaker_embedding2,
                pitch_contour=pitch_contour1,
                compute_pitch=compute_pitch,
                compute_duration=compute_duration,
                durs_gt=duration1,
                dataset_id=0,
            )
            wav_generated = wav_generated[0][0][:source_audio_length]
            soundfile.write(out_path, wav_generated, fpssl_sample_rate)

    if combine_outpaths:
        print("Combining segments into one file")
        out_paths = [r[2] for r in source_target_out_pairs]
        out_wavs = [wav_featurizer.process(out_path) for out_path in out_paths]
        out_wav = torch.cat(out_wavs, dim=0).cpu().numpy()
        soundfile.write(args.out_path, out_wav, fpssl_sample_rate)


if __name__ == "__main__":
    main()