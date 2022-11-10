import argparse
import os

import librosa
import soundfile
import torch

from nemo.collections.asr.parts.preprocessing.features import WaveformFeaturizer
from nemo.collections.tts.models import fastpitch_ssl, hifigan, ssl_tts
from nemo.collections.tts.torch.helpers import get_base_dir


def load_wav(wav_path, wav_featurizer, pad_multiple=1024):
    wav = wav_featurizer.process(wav_path)
    if wav.shape[0 % pad_multiple] != 0:
        wav = torch.cat([wav, torch.zeros(pad_multiple - wav.shape[0] % pad_multiple, dtype=torch.float)])
    wav = wav[:-1]

    return wav


def get_pitch_contour(wav, pitch_mean=None, pitch_std=None, compute_mean_std=False):
    f0, _, _ = librosa.pyin(
        wav.numpy(),
        fmin=librosa.note_to_hz('C2'),
        fmax=librosa.note_to_hz('C7'),
        frame_length=1024,
        hop_length=256,
        sr=22050,
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


def segment_wav(wav, segment_length=44100, hop_size=44100, min_segment_size=22050):
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


def get_speaker_embedding(ssl_model, wav_featurizer, audio_paths, duration=None, device="cpu"):
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
    print("signal batch", signal_batch.shape)
    # print("signal batch", signal_batch.shape)
    signal_length_batch = torch.stack([torch.tensor(signal_batch.shape[1]) for _i in range(len(all_segments))])
    # print("signal length", signal_length_batch.shape)
    signal_batch = signal_batch.to(device)
    signal_length_batch = signal_length_batch.to(device)
    _, speaker_embeddings, _, _, _ = ssl_model.forward_for_export(
        input_signal=signal_batch, input_signal_length=signal_length_batch, normalize_content=True
    )

    speaker_embedding = torch.mean(speaker_embeddings, dim=0)
    l2_norm = torch.norm(speaker_embedding, p=2)
    speaker_embedding = speaker_embedding / l2_norm

    return speaker_embedding[None]


def get_ssl_features_disentsngled(
    ssl_model, wav_featurizer, audio_path, emb_type="embedding_and_probs", use_unique_tokens=False, device="cpu"
):
    wav = load_wav(audio_path, wav_featurizer)
    audio_signal = wav[None]
    audio_signal_length = torch.tensor([wav.shape[0]])
    audio_signal = audio_signal.to(device)
    audio_signal_length = audio_signal_length.to(device)
    _, speaker_embedding, content_embedding, content_log_probs, encoded_len = ssl_model.forward_for_export(
        input_signal=audio_signal, input_signal_length=audio_signal_length, normalize_content=True
    )

    content_embedding = content_embedding[0, : encoded_len[0].item()]
    content_log_probs = content_log_probs[: encoded_len[0].item(), 0, :]
    content_embedding = content_embedding.t()
    content_log_probs = content_log_probs.t()
    content_probs = torch.exp(content_log_probs)

    if emb_type == "probs":
        final_content_embedding = content_probs

    elif emb_type == "embedding":
        final_content_embedding = content_embedding

    elif emb_type == "log_probs":
        final_content_embedding = content_log_probs

    elif emb_type == "embedding_and_probs":
        final_content_embedding = torch.cat([content_embedding, content_probs], dim=0)

    duration = torch.ones(final_content_embedding.shape[1]) * 4.0
    if use_unique_tokens:
        token_predictions = torch.argmax(content_probs, dim=0)
        # print("token predictions:", token_predictions)
        content_buffer = [final_content_embedding[:, 0]]
        unique_content_embeddings = []
        unique_tokens = []
        durations = []
        for _t in range(1, final_content_embedding.shape[1]):
            if token_predictions[_t] == token_predictions[_t - 1]:
                content_buffer.append(final_content_embedding[:, _t])
            else:
                durations.append(len(content_buffer) * 4)
                unique_content_embeddings.append(torch.mean(torch.stack(content_buffer), dim=0))
                content_buffer = [final_content_embedding[:, _t]]
                unique_tokens.append(token_predictions[_t].item())

        if len(content_buffer) > 0:
            durations.append(len(content_buffer) * 4)
            unique_content_embeddings.append(torch.mean(torch.stack(content_buffer), dim=0))
            unique_tokens.append(token_predictions[_t].item())

        unique_content_embedding = torch.stack(unique_content_embeddings)
        final_content_embedding = unique_content_embedding.t()
        duration = torch.tensor(durations).float()

    duration = duration.to(device)
    return final_content_embedding[None], speaker_embedding, duration[None]


def main():
    parser = argparse.ArgumentParser(description='Evaluate the model')
    parser.add_argument('--ssl_model_ckpt_path', type=str)
    parser.add_argument('--hifi_ckpt_path', type=str)
    parser.add_argument('--fastpitch_ckpt_path', type=str)
    parser.add_argument('--source_audio_path', type=str)
    parser.add_argument('--target_audio_path', type=str)  # can be a list seperated by comma
    parser.add_argument('--out_path', type=str)
    parser.add_argument('--source_target_out_pairs', type=str)
    parser.add_argument('--use_unique_tokens', type=int, default=0)
    parser.add_argument('--compute_pitch', type=int, default=0)
    parser.add_argument('--compute_duration', type=int, default=0)
    args = parser.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    if args.source_target_out_pairs is not None:
        assert args.source_audio_path is None, "source_audio_path and source_target_out_pairs are mutually exclusive"
        assert args.target_audio_path is None, "target_audio_path and source_target_out_pairs are mutually exclusive"
        assert args.out_path is None, "out_path and source_target_out_pairs are mutually exclusive"
        with open(args.source_target_out_pairs, "r") as f:
            lines = f.readlines()
            source_target_out_pairs = [line.strip().split(";") for line in lines]
    else:
        assert args.source_audio_path is not None, "source_audio_path is required"
        assert args.target_audio_path is not None, "target_audio_path is required"
        if args.out_path is None:
            source_name = os.path.basename(args.source_audio_path).split(".")[0]
            target_name = os.path.basename(args.target_audio_path).split(".")[0]
            args.out_path = "swapped_{}_{}.wav".format(source_name, target_name)

        source_target_out_pairs = [(args.source_audio_path, args.target_audio_path, args.out_path)]

    out_paths = [r[2] for r in source_target_out_pairs]
    out_dir = get_base_dir(out_paths)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    ssl_model = ssl_tts.SSLDisentangler.load_from_checkpoint(args.ssl_model_ckpt_path, strict=False)
    ssl_model = ssl_model.to(device)
    ssl_model.eval()

    vocoder = hifigan.HifiGanModel.load_from_checkpoint(args.hifi_ckpt_path).to(device)
    vocoder.eval()

    fastpitch_model = fastpitch_ssl.FastPitchModel_SSL.load_from_checkpoint(args.fastpitch_ckpt_path, strict=False)
    fastpitch_model = fastpitch_model.to(device)
    fastpitch_model.eval()
    fastpitch_model.non_trainable_models = {'vocoder': vocoder}

    wav_featurizer = WaveformFeaturizer(sample_rate=22050, int_values=False, augmentor=None)

    use_unique_tokens = args.use_unique_tokens == 1
    compute_pitch = args.compute_pitch == 1
    compute_duration = args.compute_duration == 1

    for source_target_out in source_target_out_pairs:
        source_audio_path = source_target_out[0]
        target_audio_paths = source_target_out[1].split(",")
        out_path = source_target_out[2]

        with torch.no_grad():
            content_embedding1, _, duration1 = get_ssl_features_disentsngled(
                ssl_model,
                wav_featurizer,
                source_audio_path,
                emb_type="embedding_and_probs",
                use_unique_tokens=use_unique_tokens,
                device=device,
            )

            speaker_embedding2 = get_speaker_embedding(
                ssl_model, wav_featurizer, target_audio_paths, duration=None, device=device
            )

            pitch_contour1 = None
            if not compute_pitch:
                pitch_contour1 = get_pitch_contour(load_wav(source_audio_path, wav_featurizer), compute_mean_std=True)[
                    None
                ]
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
            wav_generated = wav_generated[0][0]
            soundfile.write(out_path, wav_generated, 22050)


if __name__ == "__main__":
    main()
