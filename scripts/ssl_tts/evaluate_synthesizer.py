import argparse
import fnmatch
import json
import os
import pickle
import random

import librosa
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import soundfile
import torch
import torch.nn.functional as F
import torchaudio
from numpy import dot
from numpy.linalg import norm
from scipy import linalg
from sklearn.manifold import TSNE
from sklearn.metrics import auc, roc_curve
from tqdm import tqdm
from transformers import Wav2Vec2Model, Wav2Vec2Processor

from nemo.collections.asr.models import label_models
from nemo.collections.asr.parts.preprocessing.features import WaveformFeaturizer
from nemo.collections.tts.models import fastpitch_ssl, hifigan, hifigan_ssl, ssl_tts

plt.rcParams["figure.figsize"] = (10, 10)
device = "cpu"

fid_wave_model = WaveformFeaturizer(sample_rate=16000)
fid_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
fid_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large-xlsr-53")


def load_wav(wav_path, wav_featurizer, pad_multiple=1024):
    wav = wav_featurizer.process(wav_path)
    if wav.shape[0 % pad_multiple] != 0:
        wav = torch.cat([wav, torch.zeros(pad_multiple - wav.shape[0] % pad_multiple, dtype=torch.float)])
    wav = wav[:-1]

    return wav


def get_pitch_contour(wav, pitch_mean=None, pitch_std=None):
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
    if (pitch_mean is not None) and (pitch_std is not None):
        pitch_contour = pitch_contour - pitch_mean
        pitch_contour[pitch_contour == -pitch_mean] = 0.0
        pitch_contour = pitch_contour / pitch_std

    return pitch_contour


def get_speaker_stats(ssl_model, wav_featurizer, audio_paths, duration):
    all_segments = []
    all_wavs = []
    for audio_path in audio_paths:
        wav = load_wav(audio_path, wav_featurizer)
        segments = segment_wav(wav)
        all_segments += segments
        all_wavs.append(wav)

        if len(all_segments) >= duration:
            # each segment is 2 seconds with one second overlap.
            # so 10 segments would mean 0 to 2, 1 to 3.. 9 to 11 (11 seconds.)
            break

    signal_batch = torch.stack(all_segments)
    # print("signal batch", signal_batch.shape)
    signal_length_batch = torch.stack([torch.tensor(signal_batch.shape[1]) for _i in range(len(all_segments))])
    # print("signal length", signal_length_batch.shape)
    _, speaker_embeddings, _, _, _ = ssl_model.forward_for_export(
        input_signal=signal_batch, input_signal_length=signal_length_batch, normalize_content=True
    )

    speaker_embedding = torch.mean(speaker_embeddings, dim=0)
    l2_norm = torch.norm(speaker_embedding, p=2)
    speaker_embedding = speaker_embedding / l2_norm
    non_zero_pc = []
    for wav in all_wavs:
        pitch_contour = get_pitch_contour(wav)
        pitch_contour_nonzero = pitch_contour[pitch_contour != 0]
        non_zero_pc.append(pitch_contour_nonzero)

    non_zero_pc = torch.cat(non_zero_pc)
    if len(non_zero_pc) > 0:
        pitch_mean = non_zero_pc.mean().item()
        pitch_std = non_zero_pc.std().item()
    else:
        print("could not find pitch contour")
        pitch_mean = 212.0
        pitch_std = 70.0

    return speaker_embedding[None], pitch_mean, pitch_std


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


def load_speaker_stats(
    speaker_wise_paths,
    ssl_model,
    wav_featurizer,
    manifest_name,
    duration_per_speaker=10,
    pitch_stats_json=None,
    recache=False,
    out_dir=".",
):
    pickle_path = os.path.join(out_dir, "{}_speaker_stats.pkl".format(manifest_name))
    if os.path.exists(pickle_path) and not recache:
        with open(pickle_path, 'rb') as f:
            return pickle.load(f)

    speaker_stats = {}
    pitch_stats = {}
    if pitch_stats_json is not None:
        with open(pitch_stats_json, "r") as f:
            pitch_stats = json.loads(f.read())

    for speaker in speaker_wise_paths:
        print("computing stats for {}".format(speaker))
        speaker_embedding, pitch_mean, pitch_std = get_speaker_stats(
            ssl_model, wav_featurizer, speaker_wise_paths[speaker], duration_per_speaker
        )
        speaker_stats[speaker] = {
            'speaker_embedding': speaker_embedding,
            'pitch_mean': pitch_mean,
            'pitch_std': pitch_std,
        }
        if str(speaker) in pitch_stats:
            speaker_stats[speaker]["pitch_mean"] = pitch_stats[str(speaker)]["pitch_mean"]
            speaker_stats[speaker]["pitch_std"] = pitch_stats[str(speaker)]["pitch_std"]

    with open(pickle_path, 'wb') as f:
        pickle.dump(speaker_stats, f)

    return speaker_stats


def get_ssl_features_disentsngled(
    ssl_model, wav_featurizer, audio_path, emb_type="embedding_and_probs", use_unique_tokens=False
):
    wav = load_wav(audio_path, wav_featurizer)
    audio_signal = wav[None]
    audio_signal_length = torch.tensor([wav.shape[0]])
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

    return final_content_embedding[None], speaker_embedding, duration[None]


def mscatter(x, y, ax=None, m=None, **kw):
    import matplotlib.markers as mmarkers

    # ax = ax or plt.gca()
    sc = plt.scatter(x, y, **kw)
    if (m is not None) and (len(m) == len(x)):
        paths = []
        for marker in m:
            if isinstance(marker, mmarkers.MarkerStyle):
                marker_obj = marker
            else:
                marker_obj = mmarkers.MarkerStyle(marker)
            path = marker_obj.get_path().transformed(marker_obj.get_transform())
            paths.append(path)
        sc.set_paths(paths)
    return sc


def visualize_embeddings(embedding_dict_np, title="TSNE", out_dir=".", out_file="tsne.png"):
    """
    Arguments:
    embedding_dict_np : Dictionary with keys as speaker ids/labels and value as list of np arrays (embeddings).
    """
    plt.clf()
    color = []
    marker_shape = []
    universal_embed_list = []
    handle_list = []
    _unique_speakers = {}
    marker_list = ['<', '*', 'h', 'X', 's', 'H', 'D', 'd', 'P', 'v', '^', '>', '8', 'p']
    for kidx, key in enumerate(embedding_dict_np.keys()):
        universal_embed_list += embedding_dict_np[key]
        _num_samples = len(embedding_dict_np[key])
        id_color = plt.cm.tab20(kidx)
        _color = [id_color] * _num_samples
        color += _color
        _marker_shape = [marker_list[kidx % len(marker_list)]] * _num_samples
        marker_shape += _marker_shape
        _label = key
        handle_list.append(mpatches.Patch(color=id_color, label=_label))
    speaker_embeddings = TSNE(n_components=2, random_state=0).fit_transform(universal_embed_list)
    mscatter(speaker_embeddings[:, 0], speaker_embeddings[:, 1], m=marker_shape, c=color, s=20)
    plt.legend(handles=handle_list, title="Type")
    plt.title(title)

    plt.savefig(os.path.join(out_dir, out_file))


def swap_speakers(
    fastpitch_model,
    ssl_model,
    wav_featurizer,
    speaker_stats,
    speaker_wise_paths,
    out_dir,
    spk1,
    spk2,
    pitch_conditioning=True,
    compute_pitch=False,
    compute_duration=False,
    use_unique_tokens=False,
    dataset_id=0,
):
    """
    source speaker is spk1
    target speaker is spk2
    """
    swapped_wav_paths = []
    print("swapping speakers:", spk1, "and", spk2)
    for _idx in range(len(speaker_wise_paths[spk1])):
        wav_name = "{}_{}_{}.wav".format(spk1, spk2, _idx)
        wav_name_swapped = "swapped_" + wav_name
        wav_path_swapped = os.path.join(out_dir, wav_name_swapped)

        if os.path.exists(wav_path_swapped):
            swapped_wav_paths.append(wav_path_swapped)
            continue

        wav_path1 = speaker_wise_paths[spk1][_idx]
        speaker_embedding2 = speaker_stats[spk2]["speaker_embedding"]

        content_embedding1, _, duration1 = get_ssl_features_disentsngled(
            ssl_model, wav_featurizer, wav_path1, emb_type="embedding_and_probs", use_unique_tokens=use_unique_tokens
        )

        pitch_contour1 = get_pitch_contour(
            load_wav(wav_path1, wav_featurizer),
            pitch_mean=speaker_stats[spk1]["pitch_mean"],
            pitch_std=speaker_stats[spk1]["pitch_std"],
        )[None]

        wav_generated = fastpitch_model.synthesize_wav(
            content_embedding1,
            speaker_embedding2,
            pitch_contour=pitch_contour1,
            compute_pitch=compute_pitch,
            compute_duration=compute_duration,
            durs_gt=duration1,
            dataset_id=dataset_id,
        )
        wav_generated = wav_generated[0][0]

        soundfile.write(wav_path_swapped, wav_generated, 22050)
        swapped_wav_paths.append(wav_path_swapped)

    return swapped_wav_paths


def reconstruct_audio(
    fastpitch_model,
    ssl_model,
    wav_featurizer,
    speaker_stats,
    speaker_wise_paths,
    out_dir,
    pitch_conditioning=True,
    compute_pitch=False,
    compute_duration=False,
    use_unique_tokens=False,
    dataset_id=0,
):
    spk_count = 0
    reconstructed_file_paths = {}
    for speaker in speaker_wise_paths:
        reconstructed_file_paths[speaker] = []
        print("reconstructing audio for {}".format(speaker))
        spk_count += 1
        speaker_stat = speaker_stats[speaker]
        for widx, wav_path in enumerate(speaker_wise_paths[speaker]):
            wav_name = "{}_{}.wav".format(speaker, widx)
            wav_name_original = "original_" + wav_name
            wav_path_original = os.path.join(out_dir, wav_name_original)

            wav_name_reconstructed = "reconstructed_" + wav_name
            wav_path_reconstructed = os.path.join(out_dir, wav_name_reconstructed)

            if os.path.exists(wav_path_reconstructed):
                reconstructed_file_paths[speaker].append(wav_path_reconstructed)
                continue

            content_embedding, _, duration = get_ssl_features_disentsngled(
                ssl_model,
                wav_featurizer,
                wav_path,
                emb_type="embedding_and_probs",
                use_unique_tokens=use_unique_tokens,
            )
            pitch_contour = get_pitch_contour(
                load_wav(wav_path, wav_featurizer),
                pitch_mean=speaker_stat["pitch_mean"],
                pitch_std=speaker_stat["pitch_std"],
            )[None]

            wav_original = load_wav(wav_path, wav_featurizer)

            soundfile.write(wav_path_original, wav_original.cpu().numpy(), 22050)

            with torch.no_grad():
                print("Reconstructed Audio Speaker {}".format(speaker))
                wav_generated = fastpitch_model.synthesize_wav(
                    content_embedding,
                    speaker_stat['speaker_embedding'],
                    pitch_contour=pitch_contour,
                    compute_pitch=compute_pitch,
                    compute_duration=compute_duration,
                    durs_gt=duration,
                    dataset_id=dataset_id,
                )
                wav_generated = wav_generated[0][0]
                wav_name_reconstructed = "reconstructed_" + wav_name
                print("Wav generated", wav_generated.shape)
                soundfile.write(wav_path_reconstructed, wav_generated, 22050)
                reconstructed_file_paths[speaker].append(wav_path_reconstructed)

    return reconstructed_file_paths


def get_similarity(emb1, emb2):
    similarity = dot(emb1, emb2) / (norm(emb1) * norm(emb2))
    return similarity


def calculate_eer(speaker_embeddings, mode="generated"):
    generated_embeddings = {}
    real_embeddings = {}
    for key in speaker_embeddings:
        speaker = key.split("_")[1]
        if speaker not in generated_embeddings:
            generated_embeddings[speaker] = []
        if speaker not in real_embeddings:
            real_embeddings[speaker] = []
        if "generated" in key:
            generated_embeddings[speaker] += speaker_embeddings[key]
        else:
            real_embeddings[speaker] += speaker_embeddings[key]

    y_score = []
    y_true = []
    if mode == "generated":
        anchor_embeddings = generated_embeddings
    else:
        anchor_embeddings = real_embeddings

    for key in anchor_embeddings:
        alternate_keys = [k for k in real_embeddings if k != key]
        for aidx, anchor_embedding in enumerate(anchor_embeddings[key]):
            for ridx, real_same_embedding in enumerate(real_embeddings[key]):
                if mode == "real" and ridx == aidx:
                    # skip if same speaker and same utterance
                    continue

                y_score.append(get_similarity(anchor_embedding, real_same_embedding))
                y_true.append(1)

                alternate_speaker = random.choice(alternate_keys)
                alternate_audio_idx = random.randint(0, len(real_embeddings[alternate_speaker]) - 1)
                alternate_embedding = real_embeddings[alternate_speaker][alternate_audio_idx]
                y_score.append(get_similarity(anchor_embedding, alternate_embedding))
                y_true.append(0)

    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    _auc = auc(fpr, tpr)
    fnr = 1 - tpr
    eer_threshold = thresholds[np.nanargmin(np.absolute((fnr - fpr)))]
    eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    eer_verify = fnr[np.nanargmin(np.absolute((fnr - fpr)))]

    assert abs(eer - eer_verify) < 1.0

    return {
        'eer': eer,
        'auc': _auc,
    }


def get_activation(filename):
    audio = fid_wave_model.process(filename, trim=False)
    inputs = fid_processor(audio, sampling_rate=16000, return_tensors="pt")
    with torch.no_grad():
        outputs = fid_model(**inputs)
    # activation = outputs.last_hidden_state.mean(axis=1)
    activation = outputs.extract_features.mean(axis=1)
    return activation


def frechet_classifier_distance_from_activations(activations1, activations2):
    mu1, sigma1 = calculate_activation_statistics(activations1)
    mu2, sigma2 = calculate_activation_statistics(activations2)
    fid = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
    return fid


def calculate_activation_statistics(act):
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, "Training and test mean vectors have different lengths"
    assert sigma1.shape == sigma2.shape, "Training and test covariances have different dimensions"

    diff = mu1 - mu2
    # product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = "fid calculation produces singular product; adding %s to diagonal of cov estimates" % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


def evaluate(
    manifest_path,
    fastpitch_ckpt_path,
    ssl_model_ckpt_path,
    hifi_ckpt_path,
    sv_model_name,
    base_out_dir,
    n_speakers,
    min_samples_per_spk,
    max_samples_per_spk,
    evaluation_type,
    pitch_stats_json=None,
    precomputed_stats_fp=None,
    compute_pitch=1,
    compute_duration=1,
    use_unique_tokens=1,
    duration_per_speaker=10,  # in seconds.
    dataset_id=0,
):
    manifest_name = manifest_path.split("/")[-1].split(".")[0]
    fp_ckpt_name = "FastPitch" + fastpitch_ckpt_path.split("/")[-1].split(".")[0]
    out_dir_name = "{}_{}_SpkDur_{}_dataid_{}".format(
        manifest_name, fp_ckpt_name, int(duration_per_speaker), dataset_id
    )
    out_dir = os.path.join(base_out_dir, out_dir_name)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    ssl_model = ssl_tts.SSLDisentangler.load_from_checkpoint(ssl_model_ckpt_path, strict=False)
    ssl_model = ssl_model.to(device)
    ssl_model.eval()

    vocoder = hifigan.HifiGanModel.load_from_checkpoint(hifi_ckpt_path).to(device)
    vocoder.eval()

    fastpitch_model = fastpitch_ssl.FastPitchModel_SSL.load_from_checkpoint(fastpitch_ckpt_path, strict=False)
    fastpitch_model = fastpitch_model.to(device)
    fastpitch_model.eval()
    fastpitch_model.non_trainable_models = {'vocoder': vocoder}

    nemo_sv_model = label_models.EncDecSpeakerLabelModel.from_pretrained(sv_model_name)
    nemo_sv_model = nemo_sv_model.to(device)
    nemo_sv_model.eval()

    wav_featurizer = WaveformFeaturizer(sample_rate=22050, int_values=False, augmentor=None)

    speaker_wise_audio_paths = {}
    with open(manifest_path) as f:
        lines = f.readlines()
        for line in lines:
            record = json.loads(line)
            if record['speaker'] not in speaker_wise_audio_paths:
                speaker_wise_audio_paths[record['speaker']] = []
            speaker_wise_audio_paths[record['speaker']].append(record['audio_filepath'])

    filtered_paths = {}
    spk_count = 0
    sorted_keys = sorted(speaker_wise_audio_paths.keys())
    for key in sorted_keys:
        if len(speaker_wise_audio_paths[key]) >= min_samples_per_spk:
            filtered_paths[key] = speaker_wise_audio_paths[key][:max_samples_per_spk]
            spk_count += 1
            if spk_count >= n_speakers:
                break

    if precomputed_stats_fp is not None:
        # loaf from pickle
        with open(precomputed_stats_fp, 'rb') as f:
            speaker_stats = pickle.load(f)
    else:
        speaker_stats = load_speaker_stats(
            filtered_paths,
            ssl_model,
            wav_featurizer,
            manifest_name=manifest_name,
            out_dir=out_dir,
            duration_per_speaker=duration_per_speaker,
            pitch_stats_json=pitch_stats_json,
        )

    compute_pitch = compute_pitch == 1
    compute_duration = compute_duration == 1
    pitch_conditioning = True
    use_unique_tokens = use_unique_tokens == 1

    if evaluation_type == "reconstructed":
        generated_file_paths = reconstruct_audio(
            fastpitch_model,
            ssl_model,
            wav_featurizer,
            speaker_stats,
            filtered_paths,
            out_dir,
            pitch_conditioning=pitch_conditioning,
            compute_pitch=compute_pitch,
            compute_duration=compute_duration,
            use_unique_tokens=use_unique_tokens,
            dataset_id=dataset_id,
        )
    elif evaluation_type == "swapping":
        generated_file_paths = {}
        speakers = list(filtered_paths.keys())
        for tidx, target_speaker in enumerate(speakers):
            source_speaker_idx = tidx + 1 if tidx + 1 < len(speakers) else 0
            source_speaker = speakers[source_speaker_idx]
            generated_file_paths[target_speaker] = swap_speakers(
                fastpitch_model,
                ssl_model,
                wav_featurizer,
                speaker_stats,
                filtered_paths,
                out_dir,
                source_speaker,
                target_speaker,
                pitch_conditioning=pitch_conditioning,
                compute_pitch=compute_pitch,
                compute_duration=compute_duration,
                use_unique_tokens=use_unique_tokens,
                dataset_id=dataset_id,
            )

    speaker_embeddings = {}
    original_filepaths = filtered_paths

    generated_fp_list = []
    original_fp_list = []
    for key in generated_file_paths:
        speaker_embeddings["generated_{}".format(key)] = []
        speaker_embeddings["original_{}".format(key)] = []

        for fp in generated_file_paths[key]:
            generated_fp_list.append(fp)
            print("getting embedding for {}".format(fp))
            embedding = nemo_sv_model.get_embedding(fp)
            embedding = embedding.cpu().detach().numpy().flatten()
            speaker_embeddings["generated_{}".format(key)].append(embedding)

        for fp in original_filepaths[key]:
            original_fp_list.append(fp)
            print("getting embedding for {}".format(fp))
            embedding = nemo_sv_model.get_embedding(fp)
            embedding = embedding.cpu().detach().numpy().flatten()
            speaker_embeddings["original_{}".format(key)].append(embedding)

    gns = []
    gts = []
    for gn, gt in tqdm(zip(generated_fp_list, original_fp_list)):
        gns.append(get_activation(gn))
        gts.append(get_activation(gt))

    fid = frechet_classifier_distance_from_activations(np.concatenate(gts), np.concatenate(gns))

    sv_metrics = calculate_eer(speaker_embeddings)
    sv_metrics_real = calculate_eer(speaker_embeddings, mode="real")
    metrics = {'generated': sv_metrics, 'real': sv_metrics_real, 'fid': fid}
    with open(os.path.join(out_dir, "sv_metrics_{}_{}.json".format(evaluation_type, sv_model_name)), "w") as f:
        json.dump(metrics, f)
    print("Metrics", metrics)
    # print("Real Metrics", sv_metrics_real)
    eer_str = "EER: {:.3f}".format(sv_metrics['eer'])
    visualize_embeddings(
        speaker_embeddings,
        out_dir=out_dir,
        title="TSNE {} {}".format(evaluation_type, eer_str),
        out_file="tsne_{}_{}".format(evaluation_type, sv_model_name),
    )


def main():
    parser = argparse.ArgumentParser(description='Evaluate the model')
    parser.add_argument(
        '--ssl_model_ckpt_path',
        type=str,
        default="/home/shehzeenh/Conformer-SSL/3253979_/Conformer-SSL/checkpoints/Conformer22050_Epoch37.ckpt",
    )
    parser.add_argument(
        '--hifi_ckpt_path', type=str, default="/home/shehzeenh/HiFiModel/hifigan_libritts/HiFiLibriEpoch334.ckpt"
    )
    parser.add_argument('--fastpitch_ckpt_path', type=str, default="/home/shehzeenh/FastPitchSSL/Epoch264.ckpt")
    parser.add_argument('--sv_model_name', type=str, default="ecapa_tdnn")
    parser.add_argument('--manifest_path', type=str, default="/home/shehzeenh/libritts_dev_clean_local.json")
    parser.add_argument(
        '--train_manifest_path', type=str, default="/home/shehzeenh/libritts_train_formatted_local.json"
    )
    parser.add_argument(
        '--pitch_stats_json', type=str, default="/home/shehzeenh/SpeakerStats/libri_speaker_stats.json"
    )
    parser.add_argument('--out_dir', type=str, default="/home/shehzeenh/Evaluations/testing/")
    parser.add_argument('--evaluation_type', type=str, default="reconstructed")  # reconstructed, swapping
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--n_speakers', type=int, default=10)
    parser.add_argument('--min_samples_per_spk', type=int, default=2)
    parser.add_argument('--max_samples_per_spk', type=int, default=10)
    parser.add_argument('--pitch_conditioning', type=int, default=1)
    parser.add_argument('--compute_pitch', type=int, default=1)
    parser.add_argument('--compute_duration', type=int, default=1)
    parser.add_argument('--use_unique_tokens', type=int, default=1)
    parser.add_argument('--precomputed_stats_fp', type=str, default=None)
    parser.add_argument('--duration_per_speaker', type=int, default=10)
    parser.add_argument('--dataset_id', type=int, default=0)
    args = parser.parse_args()

    evaluate(
        manifest_path=args.manifest_path,
        fastpitch_ckpt_path=args.fastpitch_ckpt_path,
        ssl_model_ckpt_path=args.ssl_model_ckpt_path,
        hifi_ckpt_path=args.hifi_ckpt_path,
        sv_model_name=args.sv_model_name,
        base_out_dir=args.out_dir,
        n_speakers=args.n_speakers,
        min_samples_per_spk=args.min_samples_per_spk,
        max_samples_per_spk=args.max_samples_per_spk,
        evaluation_type=args.evaluation_type,
        pitch_stats_json=args.pitch_stats_json,
        precomputed_stats_fp=args.precomputed_stats_fp,
        compute_pitch=args.compute_pitch,
        compute_duration=args.compute_duration,
        use_unique_tokens=args.use_unique_tokens,
        duration_per_speaker=args.duration_per_speaker,
        dataset_id=args.dataset_id,
    )


if __name__ == '__main__':
    main()
