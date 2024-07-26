from pathlib import Path
import json
import os
import shutil
import csv
import soundfile as sf
from nemo.collections.tts.models import AudioCodecModel
import librosa
import torch
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt


def create_shar_from_manifest(
    manifest="/home/jasoli/data_prime/manifests/smallvctk_nemo_codec_bw_6.0_phoneme.json",
    codec_dir="/home/jasoli/data_prime/codecs/",
    codec_np_dir="/home/jasoli/data_prime/codecs_numpy/",
    out_shar_dir="/home/jasoli/data_prime/shars/"
):
    from lhotse import CutSet, SupervisionSegment
    from lhotse.cut.base import Cut
    from lhotse.features.base import Features, FeatureSet
    from lhotse.shar.writers import ArrayTarWriter

    if isinstance(manifest, str):
        manifest = [manifest]

    lines = []
    for mani in manifest:
        print(f"Opening {mani}")
        with open(mani, "r") as f:
            ls = f.readlines()
            lines.extend(ls)
    codec_dir = Path(codec_dir)
    codec_np_dir = Path(codec_np_dir)

    data_list = []
    context_list = []
    question_list = []
    for i, line in enumerate(lines):
        datapoint = json.loads(line)
        answer_path = Path(datapoint["answer"])
        if answer_path.is_relative_to("/datap/misc/speechllm_codecdatasets/codecs"):
            answer_path = answer_path.relative_to("/datap/misc/speechllm_codecdatasets/codecs")
        context_path = Path(datapoint["context"])
        if context_path.is_relative_to("/datap/misc/speechllm_codecdatasets/codecs"):
            context_path = context_path.relative_to("/datap/misc/speechllm_codecdatasets/codecs")
        answer = codec_dir / answer_path
        answer_tensor = torch.load(answer).to(torch.int32).T
        context = codec_dir / context_path
        context_tensor = torch.load(context).to(torch.int32).T
        # Save answer
        numpy_file = codec_np_dir / answer_path
        if not numpy_file.parent.exists():
            numpy_file.parent.mkdir()
        np.save(numpy_file.with_suffix(""), answer_tensor.numpy())
        if i % 10000 == 0:
            print(f"Saving {answer} to {numpy_file}")
        # Save context
        numpy_context_file = codec_np_dir / context_path
        np.save(numpy_context_file.with_suffix(""), context_tensor.numpy())
        answer_feat = Features(
            type="codec_codes",
            num_frames=answer_tensor.shape[0],
            frame_shift=1/86.,
            num_features=answer_tensor.shape[1],
            sampling_rate=22050,
            start=0,
            duration=answer_tensor.shape[0]/86.,
            storage_type='numpy_files',
            storage_path=str(numpy_file.parent),
            storage_key=str(numpy_file.with_suffix(".npy").name),
        )
        data_list.append(answer_feat)
        context_list.append(numpy_context_file)
        question_list.append(datapoint["question"])

    print("Done loading torch and saving numpy")
    print(len(data_list))
    cuts = CutSet.from_manifests(features=data_list)

    # Attach text
    for i, cut in enumerate(cuts):
        cut.supervisions.append(
            SupervisionSegment(
                id=cut.id,
                recording_id=cut.id,
                start=0,
                duration=cut.duration,
                text=question_list[i],
                language="EN",
            )
        )

    print("Making Shars")
    out_shar_dir = Path(out_shar_dir)
    out_shar_dir.mkdir(parents=True, exist_ok=True)
    shard_size = 1000
    assert len(data_list) % shard_size != 0, "Lhotse breaks if feat_list is a multiple of shard_size"
    exported = cuts.to_shar(
        out_shar_dir, fields={"features": "numpy"}, num_jobs=4, shard_size=shard_size
    )

    field = "context"
    for i, path in enumerate(exported["cuts"]):
        path = path[0]
        out_path = path.replace("cuts", field).replace(".jsonl.gz", ".tar")
        with ArrayTarWriter(
            out_path, shard_size=None, compression="numpy"
        ) as writer:
            for cut in CutSet.from_file(path):
                feature_path = context_list[i].with_suffix(".npy")
                data = np.load(feature_path)
                cut = cut.attach_tensor(
                    field, data,
                    # frame_shift=1/86., temporal_dim=0
                    # no to these two so it stays as a regular array and not a temporal one
                    # If temporal, it will cut context to answer_length
                )
                writer.write(cut.id, data, manifest=cut.custom[field])


def read_shar(out_shar_dir="/home/jasoli/data_prime/shars/", codec_model=None):
    from lhotse import CutSet
    from lhotse.dataset import DynamicBucketingSampler
    from lhotse.dataset.collation import _read_features, maybe_pad
    from typing import Tuple, Optional

    def collate_features(
        cuts: "CutSet",
        pad_direction: str = "right",
        executor: Optional["Executor"] = None,
    ) -> Tuple["torch.Tensor", "torch.Tensor"]:
        """
        Load features for all the cuts and return them as a batch in a torch tensor.
        The output shape is ``(batch, time, features)``.
        The cuts will be padded with silence if necessary.

        :param cuts: a :class:`CutSet` used to load the features.
        :param pad_direction: where to apply the padding (``right``, ``left``, or ``both``).
        :param executor: an instance of ThreadPoolExecutor or ProcessPoolExecutor; when provided,
            we will use it to read the features concurrently.
        :return: a tuple of tensors ``(features, features_lens)``.
        """
        assert all(cut.has_features for cut in cuts)
        features_lens = torch.tensor([cut.num_frames for cut in cuts], dtype=torch.int)
        cuts = maybe_pad(
            cuts, num_frames=max(features_lens).item(), direction=pad_direction
        )
        first_cut = next(iter(cuts))
        features = torch.empty(len(cuts), first_cut.num_frames, first_cut.num_features)
        if executor is None:
            for idx, cut in enumerate(cuts):
                features[idx] = _read_features(cut)
        else:
            for idx, example_features in enumerate(executor.map(_read_features, cuts)):
                features[idx] = example_features
        return features, features_lens

    cuts = CutSet.from_shar(in_dir=out_shar_dir)

    print()
    print("Example of cut setiteration and reading custom fields")

    for cut in cuts:
        print(cut)
        answer = cut.load_features()
        print("answer", answer.shape)
        context = cut.load_context()
        print("context", context.shape)
        print("question", cut.supervisions)
        decode_savewav(answer, "answer", codec_model)
        decode_savewav(context, "context", codec_model)
        break

    print()
    print("Example of sampler iteration and reading custom fields")

    # sampler = DynamicBucketingSampler(cuts, max_duration=50, num_buckets=2, shuffle=True)
    # for cuts_batch in sampler:
    #     print(cuts_batch)
    #     print("codes", collate_features(cuts_batch))


def read_result_dir(result_path, file_prefix, codec_model):
    inputs = []
    with open(result_path+file_prefix+'inputs.jsonl', 'r') as f:
        for line in f:
            inputs.append(json.loads(line)["input"])

    n_samples = len(inputs)
    for i in tqdm(range(n_samples)):
        answer = np.load(os.path.join(result_path, 'npy', 'answers', file_prefix+f"answer_{i}.npy"))
        answer_mask = answer[:, 0] == 1 # 1 is the unk token which we apply to the text channel of speech outputs
        pred = np.load(os.path.join(result_path, 'npy', 'preds', file_prefix+f"pred_{i}.npy"))
        pred_mask = pred[:, 0] == 1
        speaker_context = np.load(os.path.join(result_path, 'npy', 'speaker_contexts', file_prefix+f"speaker_context_{i}.npy"))
        context_mask = speaker_context[:, 0] == 1

        decode_savewav(answer[answer_mask, 1:], os.path.join(result_path, 'wav', 'answers', f"answer_{i}.wav"), codec_model)
        decode_savewav(pred[pred_mask, 1:], os.path.join(result_path, 'wav', 'preds', f"pred_{i}.wav"), codec_model)
        decode_savewav(speaker_context[context_mask, 1:], os.path.join(result_path, 'wav', 'speaker_contexts', f"speaker_context_{i}.wav"), codec_model)

        self_attn = np.load(os.path.join(result_path, 'npy', 'self_attn', file_prefix+f"self_attn_{i}.npy"))
        cross_attn = np.load(os.path.join(result_path, 'npy', 'cross_attn', file_prefix+f"cross_attn_{i}.npy"))

        plot(self_attn, os.path.join(result_path, 'png', 'self_attn', f"self_attn_{i}.png"))
        plot(cross_attn, os.path.join(result_path, 'png', 'cross_attn', f"self_attn_{i}.png"))

def decode_savewav(codes, name, codec_model):
    sample_rate = 22050
    os.makedirs(os.path.dirname(name), exist_ok=True)

    codes = torch.tensor(codes).to(codec_model.device).T
    codec_len = torch.Tensor([codes.shape[1]]).long().to(codec_model.device)
    wav, _ = codec_model.decode(tokens=codes.unsqueeze(0), tokens_len=codec_len)
    wav = wav[0]

    sf.write(name, wav.detach().cpu().numpy(), sample_rate)


def plot(attn_weights, name):
    os.makedirs(os.path.dirname(name), exist_ok=True)
    plt.imshow(attn_weights)
    plt.savefig(name)
    plt.clf()


def encode_decode_savewav(wavfile_name, codec_model):
    y, sr = librosa.load(wavfile_name, sr=None)
    audio = torch.tensor(y).to(codec_model.device)
    audio_length = torch.tensor(audio.size(0)).long().to(codec_model.device)
    original_codec_codes, _ = codec_model.encode(audio=audio.unsqueeze(0), audio_len=audio_length.unsqueeze(0))
    original_codec_codes = original_codec_codes[0]

    decode_savewav(original_codec_codes.T, "answer_reencoded", codec_model)


if __name__ == "__main__":
    out_shar_dir="/home/jasoli/data_prime/shars/5k_LRHM_highsim/"
    result_path="/workspace/Results/test/"
    file_prefix="test_validation_5k_LRHM_highsim_"
    codec_model_ckpt = "/workspace/model/SpeechCodec_2402.nemo"
    # create_shar_from_manifest(
    #     manifest=[
    #         "/home/jasoli/data_prime/manifests/RivattsEnglish_train_nemo_codec_bw_6.0_sentencepiece_tts_highsimilarity3.json",
    #         "/home/jasoli/data_prime/manifests/hifitts_nemo_codec_bw_6.0_train_sentencepiece_tts_highsimilarity2.json",
    #         "/home/jasoli/data_prime/manifests/LibriTTSCorrectContext_train_nemo_codec_bw_6.0_sentencepiece_tts_highsimilarity2.json",
    #         "/home/jasoli/data_prime/manifests/MLS_textformatted_5586hrs_nemo_codec_bw_6.0_sentencepiece_tts_highsimilarity2.json",
    #     ],
    #     out_shar_dir=out_shar_dir
    # )
    codec_model = AudioCodecModel.restore_from(codec_model_ckpt)
    codec_model.to('cuda')
    codec_model.eval()
    read_result_dir(result_path, file_prefix, codec_model)
    # encode_decode_savewav("answer.wav", codec_model)
