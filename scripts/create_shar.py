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

def find_wav_files(wav_dirs, wav_name):
    if not os.path.isabs(wav_name):
        wav_path = ""
        for wav_dir in wav_dirs:
            if os.path.exists(os.path.join(wav_dir, wav_name)):
                wav_path = os.path.join(wav_dir, wav_name)
                break
    else:
        wav_path = wav_name
    assert os.path.exists(wav_path), "wav path does not exists"
    return wav_path

def create_shar_from_manifest(
    manifest, wav_dirs, codec_dir, codec_np_dir, out_shar_dir, source_lang, target_lang
):
    from lhotse import CutSet, SupervisionSegment, Recording, AudioSource
    from lhotse.cut.base import Cut
    from lhotse.features.base import Features, FeatureSet
    from lhotse.array import TemporalArray, Array
    from lhotse.shar.writers import ArrayTarWriter
    from lhotse.audio import RecordingSet

    lines = []
    print(f"Opening {manifest}")
    with open(manifest, "r") as f:
        ls = f.readlines()
        lines.extend(ls)
    codec_dir = Path(codec_dir)
    codec_np_dir = Path(codec_np_dir)

    data_list = []
    context_list = []
    answer_list = []
    rec_list = []
    source_text_list = []
    question_list = []
    ali_score_list = []
    for i, line in tqdm(enumerate(lines)):
        datapoint = json.loads(line)
        answer_path = Path(datapoint["target_codec"].split('/')[-1])
        context_path = "/workspace/data/s2s/context/target_codes_en_Rodney_44khz_WIZWIKI_RODNEY_WIZWIKI_005619.pt"
        answer = codec_dir / answer_path
        answer_tensor = torch.load(answer).to(torch.int32).T
        context = context_path
        context_tensor = torch.load(context).to(torch.int32).T
        # Save answer
        numpy_file = codec_np_dir / answer_path
        if not numpy_file.parent.exists():
            numpy_file.parent.mkdir()
        np.save(numpy_file.with_suffix(""), answer_tensor.numpy())
        if i % 10000 == 0:
            print(f"Saving {answer} to {numpy_file}")
        # Save context
        numpy_context_file = Path("/workspace/data/s2s/context/target_codes_en_Rodney_44khz_WIZWIKI_RODNEY_WIZWIKI_005619.pt")
        np.save(numpy_context_file.with_suffix(""), context_tensor.numpy())
        answer_feat = TemporalArray(
            # type="codec_codes",
            Array(
                storage_type='numpy_files',
                storage_path=str(numpy_file.parent),
                storage_key=str(numpy_file.with_suffix(".npy").name),
                shape=answer_tensor.shape,
            ),
            temporal_dim=0,
            frame_shift=1/86.,
            start=0,
        )
        wav_path = find_wav_files(wav_dirs, datapoint["audio_filepath"])
        wav, _ = sf.read(wav_path)
        rec = Recording(
            id=datapoint["audio_filepath"], 
            duration=datapoint["duration"], 
            sampling_rate=22050, 
            sources=[AudioSource(type='file', channels=[0], source=wav_path)],
            num_samples=wav.shape[0],
        )

        data_list.append(answer_feat)
        rec_list.append(rec)
        context_list.append(numpy_context_file)
        answer_list.append(datapoint["answer"])
        source_text_list.append(datapoint["source_text"])
        question_list.append(datapoint["question"])
        ali_score_list.append(datapoint["ali_score"])

    print("Done loading torch and saving numpy")
    print(len(data_list))
    cuts = CutSet.from_manifests(recordings=RecordingSet.from_recordings(rec_list))

    # Attach text
    for i, cut in tqdm(enumerate(cuts)):
        cut.supervisions.append(
            SupervisionSegment(
                id=cut.id,
                recording_id=cut.id,
                start=0,
                duration=cut.recording.duration,
                text=question_list[i],
                language="EN",
            ),
        )
        cut.supervisions.append(
            SupervisionSegment(
                id=cut.id,
                recording_id=cut.id,
                start=0,
                duration=cut.recording.duration,
                text=source_text_list[i],
                language=source_lang.upper(),
            ),
        )
        cut.supervisions.append(
            SupervisionSegment(
                id=cut.id,
                recording_id=cut.id,
                start=0,
                duration=cut.recording.duration,
                text=answer_list[i],
                language=target_lang.upper(),
            ),
        )
        cut.target_codes = data_list[i]
        cut.duration = cut.recording.duration

    print("Making Shars")
    out_shar_dir = Path(out_shar_dir)
    out_shar_dir.mkdir(parents=True, exist_ok=True)
    shard_size = 10
    assert len(data_list) % shard_size != 0, "Lhotse breaks if feat_list is a multiple of shard_size"
    exported = cuts.to_shar(
        out_shar_dir, fields={"features": "numpy", "recordings": "wav"}, num_jobs=4, shard_size=shard_size
    )

    for i, path in tqdm(enumerate(exported["cuts"])):
        path = path[0]
        out_path = path.replace("cuts", "context").replace(".jsonl.gz", ".tar")
        with ArrayTarWriter(
            out_path, shard_size=None, compression="numpy"
        ) as writer:
            for cut in CutSet.from_file(path):
                feature_path = context_list[i].with_suffix(".npy")
                data = np.load(feature_path)
                cut = cut.attach_tensor(
                    "context", data,
                    # frame_shift=1/86., temporal_dim=0
                    # no to these two so it stays as a regular array and not a temporal one
                    # If temporal, it will cut context to answer_length
                )
                writer.write(cut.id, data, manifest=cut.custom["context"])
        
        out_path = path.replace("cuts", "ali_score").replace(".jsonl.gz", ".tar")
        with ArrayTarWriter(
            out_path, shard_size=None, compression="numpy"
        ) as writer:
            for cut in CutSet.from_file(path):
                ali_score = np.array([ali_score_list[i]])
                cut = cut.attach_tensor(
                    "ali_score", ali_score,
                )
                writer.write(cut.id, ali_score, manifest=cut.custom["ali_score"])


if __name__ == "__main__":
    create_shar_from_manifest(
        manifest="/workspace/data/s2s/es/manifest_es_to_rodney_no_emotion_context_Rodney_44khz_WIZWIKI_RODNEY_WIZWIKI_005619_s2s_codec_forced_ali_score.json",
        wav_dirs=[
            f"/workspace/data/s2s/es/source_wav/audio_{i}" for i in range(512)
        ],
        out_shar_dir="/workspace/data/s2s_shars/es/",
        codec_dir="/workspace/data/s2s_codec/es/codecs/nemo_codec_bw_6.0/",
        codec_np_dir="/workspace/data/s2s_codec/es/codecs/numpy/",
        source_lang="es",
        target_lang="en",
    )
