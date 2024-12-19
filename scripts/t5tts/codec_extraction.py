import json
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.strategies import DDPStrategy
from nemo.collections.tts.models import AudioCodecModel
import os
from nemo.collections.asr.parts.preprocessing.segment import AudioSegment
import argparse
from pytorch_lightning.utilities.rank_zero import rank_zero_only

class AudioDataset(Dataset):
    def __init__(self, file_lists, base_audio_dirs, dataset_names, out_dir, sample_rate=22050, pad_multiple=1024):
        self.file_list = file_lists
        self.base_audio_dirs = base_audio_dirs
        self.sample_rate = sample_rate
        self.pad_multiple = pad_multiple
        self.out_dir = out_dir
        self.combined_file_list = []
        for fidx, file_list in enumerate(file_lists):
            base_audio_dir = base_audio_dirs[fidx]
            dataset_name = dataset_names[fidx]
            for file_path in file_list:
                audio_file_path = os.path.join(base_audio_dir, file_path)
                self.combined_file_list.append({
                    "file_path": file_path,
                    "audio_file_path": audio_file_path,
                    "dataset_name": dataset_name
                })

    def __len__(self):
        return len(self.combined_file_list)

    def get_wav_from_filepath(self, file_path):
        features = AudioSegment.segment_from_file(
            file_path, target_sr=self.sample_rate, n_segments=-1, trim=False,
        )
        audio_samples = features.samples
        audio = torch.tensor(audio_samples)
        audio = torch.nn.functional.pad(audio, (0, self.pad_multiple - audio.size(0) % self.pad_multiple), value=0)
        audio_length = torch.tensor(audio.size(0)).long()
        return audio, audio_length

    def __getitem__(self, idx):
        file_path = self.combined_file_list[idx]["file_path"]
        audio_file_path = self.combined_file_list[idx]["audio_file_path"]
        dataset_name = self.combined_file_list[idx]["dataset_name"]
        assert not file_path.startswith("/"), "file_path should be relative"
        audio, audio_length = self.get_wav_from_filepath(audio_file_path)
        codec_file_path_rel = file_path.replace(".wav", ".pt").replace(".flac", ".pt")
        return {
            "audio": audio,
            "audio_length": audio_length,
            "file_path": file_path,
            "codec_file_path": os.path.join(self.out_dir, dataset_name, codec_file_path_rel)
        }
    
    def collate_fn(self, batch):
        audios_padded = []
        audio_lengths = []
        file_paths = []
        codec_file_paths = []
        max_audio_length = max(item["audio_length"].item() for item in batch)
        for item in batch:
            audio = torch.nn.functional.pad(
                item["audio"], (0, max_audio_length - item["audio"].size(0)), value=0
            )
            audios_padded.append(audio)
            audio_lengths.append(item["audio_length"])
            file_paths.append(item["file_path"])
            codec_file_paths.append(item["codec_file_path"])
        
        return {
            "audios": torch.stack(audios_padded),
            "audio_lengths": torch.stack(audio_lengths),
            "audio_file_paths": file_paths,
            "codec_file_paths": codec_file_paths
        }


class CodecExtractor(pl.LightningModule):
    def __init__(self, model_path):
        super().__init__()
        self.codec_model = AudioCodecModel.restore_from(model_path, strict=False)
        self.codec_model.eval()

    def forward(self, batch):
        with torch.no_grad():
            codes, codes_lengths = self.codec_model.encode(audio=batch["audios"], audio_len=batch["audio_lengths"])
        return codes, codes_lengths

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        codes, codes_lengths = self(batch)
        for i, file_path in enumerate(batch["codec_file_paths"]):
            # get directory from file path
            item_codes = codes[i, :, :codes_lengths[i]] # 8, T
            torch.save(item_codes.cpu().type(torch.int16), file_path)
        return None

def read_manifest(manifest_path):
    records = []
    with open(manifest_path, 'r') as f:
        all_lines = f.readlines()
        for line in all_lines:
            line = line.strip()
            record = json.loads(line)
            records.append(record)
    return records

def write_manifest(manifest_path, records):
    with open(manifest_path, 'w') as f:
        file_str = ""
        for record in records:
            file_str += json.dumps(record) + "\n"
        file_str = file_str.strip()
        f.write(file_str)
        print("Wrote {} records to: {}".format(len(records), manifest_path))

@rank_zero_only
def update_manifests(manifests, save_dir, dataset_names, codec_model_name):
    for midx, manifest in enumerate(manifests):
        records = read_manifest(manifest)
        for ridx, record in enumerate(records):
            audio_codes_path = record["audio_filepath"].replace(".wav", ".pt").replace(".flac", ".pt")
            audio_codes_path = os.path.join(save_dir, dataset_names[midx], audio_codes_path)
            record["target_audio_codes_path"] = audio_codes_path
            if ridx % 10 == 0:
                assert os.path.exists(audio_codes_path), "Audio codes not found: {}".format(audio_codes_path)

            if "context_audio_filepath" in record:
                context_audio_codes_path = record["context_audio_filepath"].replace(".wav", ".pt").replace(".flac", ".pt")
                context_audio_codes_path = os.path.join(save_dir, dataset_names[midx], context_audio_codes_path)
                record["context_audio_codes_path"] = context_audio_codes_path
                if ridx % 10 == 0:
                    assert os.path.exists(context_audio_codes_path), "Context audio codes not found: {}".format(context_audio_codes_path)
        
        write_manifest(manifest.replace(".json", "_withAudioCodes_{}.json".format(codec_model_name)), records)

def prepare_directories(base_save_dir, codec_model_name, manifests, audio_base_dirs, dataset_names):
    print("In prepare_directories")
    save_dir = os.path.join(base_save_dir, codec_model_name)
    file_lists = []
    for midx, manifest in enumerate(manifests):
        records = read_manifest(manifest)
        unique_audio_file_paths = {}
        for record in records:
            unique_audio_file_paths[record["audio_filepath"]] = 1
            if "context_audio_filepath" in record:
                unique_audio_file_paths[record["context_audio_filepath"]] = 1
        file_list = list(unique_audio_file_paths.keys())
        file_lists.append(file_list)
        for file_path in file_list:
            dir_path = os.path.dirname(file_path)
            out_dir_path = os.path.join(save_dir, dataset_names[midx], dir_path)
            if not os.path.exists(out_dir_path):
                os.makedirs(out_dir_path, exist_ok=True)
    print("Created directories for saving audio codes at: ", save_dir, len(file_lists))
    return save_dir, file_lists

if __name__ == "__main__":
    """
    Usage:
    python scripts/t5tts/codec_extraction.py \
        --manifests /home/pneekhara/2023/SimpleT5NeMo/manifests/smallvctk__phoneme__nemo_audio_21fps_8codebooks_2kcodes_v2bWithWavLM_simplet5_withcontextaudiopaths.json \
        --audio_base_dirs /datap/misc/Datasets/VCTK-Corpus \
        --codec_model_name codec21Khz_no_eliz \
        --dataset_names smallvctk \
        --save_dir /home/pneekhara/2023/SimpleT5NeMo/codec_outputs_21Khz \
        --codec_model_path /datap/misc/checkpoints/AudioCodec_21Hz_no_eliz.nemo \
        --sample_rate 22050 \
        --pad_multiple 1024 \
        --devices -1 \
        --num_nodes 1 
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifests", type=str)
    parser.add_argument("--audio_base_dirs", type=str)
    parser.add_argument("--dataset_names", type=str)
    parser.add_argument("--save_dir", type=str)
    parser.add_argument("--codec_model_path", type=str)
    parser.add_argument("--codec_model_name", type=str)
    parser.add_argument("--sample_rate", type=int)
    parser.add_argument("--pad_multiple", type=int)
    parser.add_argument("--devices", type=int, default=-1)
    parser.add_argument("--num_nodes", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    trainer = Trainer(
        devices=args.devices,
        accelerator="gpu",
        strategy=DDPStrategy(find_unused_parameters=False),
        num_nodes=args.num_nodes,
        log_every_n_steps=1,
        max_epochs=1,
        logger=False,
    )
    
    audio_base_dirs = args.audio_base_dirs.split(",")
    dataset_names = args.dataset_names.split(",")
    manifests = args.manifests.split(",")

    if torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
        if rank == 0:
            save_dir, file_lists = prepare_directories(
                args.save_dir,
                args.codec_model_name,
                manifests,
                audio_base_dirs,
                dataset_names
            )
            results = [save_dir, file_lists]
        else:
            results = [None, None]
        torch.distributed.broadcast_object(results, src=0)
        save_dir, file_lists = results
    else:
        save_dir, file_lists = prepare_directories(
            args.save_dir,
            args.codec_model_name,
            manifests,
            audio_base_dirs,
            dataset_names
        )

    codec_extractor = CodecExtractor(args.codec_model_path)

    # Dataset and DataLoader
    dataset = AudioDataset(
        file_lists=file_lists,
        base_audio_dirs=audio_base_dirs,
        dataset_names=dataset_names,
        out_dir=save_dir,
        sample_rate=args.sample_rate,
        pad_multiple=args.pad_multiple,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        collate_fn=dataset.collate_fn,
    )

    # Run prediction (Saves the audio codes to files)
    trainer.predict(codec_extractor, dataloaders=dataloader)

    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    update_manifests(manifests, save_dir, dataset_names, args.codec_model_name)