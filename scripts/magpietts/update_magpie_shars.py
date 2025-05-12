import torch
import argparse
from pathlib import Path
import torch
import lightning.pytorch as pl
from lightning.pytorch.strategies import DDPStrategy
from torch.utils.data import DataLoader
from nemo.collections.tts.models import AudioCodecModel
from lightning.pytorch.callbacks import BasePredictionWriter
from lhotse import CutSet, MonoCut
from lhotse.shar.writers.shar import SharWriter
from lhotse.dataset import SimpleCutSampler
import time

class SharPredictionWriter(BasePredictionWriter):
    def __init__(
        self,
        output_dir: str,
        codec_model_name: str,
        fields: dict,
        shard_size: int = 1000,
        buffer_size: int = 128,
    ) -> None:
        super().__init__(write_interval="batch")

        self.output_dir = Path(output_dir)
        self.codec_model_name = codec_model_name
        self.fields = fields
        self.shard_size = shard_size
        self.is_initialized = False

        self.cuts_buffer = list()
        self.buffer_size = buffer_size

    def setup(self, trainer, pl_module, stage=None):
        if not self.is_initialized:
            if trainer.global_rank == 0:
                self.output_dir.mkdir(parents=True, exist_ok=True)
                self.shar_writer = SharWriter(
                    output_dir=str(self.output_dir),
                    fields=self.fields,
                    shard_size=self.shard_size,
                )
                self.shar_writer.__enter__()

            self.is_initialized = True

    def process_cut_for_saving(self, cut: MonoCut) -> MonoCut:
        new_custom = {}
        for k, v in cut.custom.items():
            if k in self.fields:
                new_custom[k] = v
        
        cut.custom = new_custom
        return cut

    def write_on_batch_end(self, trainer, pl_module, prediction, batch_indices, batch, batch_idx, dataloader_idx):
        import time
        st = time.time()
        cuts_local, codes_local, context_codes_local = prediction["cuts"], prediction["codes"], prediction["context_codes"]
        world_size = trainer.world_size
        if world_size > 1:
            gathered = [None] * trainer.world_size
            torch.distributed.all_gather_object(gathered, (cuts_local, codes_local, context_codes_local))
        else:
            gathered = [(cuts_local, codes_local, context_codes_local)]

        if trainer.global_rank == 0:
            for cuts_rank, codes_rank, context_codes_rank in gathered:
                for cut, code, context_code in zip(cuts_rank, codes_rank, context_codes_rank):
                    cut = cut.attach_tensor(
                        name=f"codes_{self.codec_model_name}",
                        data=code.cpu().type(torch.int16).numpy(),
                    ).attach_tensor(
                        name=f"context_codes_{self.codec_model_name}",
                        data=context_code.cpu().type(torch.int16).numpy(),
                    )
                    self.cuts_buffer.append(cut)

            if len(self.cuts_buffer) >= self.buffer_size:
                self._write_buffer()
            
        et = time.time()
        print("Time to write:", et - st)

    def _write_buffer(self):
        print("Writing buffer of size:", len(self.cuts_buffer))
        for cut in self.cuts_buffer:
            # print("Writing cut:", cut.id)
            processed_cut = self.process_cut_for_saving(cut)
            self.shar_writer.write(processed_cut)
        self.cuts_buffer.clear()

    def teardown(self, trainer, pl_module, stage: str | None = None):
        if trainer.world_size > 1:
            torch.distributed.barrier()

        if trainer.global_rank == 0 and self.shar_writer is not None:
            if self.cuts_buffer:
                self._write_buffer()
            self.shar_writer.close()
            self.shar_writer = None

def ddp_info():
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_world_size(), torch.distributed.get_rank()
    return 1, 0

class CodecExtractor(pl.LightningModule):
    def __init__(self, model_path, pad_multiple, batch_size, shar_root, num_workers):
        super().__init__()
        self.pad_multiple = pad_multiple
        self.codec_model = AudioCodecModel.restore_from(restore_path=model_path, strict=False)
        self.codec_model.eval()
        self._dataloader = None
        self.batch_size = batch_size
        self.shar_root = shar_root
        self.num_workers = num_workers
        self.total_cuts = 0

    def setup(self, stage=None):
        if self._dataloader is None:
            world_size, rank = ddp_info()
            print("In model.setup - world size:", world_size, "rank:", rank)
            cuts = CutSet.from_shar(in_dir=self.shar_root)
            self.total_cuts = len(cuts)
            self.approx_total_batches = self.total_cuts // (self.batch_size * world_size)
            sampler = SimpleCutSampler(
                cuts, shuffle=False, max_cuts=self.batch_size,
                world_size=world_size, rank=rank,
            )
            dataset = SimpleCutDataset(pad_multiple=self.pad_multiple)
            self._dataloader = DataLoader(
                dataset,
                sampler=sampler,
                batch_size=None,
                num_workers=self.num_workers,
            )
    
    def predict_dataloader(self):
        return self._dataloader
    
    def predict_step(self, batch, _):
        audio_stacked = batch["audio"].to(device=self.device)
        context_audio_stacked = batch["context_audio"].to(device=self.device)
        audio_lens = batch["audio_lens"].to(device=self.device)
        context_audio_lens = batch["context_audio_lens"].to(device=self.device)
        cuts = batch["cuts"]
        print("Total Steps", self.approx_total_batches)
        with torch.no_grad():
            codes, lengths = self.codec_model.encode(
                audio=audio_stacked, audio_len=audio_lens
            )
            context_codes, context_lengths = self.codec_model.encode(
                audio=context_audio_stacked, audio_len=context_audio_lens
            )

        codes = [c[:, :L].cpu().type(torch.int16) for c, L in zip(codes, lengths)]
        context_codes = [c[:, :L].cpu().type(torch.int16) for c, L in zip(context_codes, context_lengths)]
        
        return {"cuts": cuts, "codes": codes, "context_codes": context_codes}

class SimpleCutDataset(torch.utils.data.Dataset):
    def __init__(self, pad_multiple: int = 1024):
        super().__init__()
        self.pad_multiple = pad_multiple

    def __getitem__(self, cuts: CutSet):
        pad_multiple = self.pad_multiple
        audios = cuts.load_audio()
        context_audios_torch = []
        audios_torch = []
        context_audio_lens = []
        audio_lens = []
        for idx, audio in enumerate(audios):
            audio = torch.from_numpy(audio)[0]
            audio = torch.nn.functional.pad(audio, (0, pad_multiple - audio.size(0) % pad_multiple), value=0)
            audio_length = torch.tensor(audio.size(0)).long()

            context_audio = cuts[idx].context_recording.load_audio()
            context_audio = torch.from_numpy(context_audio)[0]
            context_audio = torch.nn.functional.pad(context_audio, (0, pad_multiple - context_audio.size(0) % pad_multiple), value=0)
            context_audio_length = torch.tensor(context_audio.size(0)).long()

            context_audios_torch.append(context_audio)
            audios_torch.append(audio)
            context_audio_lens.append(context_audio_length)
            audio_lens.append(audio_length)
            
        # Pad to max length
        audio_stacked = torch.nn.utils.rnn.pad_sequence(audios_torch, batch_first=True)
        context_audio_stacked = torch.nn.utils.rnn.pad_sequence(context_audios_torch, batch_first=True)
        audio_lens = torch.stack(audio_lens)
        context_audio_lens = torch.stack(context_audio_lens)

        return {
            "audio": audio_stacked,
            "context_audio": context_audio_stacked,
            "audio_lens": audio_lens,
            "context_audio_lens": context_audio_lens,
            "cuts": cuts,
        }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_dir")
    parser.add_argument("--out_dir")
    parser.add_argument("--codec_ckpt")
    parser.add_argument("--codec_name", default="testcodec")
    parser.add_argument("--pad_multiple", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--devices", type=int, default=-1)
    parser.add_argument("--num_nodes", type=int, default=1)
    parser.add_argument("--shard_size", type=int, default=4096)
    parser.add_argument("--buffer_size", type=int, default=128)
    args = parser.parse_args()

    writer = SharPredictionWriter(
        output_dir=args.out_dir,
        codec_model_name=args.codec_name,
        fields={"recording": "flac", f"codes_{args.codec_name}": "numpy", f"context_codes_{args.codec_name}": "numpy"},
        shard_size=args.shard_size,
        buffer_size=args.batch_size,
    )

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=args.devices,
        num_nodes=args.num_nodes,
        strategy=DDPStrategy(find_unused_parameters=False),
        logger=False,
        callbacks=[writer],
        use_distributed_sampler=False, 
    )

    model = CodecExtractor(
        args.codec_ckpt,
        pad_multiple=args.pad_multiple,
        batch_size=args.batch_size,
        shar_root=args.in_dir,
        num_workers=args.num_workers,
    )

    trainer.predict(model, return_predictions=False)
