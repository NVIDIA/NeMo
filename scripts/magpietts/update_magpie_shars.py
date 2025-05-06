import os
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


class SharPredictionWriter(BasePredictionWriter):
    def __init__(
        self,
        output_dir: str,
        codec_model_name: str,
        codec_frame_rate: float,
        fields: dict,
        shard_size: int = 1000,
    ) -> None:
        super().__init__(write_interval="batch")

        self.output_dir = Path(output_dir)
        self.codec_model_name = codec_model_name
        self.codec_frame_rate = codec_frame_rate
        self.fields = fields
        self.shard_size = shard_size
        self.is_initialized = False

        self.cuts_buffer = list()
        self.buffer_size = 10

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

    def _write_buffer(self):
        for cut in self.cuts_buffer:
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
            
class CodecExtractor(pl.LightningModule):
    def __init__(self, model_path: str, pad_multiple: int = 1024):
        super().__init__()
        self.pad_multiple = pad_multiple
        self.codec_model = AudioCodecModel.restore_from(restore_path=model_path, strict=False)
        self.codec_model.eval()

    def predict_step(self, cuts, _):
        audios = cuts.load_audio()
        context_audios_torch = []
        audios_torch = []
        context_audio_lens = []
        audio_lens = []
        for idx, audio in enumerate(audios):
            audio = torch.from_numpy(audio)[0]
            audio = torch.nn.functional.pad(audio, (0, self.pad_multiple - audio.size(0) % self.pad_multiple), value=0)
            audio_length = torch.tensor(audio.size(0)).long()

            context_audio = cuts[idx].context_recording.load_audio()
            context_audio = torch.from_numpy(context_audio)[0]
            context_audio = torch.nn.functional.pad(context_audio, (0, self.pad_multiple - context_audio.size(0) % self.pad_multiple), value=0)
            context_audio_length = torch.tensor(context_audio.size(0)).long()

            context_audios_torch.append(context_audio)
            audios_torch.append(audio)
            context_audio_lens.append(context_audio_length)
            audio_lens.append(audio_length)
            
        # Pad to max length
        audio_stacked = torch.nn.utils.rnn.pad_sequence(audios_torch, batch_first=True).to(device=self.device)
        context_audio_stacked = torch.nn.utils.rnn.pad_sequence(context_audios_torch, batch_first=True).to(device=self.device)
        audio_lens = torch.stack(audio_lens).to(device=self.device)
        context_audio_lens = torch.stack(context_audio_lens).to(device=self.device)
        
        codes, lengths = self.codec_model.encode(
            audio=audio_stacked, audio_len=audio_lens
        )
        context_codes, context_lengths = self.codec_model.encode(
            audio=context_audio_stacked, audio_len=context_audio_lens
        )
        

        codes = [c[:, :L].cpu().type(torch.int16) for c, L in zip(codes, lengths)]
        context_codes = [c[:, :L].cpu().type(torch.int16) for c, L in zip(context_codes, context_lengths)]
        return {"cuts": cuts, "codes": codes, "context_codes": context_codes}


def make_loader(shar_root, batch_size, num_workers):
    cuts = CutSet.from_shar(
        in_dir=shar_root,
    )
    return DataLoader(cuts, batch_size=batch_size,
                      num_workers=num_workers, shuffle=False,
                      collate_fn=lambda batch: CutSet.from_cuts(batch))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_dir")
    parser.add_argument("--out_dir")
    parser.add_argument("--codec_ckpt")
    parser.add_argument("--codec_name", default="48k")
    parser.add_argument("--frame_rate", type=float, default=50.0)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--devices", type=int, default=-1)
    parser.add_argument("--num_nodes", type=int, default=1)
    parser.add_argument("--shard_size", type=int, default=4096)
    args = parser.parse_args()

    writer = SharPredictionWriter(
        output_dir=args.out_dir,
        codec_model_name=args.codec_name,
        codec_frame_rate=args.frame_rate,
        fields={"recording": "flac", f"codes_{args.codec_name}": "numpy", f"context_codes_{args.codec_name}": "numpy"},
        shard_size=args.shard_size,
    )

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=args.devices,
        num_nodes=args.num_nodes,
        strategy=DDPStrategy(find_unused_parameters=False),
        logger=False,
        callbacks=[writer],
    )

    model = CodecExtractor(args.codec_ckpt)
    dataloader = make_loader(args.in_dir, args.batch_size, args.num_workers)

    trainer.predict(model, dataloaders=dataloader, return_predictions=False)
