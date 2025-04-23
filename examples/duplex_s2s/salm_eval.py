import importlib
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Optional

import lhotse.dataset
import torch
from lhotse import CutSet
from lhotse.cut import Cut
from lhotse.serialization import SequentialJsonlWriter
from omegaconf import OmegaConf
from torch.distributed.checkpoint import load as tdc_load
from transformers import GenerationConfig
from whisper_normalizer.english import EnglishTextNormalizer

from nemo.collections.asr.metrics.wer import word_error_rate
from nemo.collections.common.data.lhotse.cutset import guess_parse_cutset
from nemo.collections.common.data.lhotse.text_adapters import AudioTurn, NeMoMultimodalConversation, TextTurn
from nemo.collections.duplex_s2s.models.salm import SALM
from nemo.core.config import hydra_runner
from nemo.utils import logging


def load_checkpoint(model: torch.nn.Module, checkpoint_dir_or_file: str) -> None:
    # Load non-distributed checkpoint
    if Path(checkpoint_dir_or_file).is_file():
        model.load_state_dict(torch.load(checkpoint_dir_or_file, map_location=model.device)['state_dict'])
        return

    # Load distributed checkpoint
    state_dict = {"state_dict": model.state_dict()}
    tdc_load(state_dict, checkpoint_id=checkpoint_dir_or_file)
    model.load_state_dict(state_dict["state_dict"])


def cut_to_conversation(cut: Cut, audio_locator_tag: str) -> NeMoMultimodalConversation:
    turns = [
        AudioTurn(cut=cut, role="user", audio_locator_tag=audio_locator_tag),
        TextTurn(value=cut.supervisions[0].text, role="assistant"),
    ]
    if hasattr(cut, "context"):
        turns = [TextTurn(value=cut.context, role="user")] + turns
    return NeMoMultimodalConversation(
        id=cut.id,
        turns=turns,
        token_equivalent_duration=0.08,
        custom=cut.custom,
    )


def add_system_prompt(
    example, system_prompt: str = "detailed thinking off", context: str = "Repeat after me, typing in lowercase."
):
    example.system_prompt = system_prompt
    example.context = context
    return example


class ToAudio(torch.utils.data.Dataset):
    def __getitem__(self, cuts: CutSet):
        audios, audio_lens = cuts.load_audio(collate=True)
        return {"cuts": cuts, "audios": audios, "audio_lens": audio_lens}


@dataclass
class SalmEvalConfig:
    # inputs: str = "/ws2/datasets/ast/covost_v2_full/test/covost_v2.es_en.test.es.json"
    inputs: str = "/home/pzelasko/data/librispeech/dev-other-wav.json"
    batch_size: int = 64
    max_new_tokens: int = 128
    output_manifest: Optional[str] = "generations.jsonl"
    verbose: bool = True
    use_normalizer: bool = True


@hydra_runner(config_name="SalmEvalConfig", schema=SalmEvalConfig)
def main(cfg: SalmEvalConfig):
    logging.info(f'Hydra config:\n{OmegaConf.to_yaml(cfg)}')

    # TODO: load from pretrained HF should involve all of the below
    model_cfg = OmegaConf.load("/home/pzelasko/code/NeMo/examples/duplex_s2s/conf/salm.yaml")
    ckpt = "oci-salm/step=5000.ckpt/"
    model = SALM(model_cfg.model).eval().to(torch.bfloat16).cuda()
    load_checkpoint(model, ckpt)
    model.configure_model()

    cuts = guess_parse_cutset(cfg.inputs).sort_by_duration()
    dloader = torch.utils.data.DataLoader(
        dataset=ToAudio(),
        sampler=lhotse.dataset.DynamicCutSampler(cuts, max_cuts=cfg.batch_size),
        num_workers=1,
        batch_size=None,
    )

    if cfg.use_normalizer:
        normalizer = EnglishTextNormalizer()
    else:
        normalizer = lambda x: x

    refs = []
    hyps = []
    input_durations = []
    infer_durations = []
    for batch_idx, batch in enumerate(dloader):
        ts = perf_counter()
        answer_ids = model.generate(
            prompts=[
                [
                    {"role": "system", "slots": {"message": "detailed thinking off"}},
                    {"role": "user", "slots": {"message": f"Repeat after me. {model.audio_locator_tag}"}},
                    # {"role": "user", "slots": {"message": f"Repeat after me, typing in lowercase. {model.audio_locator_tag}"}}
                ]
            ]
            * len(batch["cuts"]),
            audios=batch["audios"].to(model.device, non_blocking=True),
            audio_lens=batch["audio_lens"].to(model.device, non_blocking=True),
            generation_config=GenerationConfig(
                max_new_tokens=cfg.max_new_tokens,
                bos_token_id=model.text_bos_id,
                eos_token_id=model.text_eos_id,
                pad_token_id=model.text_pad_id,
            ),
        )
        answer_ids = answer_ids.cpu()
        batch_infer_duration = perf_counter() - ts

        batch_duration = sum(c.duration for c in batch["cuts"])
        batch_refs = [cut.supervisions[0].text for cut in batch["cuts"]]
        batch_hyps = [
            normalizer(model.tokenizer.ids_to_text(ans[ans != model.text_pad_id]).strip()) for ans in answer_ids
        ]
        if cfg.verbose:
            batch_wer = word_error_rate(batch_hyps, batch_refs)
            batch_rtfx = batch_duration / batch_infer_duration
            logging.info(f"Batch {batch_idx}: WER={batch_wer:.2%} RTFx={batch_rtfx:.1f}")

        refs.extend(batch_refs)
        hyps.extend(batch_hyps)
        input_durations.append(batch_duration)
        infer_durations.append(batch_infer_duration)

    wer = word_error_rate(hypotheses=hyps, references=refs, use_cer=False)
    rtfx = sum(input_durations) / sum(infer_durations)
    logging.info(f"WER: {wer:.2%}")
    logging.info(f"RTFx: {rtfx:.1f}")

    if cfg.output_manifest is not None:
        with SequentialJsonlWriter(cfg.output_manifest) as writer:
            for cut, ref, hyp in zip(cuts, refs, hyps):
                writer.write({"id": cut.id, "text": ref, "pred_text": hyp})


if __name__ == '__main__':
    main()
