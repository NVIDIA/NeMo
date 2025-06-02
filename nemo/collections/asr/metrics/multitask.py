from typing import Dict, Literal, Optional

import torch
from lhotse import CutSet
from lhotse.cut import MixedCut
from omegaconf import DictConfig, OmegaConf
from torch import nn

from nemo.collections.asr.data.audio_to_text_lhotse_prompted import PromptedAudioToTextMiniBatch
from nemo.collections.common.prompts.canary import TASK_TRANSLATE
from nemo.core.classes import Serialization

__all__ = ['MultiTaskMetric']

#TODO: add decoding metric updater

class MultiTaskMetric(Serialization):
    """
    Wrapper class for managing multiple torch metrics. Used primarily for `EncDecMultiTaskModel` but can support any model with defined prompt schema.

    Provides two major functionalities:
        1) Automatically populates parent `model` class with each submetric, allowing customization of logging behavior without needing to define each
            metric within model instantiation.
        2) Manages targeted calls to `update`, `compute`, and `reset` so only samples fullfilling specific conditions will be logged. This avoids excessive
            metric calculations (e.g. wer estimation for translation task) and improves metric quality over multitask setup.

    Args:
        model: nn.Module (ideally the parent model initializing the metric)
        cfg: OmegaConf for setup. (See below.)


    Assumes following cfg format:

    ```
        multitask_metrics_cfg:
            log_predictions: true
            ...
            metrics:
            -   name: wer
                _target_: nemo.collections.asr.metrics.WER
                slots: 
                    task: "transcribe"
            -   name: bleu
                tokenize: ???
                check_cuts_for_tokenizers: ???
                _target_: nemo.collections.asr.metrics.BLEU  
                slots: 
                    task: "translate"
            ...
    ```

    Where only the `metrics` schema is required. Each element of `metrics` requires a `name` for the metric, `_target_` class for serialization, and series of `slot`
    values that determines the conditions for the metric to apply. (Currently only the keyword `task` with vals `transcribe` and `translate` is supported.) All defined
    metrics in `asr.collections.metrics` are supported. 
    
    Similar to `input_cfg` for dataloading, extra defined properties assume soft inheritance. All properties defined outside 
    the `metrics` dict will be autopopulated into all metrics. Meanwhile, properties defined within a metrics entry will only populate that metric. 

    """

    # trick from torch metrics `SacreBLEUTokenizer`
    _INDEX_FN = {
        "canary": "_canary_index_fn",
        "canary2": "_canary2_index_fn",
    }
    
    def __init__(self, model: nn.Module, cfg: DictConfig):      
        super().__init__()  

        # Select function for proper task splitting
        self.prompt = model.prompt
        assert self.prompt.NAME in self._INDEX_FN, f"MultiTaskMetric logging is only supported for {[k for k in self._INDEX_FN.keys()]}"
        self.split_task_indices = getattr(self, f"{self._INDEX_FN[self.prompt.NAME]}")

        # Setup tracking hashes
        self._metric_dict, self._slot_dict, self._skip_dict = {}, {}, {}
        cfg = OmegaConf.to_container(cfg)
        for metric in cfg.pop("metrics"):
            name = metric["name"]
            
            # TODO: Expand slot coverage as metrics demands. Right now just manages two tasks.
            slots = metric["slots"]
            assert "task" in slots and len(slots) == 1, "MultiTask metric currently only supports task constraints. Check 'MultiTaskMetric' cfg."
            
            # Assume other vals are global attributes across metrics.
            for k, v in cfg.items():
                if k not in metric:  # do not override explicit metric values
                    metric[k] = v

            metric["decoding"] = model.decoding  # For decoding reliant metrics like 'WER' or 'BLEU'

            # Instantiates metric. Make property of parent model
            metric = MultiTaskMetric.from_config_dict(metric)
            setattr(model, name, metric)

            # Tracking dicts for quick lookup
            self._metric_dict[name], self._slot_dict[name], self._skip_dict[name]  = metric, {**slots}, False


    # Performs full PyMetrics validation loop for all metrics
    def eval(
        self,
        batch: PromptedAudioToTextMiniBatch,
        predictions: torch.Tensor,
        predictions_lengths: torch.Tensor,
        predictions_mask: torch.Tensor,
        return_all_metrics: Optional[bool] = False,
        prefix: Optional[str] = None,
        suffix: Optional[str] = None,
    ):
        metric_dict = {}
        self.update(
            predictions=predictions,
            predictions_lengths=predictions_lengths,
            targets=batch.transcript,
            targets_lengths=batch.transcript_lens,
            predictions_mask=predictions_mask,
            input_ids=batch.prompt,
            cuts=batch.cuts,
        )
        metric_dict.update(
            self.compute(
                prefix=f"{prefix}_" if prefix else "",
                suffix=f"{suffix}_" if suffix else "",
                return_all_metrics=return_all_metrics
                )
            )
        self.reset()
        return metric_dict

    def update(
        self,
        predictions: torch.Tensor,
        predictions_lengths: torch.Tensor,
        predictions_mask: torch.Tensor,
        targets: torch.Tensor,
        targets_lengths: torch.Tensor,
        input_ids: torch.Tensor,
        cuts: Optional[CutSet] = None,
    ):
        # Just iterate cuts to avoid expensive reindexing
        if cuts is not None:
            cuts = [c for c in cuts]

        for name, metric in self._metric_dict.items():
            indices = self.split_task_indices(input_ids, self._slot_dict[name])
            if indices.numel() == 0:  # No instances of metric in this tensor, skip
                self._skip_dict[name] = True
                continue
            
            # bleu metric allos you to pass cuts
            cuts_idx = None
            if cuts is not None:
                indices_cpu = indices.cpu().tolist()
                cuts_idx = [cuts[idx] for idx in indices_cpu]

            metric.update(
                predictions=predictions[indices],
                predictions_lengths=predictions_lengths[indices],
                predictions_mask=predictions_mask[indices],
                targets=targets[indices],
                targets_lengths=targets_lengths[indices],
                input_ids=input_ids[indices],
                cuts=cuts_idx,
            )
                
    def compute(self, return_all_metrics=False, prefix="", suffix=""):
        output_dict = {}
        for name, metric in self._metric_dict.items():
            # Check if update marked this metric empty for batch
            # Since ASR models do full loop per step this ignores breaking behavior
            if self._skip_dict[name]:
                self._skip_dict[name] = False
                continue

            # TODO: Change behavior of WER
            # so it has a dict output
            if name == "wer":
                wer, wer_num, wer_denom = metric.compute()
                if return_all_metrics:
                    output_dict.update(
                        {
                            f"{prefix}wer{suffix}": wer,
                            f"{prefix}wer_num{suffix}": wer_num,
                            f"{prefix}wer_denom{suffix}": wer_denom,
                        }
                    )
                else:
                    output_dict.update(
                        {
                            f"{prefix}wer{suffix}": wer,
                            }
                    )
            else:
                output_dict.update(
                    metric.compute(
                        return_all_metrics=return_all_metrics,
                        prefix=prefix,
                        suffix=suffix,
                    )
                )
        return output_dict

    def reset(self):
        {metric.reset() for name, metric in self._metric_dict.items()}

    # TODO: Add properties to `Canary` to simply return `task` idx.
    def _canary_index_fn(self, prompt_ids: torch.Tensor, slots: Dict[str, (str | bool)]) -> torch.Tensor:
        if slots["task"] in TASK_TRANSLATE:
            # 1 -> `source_lang` in canary, 3 -> 'target_lang. Use these instead of task ID to avoid lookup.
            condition_met = prompt_ids[:,1] != prompt_ids[:,3]
        else:  # default to transcribe
            condition_met = prompt_ids[:,1] == prompt_ids[:,3]
        indices =  torch.nonzero(condition_met, as_tuple=False)
        # reshape in case 0 dim
        return indices.reshape(indices.numel())
    
    # TODO: Add properties to `Canary2` to simply return `task` idx.
    def _canary2_index_fn(self, prompt_ids: torch.Tensor, slots: Dict[str, str | bool]) -> torch.Tensor:
        # Canary2 has variable prompt length, use bos as offset.
        bos_idx = (prompt_ids == self.prompt.tokenizer.bos_id).nonzero(as_tuple=True)[1]

        # 2 -> `source_lang`, 3 -> 'target_lang`
        bos_idx = bos_idx.unsqueeze(1)  # for gather
        src_lang, tgt_lang = prompt_ids.gather(1, bos_idx + 2), prompt_ids.gather(1, bos_idx + 3)
        src_lang, tgt_lang = src_lang.view([prompt_ids.shape[0]]), tgt_lang.view([prompt_ids.shape[0]])
        if slots["task"] in TASK_TRANSLATE:
            condition_met = src_lang != tgt_lang
        else:  # default to transcribe
            condition_met = src_lang == tgt_lang
        indices =  torch.nonzero(condition_met, as_tuple=False)
        # reshape in case 0 dimen
        return indices.reshape(indices.numel())