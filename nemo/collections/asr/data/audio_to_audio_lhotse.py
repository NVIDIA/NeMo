import torch
from lhotse import CutSet
from lhotse.cut import MixedCut
from lhotse.dataset.collation import collate_audio, collate_custom_field


class LhotseAudioToTargetDataset(torch.utils.data.Dataset):
    """
    A dataset for audio-to-audio tasks where the goal is to use
    an input signal to recover the corresponding target signal.

    .. note:: This is a Lhotse variant of :class:`nemo.collections.asr.data.audio_to_audio.AudioToTargetDataset`.
    """

    TARGET_KEY = "target_recording"
    REFERENCE_KEY = "reference_recording"
    EMBEDDING_KEY = "embedding_vector"

    def __getitem__(self, cuts: CutSet) -> dict[str, torch.Tensor]:
        src_audio, src_audio_lens = collate_audio(cuts)
        tgt_audio, tgt_audio_lens = collate_audio(cuts, recording_field=self.TARGET_KEY)
        ans = {
            "input_signal": src_audio,
            "input_length": src_audio_lens,
            "target_signal": tgt_audio,
            "target_length": tgt_audio_lens,
        }
        if _key_available(self.REFERENCE_KEY):
            ref_audio, ref_audio_lens = collate_audio(cuts, recording_field=self.REFERENCE_KEY)
            ans.update(reference_signal=ref_audio, reference_length=ref_audio_lens)
        if _key_available(self.EMBEDDING_KEY):
            emb = collate_custom_field(cuts, field=self.EMBEDDING_KEY)
            ans.update(embedding_signal=emb)
        return ans


def _key_available(cuts: CutSet, key: str) -> bool:
    for cut in cuts:
        if isinstance(cut, MixedCut):
            cut = cut._first_non_padding_cut
        if cut.custom is not None and key in cut.custom:
            continue
        else:
            return False
    return True
