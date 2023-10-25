from nemo_text_processing.text_normalization.normalize import Normalizer
import tempfile
from pathlib import Path
import wget

def get_normalizer():
    WHITELIST_URL = (
        "https://raw.githubusercontent.com/NVIDIA/NeMo-text-processing/main/"
        "nemo_text_processing/text_normalization/en/data/whitelist/lj_speech.tsv"
    )
    with tempfile.TemporaryDirectory() as data_dir:
        # data_dir = BASE_DIR / "data" / "normalizer"
        # data_dir.mkdir(parents=True, exist_ok=False)
        whitelist_path = Path(data_dir) / "lj_speech.tsv"
        if not whitelist_path.exists():
            wget.download(WHITELIST_URL, out=str(data_dir))

        normalizer = Normalizer(
            lang="en",
            input_case="cased",
            whitelist=str(whitelist_path),
            overwrite_cache=True,
            cache_dir=None,  # str(data_dir / "tts_cache_dir"),
        )
    return normalizer


def normalize(text, normalizer, do_lowercase):
    text_normalizer_call_kwargs = {"verbose": False, "punct_pre_process": True, "punct_post_process": True}
    normalized_text = normalizer.normalize(text, **text_normalizer_call_kwargs)
    if do_lowercase:
        normalized_text = normalized_text.lower()
    return normalized_text
