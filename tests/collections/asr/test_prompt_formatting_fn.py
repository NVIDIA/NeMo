import pytest
from lhotse import CutSet, SupervisionSegment
from lhotse.testing.dummies import dummy_cut

from nemo.collections.asr.data.audio_to_text_lhotse import TokenizerWrapper
from nemo.collections.asr.data.audio_to_text_lhotse_prompted import canary_natural
from nemo.collections.common.tokenizers.canary_tokenizer import CanaryTokenizer
from nemo.collections.common.tokenizers.sentencepiece_tokenizer import SentencePieceTokenizer, create_spt_model


TEMPLATE_PROMPT = "Transcribe in {target_lang} {pnc}. "
COMPLETE_PROMPT = "Transcribe in Latin with PnC. "
CUSTOM_LANG_PROMPT = "foo bar"
TEXT = "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum."


@pytest.fixture(scope="session")
def bpe_tokenizer(tmp_path_factory):
    tmpdir = tmp_path_factory.mktemp("tokenizer")
    text_path = tmpdir / "text.txt"
    text_path.write_text((COMPLETE_PROMPT + "\n") * 100 + TEXT)
    create_spt_model(
        text_path,
        vocab_size=128,
        sample_size=-1,
        do_lower_case=False,
        output_dir=str(tmpdir),
        bos=True,
        eos=True,
        pad=True,
    )
    return SentencePieceTokenizer(str(tmpdir / "tokenizer.model"))


@pytest.fixture(scope="session")
def prompt_tokenizer(tmp_path_factory):
    tmpdir = tmp_path_factory.mktemp("prompt_tokenizer")
    text_path = tmpdir / "text.txt"
    text_path.write_text(CUSTOM_LANG_PROMPT)
    create_spt_model(
        text_path,
        vocab_size=16,
        sample_size=-1,
        do_lower_case=False,
        output_dir=str(tmpdir),
        bos=True,
        eos=True,
        pad=True,
    )
    return SentencePieceTokenizer(str(tmpdir / "tokenizer.model"))


@pytest.fixture(scope="session")
def canary_tokenizer(tmp_path_factory, bpe_tokenizer, prompt_tokenizer):
    tmpdir = tmp_path_factory.mktemp("spl_tokenizer")
    spl_tokenizer = CanaryTokenizer.build_special_tokenizer(tokens=["Latin"], model_dir=tmpdir)
    return CanaryTokenizer({"spl_tokens": spl_tokenizer, "Latin": bpe_tokenizer, "prompt": prompt_tokenizer})


def test_canary_natural_prompt_filled_bpe(bpe_tokenizer):
    cut = dummy_cut(0)

    # The prompt will be tokenized using the only tokenizer we have.
    cut.prompt = COMPLETE_PROMPT
    cut.supervisions = [
        SupervisionSegment(
            cut.id, cut.recording_id, cut.start, cut.duration, text="Lorem ipsum dolor sit amet", language="Latin"
        )
    ]
    cuts = CutSet.from_cuts([cut])

    tokens, prompts = canary_natural(cuts=cuts, tokenizer=TokenizerWrapper(bpe_tokenizer), inference=False)

    # note: with BPE, 1 is bos and 2 is eos
    # fmt: off
    expected_tokens = [[1, 4, 13, 7, 19, 11, 18, 16, 6, 20, 12, 21, 5, 4, 17, 9, 10, 4, 15, 5, 14, 8, 12, 92, 27, 4, 40, 29, 39, 33, 4, 29, 9, 4, 55, 6, 25, 2]]
    assert tokens == expected_tokens

    expected_prompts = [[1, 4, 13, 7, 19, 11, 18, 16, 6, 20, 12, 21, 5, 4, 17, 9, 10, 4, 15, 5, 14, 8]]
    assert prompts == expected_prompts
    # fmt: on


def test_canary_natural_prompt_filled_agg(canary_tokenizer):
    cut = dummy_cut(0)

    # The prompt will be tokenized using language "Latin", same as cut/supervision.
    cut.prompt = COMPLETE_PROMPT
    cut.supervisions = [
        SupervisionSegment(
            cut.id, cut.recording_id, cut.start, cut.duration, text="Lorem ipsum dolor sit amet", language="Latin"
        )
    ]
    cuts = CutSet.from_cuts([cut])

    tokens, prompts = canary_natural(cuts=cuts, tokenizer=TokenizerWrapper(canary_tokenizer), inference=False)

    # note: with aggregate, 4 is bos and 3 is eos, and all non-special symbols
    #       are offset by the count of special tokens == 9 when compared to BPE
    # fmt: off
    expected_tokens = [[4, 13, 22, 16, 28, 20, 27, 25, 15, 29, 21, 30, 14, 13, 26, 18, 19, 13, 24, 14, 23, 17, 21, 101, 36, 13, 49, 38, 48, 42, 13, 38, 18, 13, 64, 15, 34, 3]]
    assert tokens == expected_tokens

    expected_prompts = [[4, 13, 22, 16, 28, 20, 27, 25, 15, 29, 21, 30, 14, 13, 26, 18, 19, 13, 24, 14, 23, 17]]
    assert prompts == expected_prompts
    # fmt: on


def test_canary_natural_prompt_template_bpe(bpe_tokenizer):
    cut = dummy_cut(0)

    # The prompt will be tokenized using the only tokenizer we have.
    cut.prompt = TEMPLATE_PROMPT
    cut.supervisions = [
        SupervisionSegment(
            cut.id, cut.recording_id, cut.start, cut.duration, text="Lorem ipsum dolor sit amet", language="Latin"
        )
    ]
    cut.target_lang = "Latin"
    cut.pnc = "with PnC"
    cuts = CutSet.from_cuts([cut])

    tokens, prompts = canary_natural(cuts=cuts, tokenizer=TokenizerWrapper(bpe_tokenizer), inference=False)

    # note: with BPE, 1 is bos and 2 is eos
    # fmt: off
    expected_tokens = [[1, 4, 13, 7, 19, 11, 18, 16, 6, 20, 12, 21, 5, 4, 17, 9, 10, 4, 15, 5, 14, 8, 12, 92, 27, 4, 40, 29, 39, 33, 4, 29, 9, 4, 55, 6, 25, 2]]
    assert tokens == expected_tokens

    expected_prompts = [[1, 4, 13, 7, 19, 11, 18, 16, 6, 20, 12, 21, 5, 4, 17, 9, 10, 4, 15, 5, 14, 8]]
    assert prompts == expected_prompts
    # fmt: on


def test_canary_natural_prompt_template_agg(canary_tokenizer):
    cut = dummy_cut(0)

    # The prompt will be tokenized using language "Latin", same as cut/supervision.
    cut.prompt = TEMPLATE_PROMPT
    cut.supervisions = [
        SupervisionSegment(
            cut.id, cut.recording_id, cut.start, cut.duration, text="Lorem ipsum dolor sit amet", language="Latin"
        )
    ]
    cut.target_lang = "Latin"
    cut.pnc = "with PnC"
    cuts = CutSet.from_cuts([cut])

    tokens, prompts = canary_natural(cuts=cuts, tokenizer=TokenizerWrapper(canary_tokenizer), inference=False)

    # note: with aggregate, 4 is bos and 3 is eos, and all non-special symbols
    #       are offset by the count of special tokens == 9 when compared to BPE
    # fmt: off
    expected_tokens = [[4, 13, 22, 16, 28, 20, 27, 25, 15, 29, 21, 30, 14, 13, 26, 18, 19, 13, 24, 14, 23, 17, 21, 101, 36, 13, 49, 38, 48, 42, 13, 38, 18, 13, 64, 15, 34, 3]]
    assert tokens == expected_tokens

    expected_prompts = [[4, 13, 22, 16, 28, 20, 27, 25, 15, 29, 21, 30, 14, 13, 26, 18, 19, 13, 24, 14, 23, 17]]
    assert prompts == expected_prompts
    # fmt: on


def test_canary_natural_prompt_template_agg_prompt_lang(canary_tokenizer):
    cut = dummy_cut(0)

    # Tells the tokenizer/prompt formatter to use a tokenizer for lang "prompt" to tokenize the prompt
    cut.prompt = CUSTOM_LANG_PROMPT
    cut.prompt_lang = "prompt"

    cut.supervisions = [
        SupervisionSegment(
            cut.id, cut.recording_id, cut.start, cut.duration, text="Lorem ipsum dolor sit amet", language="Latin"
        )
    ]
    cut.target_lang = "Latin"
    cut.pnc = "with PnC"
    cuts = CutSet.from_cuts([cut])

    tokens, prompts = canary_natural(cuts=cuts, tokenizer=TokenizerWrapper(canary_tokenizer), inference=False)

    # note: with aggregate, 4 is bos and 3 is eos, and all non-special symbols
    #       are offset by the count of special tokens == 9 when compared to BPE
    #       + since we're using an extra prompt tokenizer, there is an extra offset for the prompt tokens other than bos/eos
    # fmt: off
    expected_tokens = [[4, 115, 118, 114, 114, 115, 117, 116, 119, 21, 101, 36, 13, 49, 38, 48, 42, 13, 38, 18, 13, 64, 15, 34, 3]]
    assert tokens == expected_tokens

    expected_prompts = [[4, 115, 118, 114, 114, 115, 117, 116, 119]]
    assert prompts == expected_prompts
    # fmt: on
