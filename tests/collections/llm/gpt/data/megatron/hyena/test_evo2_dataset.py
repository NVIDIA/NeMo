# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2024 Arc Institute. All rights reserved.
# Copyright (c) 2024 Michael Poli. All rights reserved.
# Copyright (c) 2024 Stanford University. All rights reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import pytest
import torch
from nemo.collections.llm.gpt.data.megatron.hyena.evo2_dataset import Evo2Dataset, Evo2DatasetPadEodLossMask

"""
The tag token is constructed as follows: So note that one way to know you have a tag is if you look at the first
token after the pipe and it is a 'd' character. Make sure tests are consistent with this simplification.
    @staticmethod
    def _construct_taxonomy_token(
        lineage: Evo2TaxonomyLineage, dropout: float = 0.0, seed: Optional[int] = None
    ) -> Optional[str]:
        '''Construct a special Taxonomy token for natural language prompting of DNA generation models.

        Args:
            lineage (Evo2TaxonomyLineage): The taxonomy lineage information.
            dropout (float): The probability of dropping out segments of the lineage. Defaults to 0.0.
            seed (Optional[int]): The seed for the random number generator. Defaults to None.

        Returns:
            Optional[str]: The constructed taxonomy token or None if lineage is None.
        '''
        # If dropout > 0, randomly drop out segments of the lineage for training on incomplete lineages.
        with Evo2Preprocessor.preprocessing_context_manager(seed if seed is not None else None):
            return (
                "|d__{};p__{};c__{};o__{};f__{};g__{};s__{}|".format(
                    lineage.kingdom if random.random() >= dropout else None,
                    lineage.phylum if random.random() >= dropout else None,
                    lineage.clazz if random.random() >= dropout else None,
                    lineage.order if random.random() >= dropout else None,
                    lineage.family if random.random() >= dropout else None,
                    lineage.genus if random.random() >= dropout else None,
                    lineage.species if random.random() >= dropout else None,
                )
                if lineage is not None
                else None
            )
"""
@pytest.fixture
def tag_tokens():
    """Standard tokens for phylogenetic tag tests, defined in Evo2_DataseT:

    CONTROL_TAGS: ClassVar[list[int]] = [64, 35]  # '@' tag for splice splits/windows, '#' for contig splits
    TAG_BOUNDS = 124  # start and end delim: '|'
    TAG_CHARS: ClassVar[set[int]] = {95, 59, 32}  # chars only found in control tags: _, ;, space
    DEFAULT_EOD = 0
    """
    return {
        "terminal": 124,  # |
        "other_chars": {95, 59, 32},  # _, ;, space
        "eod": 0,  # end of document token
    }


def test_mask_phylogenetic_tags_with_eod(tag_tokens):
    """
    Tests a sequence where an EOD splits two partial tags.

    Example sequence (ASCII):
      65       124   100    0     124   65
      'A'      '|'   'd'   EOD   '|'   'A'

    - Segment 1: "A|d" => keep 'A' (DNA), mask '|' and 'd'
    - EOD => masked
    - Segment 2: "|A" => mask '|', keep 'A' (DNA)

    Expected masking: [1, 0, 0, 1, 0, 1]
    """
    sequence = torch.tensor([65, 124, 100, 0, 124, 65])  # "A|d" + EOD + "|A"

    mask = Evo2Dataset.mask_phylogenetic_tags(
        tokenized_sequence=sequence,
        terminal_tag_char=tag_tokens["terminal"],  # '|'
        other_tag_chars=tag_tokens["other_chars"],  # { '_',';',' ' }
        eod_token_id=tag_tokens["eod"],            # 0
    )

    expected_mask = torch.tensor([1, 0, 0, 1, 0, 1])
    assert torch.equal(mask, expected_mask)


def test_mask_phylogenetic_tags_middle(tag_tokens):
    """Tests masking a phylogenetic tag that appears in the middle of a DNA sequence.

    The sequence contains:
    1. Normal DNA (ATG)
    2. A phylo tag (|d_|)
    3. More DNA (TCGA)

    Expected behavior: The DNA should be unmasked (1s) while everything between
    and including the pipe characters should be masked (0s), as it's a valid phylo tag.
    """
    sequence = torch.tensor(
        [
            65,
            84,
            71,  # ATG
            124,
            100,
            110,
            102,
            111,
            95,
            116,
            97,
            103,
            124,  # |d__tag|
            84,
            67,
            71,
            65,  # TCGA
        ]
    )

    mask = Evo2Dataset.mask_phylogenetic_tags(
        tokenized_sequence=sequence,
        terminal_tag_char=tag_tokens["terminal"],  # |
        other_tag_chars=tag_tokens["other_chars"],  # _, ;, space
        eod_token_id=tag_tokens["eod"],
    )

    expected_mask = torch.tensor(
        [
            1,
            1,
            1,  # DNA unmasked
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,  # phylo tag masked
            1,
            1,
            1,
            1,  # DNA unmasked
        ]
    )
    assert torch.equal(mask, expected_mask)


def test_mask_partial_tag_start(tag_tokens):
    """Tests handling a sequence that starts with a partial phylogenetic tag.

    The sequence starts with characters that would be inside a phylo tag,
    followed by a closing pipe and DNA. Since we want to prevent the model from
    learning non-DNA outputs, we mask all potential tag characters even without
    complete tag delimiters.

    Sequence: "tag;_|ATG" (starting mid-tag)
    Expected: All tag characters and delimiters masked, only DNA unmasked
    """
    sequence = torch.tensor(
        [
            116,
            97,
            103,
            59,
            95,  # tag;_
            124,  # |
            65,
            84,
            71,  # ATG
        ]
    )

    mask = Evo2Dataset.mask_phylogenetic_tags(
        tokenized_sequence=sequence,
        terminal_tag_char=tag_tokens["terminal"],
        other_tag_chars=tag_tokens["other_chars"],
        eod_token_id=tag_tokens["eod"],
    )

    expected_mask = torch.tensor(
        [
            0,
            0,
            0,
            0,
            0,  # partial tag start masked
            0,  # closing pipe masked
            1,
            1,
            1,  # DNA unmasked
        ]
    )
    assert torch.equal(mask, expected_mask)


def test_mask_partial_tag_end(tag_tokens):
    """Tests handling a sequence that ends with a partial phylogenetic tag.

    The sequence contains DNA followed by an opening pipe and tag characters,
    but no closing pipe. Per requirements, we aggressively mask any potential
    tag characters to ensure the model only learns DNA bases {A,C,G,T}.

    Sequence: "ATG|info_" (ending mid-tag)
    Expected: DNA unmasked, all tag-related characters masked
    """
    sequence = torch.tensor(
        [
            65,
            84,
            71,  # ATG
            124,  # |
            100,
            110,
            102,
            111,
            95,  # info_
        ]
    )

    mask = Evo2Dataset.mask_phylogenetic_tags(
        tokenized_sequence=sequence,
        terminal_tag_char=tag_tokens["terminal"],
        other_tag_chars=tag_tokens["other_chars"],
        eod_token_id=tag_tokens["eod"],
    )

    expected_mask = torch.tensor(
        [
            1,
            1,
            1,  # DNA unmasked
            0,  # opening pipe masked
            0,
            0,
            0,
            0,
            0,  # partial tag end masked
        ]
    )
    assert torch.equal(mask, expected_mask)


def test_standalone_tag(tag_tokens):
    """Tests masking of a single complete tag with no surrounding sequence.

    Tests that a standalone tag (|tag_|) is fully masked since it contains
    non-DNA characters. This ensures the model only learns to output
    {A,C,G,T} tokens.

    Sequence: |tag_|
    Expected: All tokens masked (all zeros)
    """
    sequence = torch.tensor([124, 100, 97, 103, 95, 124])  # |dtag_|
    mask = Evo2Dataset.mask_phylogenetic_tags(
        tokenized_sequence=sequence,
        terminal_tag_char=tag_tokens["terminal"],
        other_tag_chars=tag_tokens["other_chars"],
        eod_token_id=tag_tokens["eod"],
    )
    expected = torch.tensor([0, 0, 0, 0, 0, 0])  # All masked
    assert torch.equal(mask, expected)


def test_sequence_starting_with_tag(tag_tokens):
    """Tests sequence that begins with a complete tag followed by DNA.

    Verifies that when a sequence starts with a complete tag followed by
    DNA bases, the tag portion is masked while the DNA portion remains
    unmasked.

    Sequence: |tag_|ATG
    Expected: Tag masked (zeros), DNA unmasked (ones)
    """
    sequence = torch.tensor(
        [
            124,
            100, #d token for domain
            97,
            103,
            95,
            124,  # |tag_|
            65,
            84,
            71,  # ATG
        ]
    )
    mask = Evo2Dataset.mask_phylogenetic_tags(
        tokenized_sequence=sequence,
        terminal_tag_char=tag_tokens["terminal"],
        other_tag_chars=tag_tokens["other_chars"],
        eod_token_id=tag_tokens["eod"],
    )
    expected = torch.tensor([0, 0, 0, 0, 0, 0, 1, 1, 1])  # Tag masked, DNA unmasked
    assert torch.equal(mask, expected)


def test_sequence_ending_with_tag(tag_tokens):
    """Tests sequence that ends with a complete tag.

    Verifies that when a sequence ends with a complete tag, the DNA portion
    remains unmasked while the entire tag portion is masked.

    Sequence: ATG|tag_|
    Expected: DNA unmasked (ones), tag masked (zeros)
    """
    sequence = torch.tensor(
        [
            65,
            84,
            71,  # ATG
            124,
            100,
            97,
            103,
            95,
            124,  # |tag_|
        ]
    )
    mask = Evo2Dataset.mask_phylogenetic_tags(
        tokenized_sequence=sequence,
        terminal_tag_char=tag_tokens["terminal"],
        other_tag_chars=tag_tokens["other_chars"],
        eod_token_id=tag_tokens["eod"],
    )
    expected = torch.tensor([1, 1, 1, 0, 0, 0, 0, 0, 0])  # DNA unmasked, tag masked
    assert torch.equal(mask, expected)


def test_mask_multiple_tags(tag_tokens):
    """Tests handling multiple phylogenetic tags in sequence, demonstrating state transitions.

    This tests how the masking switches states between phylo and non-phylo regions:
    1. Starts in non-phylo state with DNA
    2. Switches to phylo state at first pipe (with tag chars)
    3. Switches back to non-phylo at closing pipe
    4. Pattern repeats for second tag

    Sequence: "ATG|tag_1|CG|tag_2|AT"
    Expected: Only DNA sequences should remain unmasked
    """
    sequence = torch.tensor(
        [
            65,
            84,
            71,  # ATG
            124,
            100,
            97,
            103,
            95,
            49,
            124,  # |tag_1|
            67,
            71,  # CG
            124,
            100,
            97,
            103,
            95,
            50,
            124,  # |tag_2|
            65,
            84,  # AT
        ]
    )

    mask = Evo2Dataset.mask_phylogenetic_tags(
        tokenized_sequence=sequence,
        terminal_tag_char=tag_tokens["terminal"],
        other_tag_chars=tag_tokens["other_chars"],
        eod_token_id=tag_tokens["eod"],
    )

    expected_mask = torch.tensor(
        [
            1,
            1,
            1,  # DNA unmasked
            0,
            0,
            0,
            0,
            0,
            0,
            0,  # first tag masked
            1,
            1,  # DNA unmasked
            0,
            0,
            0,
            0,
            0,
            0,
            0,  # second tag masked
            1,
            1,  # DNA unmasked
        ]
    )
    assert torch.equal(mask, expected_mask)


def test_mask_dna_after_pipe(tag_tokens):
    """Tests the scenario where we have a pipe followed by DNA sequence.

    This tests the edge case of a pipe character appearing at the start of a sequence.
    Even if DNA follows, we mask the pipe character to prevent the model from
    learning to output non-DNA tokens.

    Sequence: "|ATG" (pipe followed by DNA)
    Expected: Pipe masked, DNA unmasked
    """
    sequence = torch.tensor(
        [
            124,  # |
            65,
            84,
            71,  # ATG
        ]
    )

    mask = Evo2Dataset.mask_phylogenetic_tags(
        tokenized_sequence=sequence,
        terminal_tag_char=tag_tokens["terminal"],
        other_tag_chars=tag_tokens["other_chars"],
        eod_token_id=tag_tokens["eod"],
    )

    expected_mask = torch.tensor([0, 1, 1, 1])  # Pipe masked, DNA unmasked
    assert torch.equal(mask, expected_mask)


def test_ambiguous_dna_char_followed_by_tag_start(tag_tokens):
    """Tests handling of an ambiguous DNA character followed by a tag start.

    When we see a character that could be either DNA or the end of a truncated tag
    followed by a pipe, we should mask both for safety since we can't disambiguate
    whether the character was part of a tag.

    Sequence: "t|AAAT" (t could be DNA or end of tag)
    Expected: First t and pipe masked (0), AAAT unmasked (1)
    """
    sequence = torch.tensor(
        [
            116,  # t (ambiguous - could be DNA or end of tag)
            124,  # |
            65,  # A
            65,  # A
            65,  # A
            84,  # T
        ]
    )

    mask = Evo2Dataset.mask_phylogenetic_tags(
        tokenized_sequence=sequence,
        terminal_tag_char=tag_tokens["terminal"],
        other_tag_chars=tag_tokens["other_chars"],
        eod_token_id=tag_tokens["eod"],
    )

    expected_mask = torch.tensor([0, 0, 1, 1, 1, 1])  # Ambiguous t and pipe masked, DNA unmasked
    assert torch.equal(mask, expected_mask)


def test_dna_followed_by_unambiguous_tag_start(tag_tokens):
    """Tests handling of DNA sequence followed by clear tag start.

    When we see DNA followed by |d, it's unambiguous - the d clearly indicates
    the start of a phylogenetic tag (d__), so we can safely unmask the DNA and
    mask the tag portion.

    Sequence: "AAAT|d" (AAAT is DNA, |d starts tag)
    Expected: AAAT unmasked (1), |d masked (0)
    """
    sequence = torch.tensor(
        [
            65,  # A
            65,  # A
            65,  # A
            84,  # T
            124,  # |
            100,  # d (clearly starts d__ tag)
        ]
    )

    mask = Evo2Dataset.mask_phylogenetic_tags(
        tokenized_sequence=sequence,
        terminal_tag_char=tag_tokens["terminal"],
        other_tag_chars=tag_tokens["other_chars"],
        eod_token_id=tag_tokens["eod"],
    )

    expected_mask = torch.tensor([1, 1, 1, 1, 0, 0])  # DNA unmasked, tag start masked
    assert torch.equal(mask, expected_mask)


def test_double_partial_tags_with_dna_middle(tag_tokens):
    """Tests a sequence that has partial tags at both ends with DNA in the middle.

    Tests the specific case where a sequence slice cuts through phylogenetic tags
    on both ends, with valid DNA sequence in the middle. The behavior we want is:
    1. The partial tag at the start should be masked
    2. The DNA in the middle should be unmasked
    3. The partial tag at the end should be masked

    Sequence: "cacata|acagataaaataTACAGGGAATA|d__"
    Expected: First partial tag masked (0s), middle DNA unmasked (1s), end tag masked (0s)
    """
    sequence = torch.tensor(
        [
            99,
            97,
            99,
            97,
            116,
            97,  # cacata
            124,  # |
            97,
            99,
            97,
            103,
            97,
            116,
            97,
            97,
            97,
            97,
            116,
            97,  # acagataaaata
            84,
            65,
            67,
            65,
            71,
            71,
            71,
            65,
            65,
            84,
            65,  # TACAGGGAATA
            124,  # |
            100,
            95,
            95,  # d__
        ]
    )

    mask = Evo2Dataset.mask_phylogenetic_tags(
        tokenized_sequence=sequence,
        terminal_tag_char=tag_tokens["terminal"],
        other_tag_chars=tag_tokens["other_chars"],
        eod_token_id=tag_tokens["eod"],
    )

    expected_mask = torch.tensor(
        [
            0,
            0,
            0,
            0,
            0,
            0,  # partial start tag masked
            0,  # pipe masked
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,  # middle DNA unmasked
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,  # middle DNA unmasked
            0,  # pipe masked
            0,
            0,
            0,  # partial end tag masked
        ]
    )

    assert torch.equal(mask, expected_mask)


def test_packed_partial_tag_subsequence_predna(tag_tokens):
    """
    Sequence: "GAATA[EOD]cacata|acagataaaataTACAGGGAATA|d__"
    Expected: First partial tag masked (0s), middle DNA unmasked (1s), end tag masked (0s)

    """
    sequence_alpha = "GAATA0cacata|acagataaaataTACAGGGAATA|d__"
    sequence = torch.tensor([ord(t) if t != "0" else 0 for t in sequence_alpha], dtype=torch.int32)
    expected_mask = torch.tensor(
        len("GAATA0") * [1] + [0] * len("cacata|") + len("acagataaaataTACAGGGAATA") * [1] + [0] * len("|d__"),
        dtype=torch.int32,
    )
    mask = Evo2Dataset.mask_phylogenetic_tags(
        tokenized_sequence=sequence,
        terminal_tag_char=tag_tokens["terminal"],
        other_tag_chars=tag_tokens["other_chars"],
        eod_token_id=tag_tokens["eod"],
    )
    torch.testing.assert_close(mask, expected_mask)


def test_packed_partial_tag_subsequence_pretag(tag_tokens):
    """
    Sequence: "cacata|[EOD]acagataaaataTACAGGGAATA|d__"
    Expected: First partial tag masked (0s), middle DNA unmasked (1s), end tag masked (0s)

    """
    sequence_alpha = "cacata|0acagataaaataTACAGGGAATA|d__"
    sequence = torch.tensor([ord(t) if t != "0" else 0 for t in sequence_alpha], dtype=torch.int32)
    expected_mask = torch.tensor(
        len("cacata") * [1] + [0] + [1] * len("0acagataaaataTACAGGGAATA") + len("|d__") * [0], dtype=torch.int32
    )
    mask = Evo2Dataset.mask_phylogenetic_tags(
        tokenized_sequence=sequence,
        terminal_tag_char=tag_tokens["terminal"],
        other_tag_chars=tag_tokens["other_chars"],
        eod_token_id=tag_tokens["eod"],
    )
    torch.testing.assert_close(mask, expected_mask)


def test_packed_partial_tag_subsequence_predna_middletag(tag_tokens):
    """
    Sequence: "GAATA[EOD]cacata|acagataaaata|d__tag;|TACAGGGAATA|d__"
    Expected: First partial tag masked (0s), middle DNA unmasked (1s), end tag masked (0s)

    """
    sequence_alpha = "GAATA0cacata|acagataaaata|d__tag;|TACAGGGAATA|d__"
    sequence = torch.tensor([ord(t) if t != "0" else 0 for t in sequence_alpha], dtype=torch.int32)
    expected_mask = torch.tensor(
        len("GAATA0") * [1]
        + len("cacata|") * [0]
        + [1] * len("acagataaaata")
        + len("|d__tag;|") * [0]
        + len("TACAGGGAATA") * [1]
        + len("|d__") * [0],
        dtype=torch.int32,
    )
    mask = Evo2Dataset.mask_phylogenetic_tags(
        tokenized_sequence=sequence,
        terminal_tag_char=tag_tokens["terminal"],
        other_tag_chars=tag_tokens["other_chars"],
        eod_token_id=tag_tokens["eod"],
    )
    torch.testing.assert_close(mask, expected_mask)


def test_packed_partial_tag_subsequence_pretag_middletag(tag_tokens):
    """
    Sequence: "cacata|[EOD]acagataaaata|d__tag;|TACAGGGAATA|d__"
    Expected: First partial tag masked (0s), middle DNA unmasked (1s), end tag masked (0s)

    """
    sequence_alpha = "cacata|0acagataaaata|d__tag;|TACAGGGAATA|d__"
    sequence = torch.tensor([ord(t) if t != "0" else 0 for t in sequence_alpha], dtype=torch.int32)
    expected_mask = torch.tensor(
        len("cacata") * [1] + [0] # masked pipe.
        + [1] * len("0acagataaaata")
        + len("|d__tag;|") * [0]
        + len("TACAGGGAATA") * [1]
        + len("|d__") * [0],
        dtype=torch.int32,
    )
    mask = Evo2DatasetPadEodLossMask.mask_phylogenetic_tags(
        tokenized_sequence=sequence,
        terminal_tag_char=tag_tokens["terminal"],
        other_tag_chars=tag_tokens["other_chars"],
        eod_token_id=tag_tokens["eod"],
    )
    torch.testing.assert_close(mask, expected_mask)

def test_packed_partial_tag_subsequence_pretag_middletag_bs2(tag_tokens):
    """
    Sequence: "cacata|[EOD]acagataaaata|d__tag;|TACAGGGAATA|d__"
    Expected: First partial tag masked (0s), middle DNA unmasked (1s), end tag masked (0s)

    """
    sequence_alpha = "cacata|0acagataaaata|d__tag;|TACAGGGAATA|d__"
    sequence = torch.tensor([ord(t) if t != "0" else 0 for t in sequence_alpha], dtype=torch.int32)
    expected_mask = torch.tensor(
        len("cacata") * [1] + [0]
        + [1] * len("0acagataaaata")
        + len("|d__tag;|") * [0]
        + len("TACAGGGAATA") * [1]
        + len("|d__") * [0],
        dtype=torch.int32,
    )
    expected_mask = torch.stack([expected_mask, expected_mask])
    mask = Evo2DatasetPadEodLossMask.mask_phylogenetic_tags(
        tokenized_sequence=torch.stack([sequence, sequence]),
        terminal_tag_char=tag_tokens["terminal"],
        other_tag_chars=tag_tokens["other_chars"],
        eod_token_id=tag_tokens["eod"],
    )
    torch.testing.assert_close(mask, expected_mask)

def test_packed_partial_tag_subsequence_pretag_middletag_bs3(tag_tokens):
    """
    Sequence: "cacata|[EOD]acagataaaata|d__tag;|TACAGGGAATA|d__"
    Expected: First partial tag masked (0s), middle DNA unmasked (1s), end tag masked (0s)

    """
    sequence_alpha = "cacata|0acagataaaata|d__tag;|TACAGGGAATA|d__somet"
    sequence = torch.tensor([ord(t) if t != "0" else 0 for t in sequence_alpha], dtype=torch.int32)
    expected_mask = torch.tensor(
        len("cacata") * [1] + [0]
        + [1] * len("0acagataaaata")
        + len("|d__tag;|") * [0]
        + len("TACAGGGAATA") * [1]
        + len("|d__somet") * [0],
        dtype=torch.int32,
    )

    sequence_alpha2 = "GAATA0cacata|acagataaaata|d__tag;|TACAGGGAATA|d__"
    sequence2 = torch.tensor([ord(t) if t != "0" else 0 for t in sequence_alpha2], dtype=torch.int32)
    expected_mask2 = torch.tensor(
        len("GAATA0") * [1]
        + len("cacata|") * [0]
        + [1] * len("acagataaaata")
        + len("|d__tag;|") * [0]
        + len("TACAGGGAATA") * [1]
        + len("|d__") * [0],
        dtype=torch.int32,
    )

    expected_mask = torch.stack([expected_mask, expected_mask, expected_mask2])

    mask = Evo2DatasetPadEodLossMask.mask_phylogenetic_tags(
        tokenized_sequence=torch.stack([sequence, sequence, sequence2]),
        terminal_tag_char=tag_tokens["terminal"],
        other_tag_chars=tag_tokens["other_chars"],
        eod_token_id=tag_tokens["eod"],
    )
    torch.testing.assert_close(mask, expected_mask)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_packed_partial_tag_subsequence_pretag_middletag_bs3_cuda(tag_tokens):
    sequence_alpha = "cacata|0acagataaaata|d__tag;|TACAGGGAATA|d__somet"
    sequence = torch.tensor([ord(t) if t != "0" else 0 for t in sequence_alpha], dtype=torch.int32)
    expected_mask = torch.tensor(
        len("cacata") * [1] + [0]
        + [1] * len("0acagataaaata")
        + len("|d__tag;|") * [0]
        + len("TACAGGGAATA") * [1]
        + len("|d__somet") * [0],
        dtype=torch.int32,
    )

    sequence_alpha2 = "GAATA0cacata|acagataaaata|d__tag;|TACAGGGAATA|d__"
    sequence2 = torch.tensor([ord(t) if t != "0" else 0 for t in sequence_alpha2], dtype=torch.int32)
    expected_mask2 = torch.tensor(
        len("GAATA0") * [1]
        + len("cacata|") * [0]
        + [1] * len("acagataaaata")
        + len("|d__tag;|") * [0]
        + len("TACAGGGAATA") * [1]
        + len("|d__") * [0],
        dtype=torch.int32,
    )

    expected_mask = torch.stack([expected_mask, expected_mask, expected_mask2])

    mask = Evo2DatasetPadEodLossMask.mask_phylogenetic_tags(
        tokenized_sequence=torch.stack([sequence, sequence, sequence2]).cuda(),
        terminal_tag_char=tag_tokens["terminal"],
        other_tag_chars=tag_tokens["other_chars"],
        eod_token_id=tag_tokens["eod"],
    )
    torch.testing.assert_close(mask.cpu(), expected_mask)

def test_multiple_packed_tags(tag_tokens):
    """
    Tests a sequence with multiple packed tags.
    """
    sequence_alpha = "|d__tag;|0|d__tag;|0|d__somet"
    sequence = torch.tensor([ord(t) if t != "0" else 0 for t in sequence_alpha], dtype=torch.int32)
    expected_mask = torch.tensor(
        len("|d__tag;|") * [0] + len("0") * [1] + len("|d__tag;|") * [0] + len("0") * [1] + len("|d__somet") * [0],
        dtype=torch.int32,
    )
    mask = Evo2DatasetPadEodLossMask.mask_phylogenetic_tags(
        tokenized_sequence=sequence,
        terminal_tag_char=tag_tokens["terminal"],
        other_tag_chars=tag_tokens["other_chars"],
        eod_token_id=tag_tokens["eod"],
    )
    torch.testing.assert_close(mask, expected_mask)

def test_multiple_eods(tag_tokens):
    """
    Tests a sequence with multiple EODs.
    """
    sequence_alpha = "ACGT0tacg0"
    sequence = torch.tensor([ord(t) if t != "0" else 0 for t in sequence_alpha], dtype=torch.int32)
    expected_mask = torch.tensor(len(sequence_alpha) * [1], dtype=torch.int32)
    mask = Evo2DatasetPadEodLossMask.mask_phylogenetic_tags(
        tokenized_sequence=sequence,
        terminal_tag_char=tag_tokens["terminal"],
        other_tag_chars=tag_tokens["other_chars"],
        eod_token_id=tag_tokens["eod"],
    )
    torch.testing.assert_close(mask, expected_mask)


def test_multiple_eods_prefix_no_suffix(tag_tokens):
    """
    Tests a sequence with multiple EODs.
    """
    sequence_alpha = "0ACGT0tacg0aa"
    sequence = torch.tensor([ord(t) if t != "0" else 0 for t in sequence_alpha], dtype=torch.int32)
    expected_mask = torch.tensor(len(sequence_alpha) * [1], dtype=torch.int32)
    mask = Evo2DatasetPadEodLossMask.mask_phylogenetic_tags(
        tokenized_sequence=sequence,
        terminal_tag_char=tag_tokens["terminal"],
        other_tag_chars=tag_tokens["other_chars"],
        eod_token_id=tag_tokens["eod"],
    )
    torch.testing.assert_close(mask, expected_mask)

def test_no_eods_with_batch(tag_tokens):
    """
    Tests a sequence with multiple EODs.
    """
    sequence_alpha = "ACATAGATTT"
    sequence = torch.tensor([ord(t) if t != "0" else 0 for t in sequence_alpha], dtype=torch.int32)
    expected_mask = torch.tensor(len(sequence_alpha) * [1], dtype=torch.int32)
    mask = Evo2DatasetPadEodLossMask.mask_phylogenetic_tags(
        tokenized_sequence=torch.stack([sequence, sequence]),
        terminal_tag_char=tag_tokens["terminal"],
        other_tag_chars=tag_tokens["other_chars"],
        eod_token_id=tag_tokens["eod"],
    )
    torch.testing.assert_close(mask, torch.stack([expected_mask, expected_mask]))

def test_no_eods_one_tag_with_batch_bs2(tag_tokens):
    """
    Tests a sequence with multiple EODs.
    """
    sequence_alpha = "ACAT|d__tag;|AGATTT"
    sequence = torch.tensor([ord(t) if t != "0" else 0 for t in sequence_alpha], dtype=torch.int32)
    expected_mask = torch.tensor(len("ACAT") * [1] + len("|d__tag;|") * [0] + len("AGATTT") * [1], dtype=torch.int32)
    mask = Evo2DatasetPadEodLossMask.mask_phylogenetic_tags(
        tokenized_sequence=torch.stack([sequence, sequence]),
        terminal_tag_char=tag_tokens["terminal"],
        other_tag_chars=tag_tokens["other_chars"],
        eod_token_id=tag_tokens["eod"],
    )
    torch.testing.assert_close(mask, torch.stack([expected_mask, expected_mask]))
