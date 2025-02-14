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

from typing import ClassVar, Dict, Optional

import torch
from megatron.core.datasets.gpt_dataset import GPTDataset
from nemo.collections.llm.gpt.model.megatron.hyena.hyena_utils import make_upper_case

class Evo2Dataset(GPTDataset):
    """Dataset for training Evo2."""

    CONTROL_TAGS: ClassVar[list[int]] = [64, 35]  # '@' tag for splice splits/windows, '#' for contig splits
    TAG_BOUNDS = 124  # start and end delim: '|'
    TAG_CHARS: ClassVar[set[int]] = {95, 59, 32}  # chars only found in control tags: _, ;, space
    DEFAULT_EOD = 0
    TO_UPPER_TOKENS: bool = True  # If set, do an in-place transform to make all tokens capital letters
    RESET_PAD_EOD_MASK: bool = True  # If set, unset the mask for [pad] and [eod] tokens (matches Evo2 paper).

    def __getitem__(self, idx: Optional[int]) -> Dict[str, torch.Tensor]:
        """Get data at the specified index."""
        # 1. Call the default gpt dataset object 
        databatch: dict = super().__getitem__(idx)
        loss_mask = databatch.get("loss_mask", None)
        if self.RESET_PAD_EOD_MASK and loss_mask is not None:
            # Reset the mask for 'pad', '[eod]', '[pad token]', which will lower the loss, but matches Evo2 pub.
            loss_mask = torch.ones_like(loss_mask)
        labels = databatch.get("labels", None)
        if labels is None or loss_mask is None:
            # No next-token labels or loss to mask.
            return databatch

        # Mask special label tags in loss.
        control_mask = torch.isin(labels, torch.tensor(self.CONTROL_TAGS, device=labels.device))
        loss_mask[control_mask] = 0
        phylotag_mask = Evo2Dataset.mask_phylogenetic_tags(
            labels,
            self.TAG_BOUNDS,
            self.TAG_CHARS,
            self.config.tokenizer.eod if self.config.tokenizer is not None else self.DEFAULT_EOD,
        )
        databatch["loss_mask"] = loss_mask * phylotag_mask
        if self.TO_UPPER_TOKENS:
            databatch["tokens"], _  = make_upper_case(databatch["tokens"])
        return databatch

    @staticmethod
    def mask_phylogenetic_tags(
        tokenized_sequence: torch.Tensor,
        terminal_tag_char: int,
        other_tag_chars: set[int],
        eod_token_id: int,
    ) -> torch.Tensor:
        """Creates a mask for sequences containing phylogenetic taxonomic tags and DNA.
        
        This function processes sequences that contain both DNA data (A,C,G,T in uppercase or lowercase)
        and taxonomic information in the format |d__kingdom;p__phylum;c__class;...| to create a binary mask.
        The mask ensures that only DNA sequences are exposed (1) while taxonomic tags and related information
        are masked (0).

        Example:
            For input "|d__Bacteria|ACGT|s__species|":
            - Returns [0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0]
            - The DNA sequence ACGT is unmasked (1s)
            - The taxonomic tags and delimiters are masked (0s)

        The function handles several specific cases:
        1. Complete tags: Sequences between pipe characters containing taxonomic information
        2. Partial tags: Incomplete taxonomic information at sequence boundaries
        3. DNA sequences: Uppercase A,C,G,T characters that should remain unmasked
        4. Special tokens: EOD tokens within tag context that should be masked

        Args:
            tokenized_sequence (torch.Tensor): Input sequence tensor of shape (batch_size, seq_length)
                or (seq_length,). Contains ASCII values representing sequence characters.
            terminal_tag_char (int): ASCII value for the tag delimiter character ('|' = 124).
            other_tag_chars (set of int): Set of ASCII values for characters used in tags 
                (e.g., '_', ';', space).
            eod_token_id (int): Token ID representing end-of-document.

        Returns:
            torch.Tensor: Binary mask of the same shape as input where:
                1 = Keep (DNA sequences)
                0 = Mask (taxonomic tags and related information).
        """
        device = tokenized_sequence.device
        dtype = tokenized_sequence.dtype

        # Handle empty sequence.
        if tokenized_sequence.numel() == 0:
            return torch.ones(0, device=device, dtype=torch.int)
        # Handle a single token.
        if tokenized_sequence.numel() == 1:
            mask = torch.ones(1, device=device, dtype=torch.int)
            token = tokenized_sequence.item()
            if token == terminal_tag_char or token in other_tag_chars:
                mask[0] = 0
            return mask

        batched_io = (tokenized_sequence.ndim == 2)
        if not batched_io:
            tokenized_sequence = tokenized_sequence.unsqueeze(0)
        batch_size, seq_len = tokenized_sequence.shape

        # Create constant tensors
        other_tag_tensor = torch.tensor(list(other_tag_chars), device=device, dtype=dtype)
        taxonomy_prefixes = torch.tensor([100, 112, 99, 111, 102, 103, 115], device=device, dtype=dtype)
        valid_dna = torch.tensor([65, 67, 71, 84, 78, 97, 99, 103, 116, 110], device=device, dtype=dtype)

        # Initialize output mask
        mask_vector = torch.ones_like(tokenized_sequence, dtype=torch.int)

        # Process each sequence
        for i in range(batch_size):
            row = tokenized_sequence[i]

            # Compute in_tag status
            in_tag = (torch.cumsum((row == terminal_tag_char).to(torch.int), dim=0) % 2) == 1

            # Find EOD tokens outside tags
            eod_outside = (row == eod_token_id) & (~in_tag)

            # Create segment boundaries
            shifted = torch.roll(eod_outside.to(torch.int64), 1)
            shifted[0] = 0
            seg_ids = torch.cumsum(shifted, dim=0)

            # Process each segment
            for seg in torch.unique(seg_ids):
                seg_idx = (seg_ids == seg).nonzero(as_tuple=True)[0]
                seg_seq = row[seg_idx]

                # Initialize segment mask
                seg_mask = torch.ones_like(seg_seq, dtype=torch.int)

                # Find terminals in segment
                term_mask = (seg_seq == terminal_tag_char)
                term_positions = torch.nonzero(term_mask, as_tuple=True)[0]

                # If no terminals but has tag chars, mask everything
                if not term_positions.numel():
                    if torch.any(torch.isin(seg_seq, other_tag_tensor)):
                        seg_mask.zero_()
                    mask_vector[i, seg_idx] = seg_mask
                    continue

                # Always mask terminal tokens
                seg_mask[term_mask] = 0

                # Handle region before first terminal
                first_pipe = term_positions[0].item()
                if first_pipe > 0:
                    prefix = seg_seq[:first_pipe]
                    if prefix[0].item() in taxonomy_prefixes.tolist() or \
                    (prefix.numel() == 1 and (97 <= prefix[0].item() <= 122)) or \
                    torch.any(torch.isin(prefix, other_tag_tensor)) or \
                    not torch.all(torch.isin(prefix, valid_dna)):
                        seg_mask[:first_pipe] = 0

                # Handle regions between terminals
                for j in range(len(term_positions) - 1):
                    start = term_positions[j].item()
                    end = term_positions[j + 1].item()
                    if torch.any(torch.isin(seg_seq[start + 1:end], other_tag_tensor)):
                        seg_mask[start + 1:end] = 0

                # Handle region after last terminal
                last_pipe = term_positions[-1].item()
                if last_pipe < len(seg_seq) - 1:
                    suffix = seg_seq[last_pipe + 1:]
                    if suffix.numel() > 0 and chr(suffix[0].item()) == 'd' or \
                    torch.any(torch.isin(suffix, other_tag_tensor)) or \
                    torch.any(suffix == eod_token_id):
                        seg_mask[last_pipe + 1:] = 0

                mask_vector[i, seg_idx] = seg_mask

        if not batched_io:
            mask_vector = mask_vector.squeeze(0)
        return mask_vector


class Evo2DatasetPadEodLossMask(Evo2Dataset):
    TO_UPPER_TOKENS: bool = True
    RESET_PAD_EOD_MASK: bool = False