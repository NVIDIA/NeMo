# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

import torch
from typing import Optional


class MultiLayerPerceptron(torch.nn.Module):
    """
    A simple MLP that can either be used independently or put on top
    of pretrained models (such as BERT) and act as a classifier.
    Args:
        hidden_size (int): the size of each layer
        num_classes (int): number of output classes
        num_layers (int): number of layers
        activation (str): type of activations for layers in between
        log_softmax (bool): whether to add a log_softmax layer before output
    """

    def __init__(
        self,
        hidden_size: int,
        num_classes: int,
        num_layers: int = 2,
        activation: str = 'relu',
        log_softmax: bool = True,
    ):
        super().__init__()
        self.layers = 0
        for _ in range(num_layers - 1):
            layer = torch.nn.Linear(hidden_size, hidden_size)
            setattr(self, f'layer{self.layers}', layer)
            setattr(self, f'layer{self.layers + 1}', getattr(torch, activation))
            self.layers += 2
        layer = torch.nn.Linear(hidden_size, num_classes)
        setattr(self, f'layer{self.layers}', layer)
        self.layers += 1
        self.log_softmax = log_softmax
        self.num_classes = num_classes

    @property
    def last_linear_layer(self):
        return getattr(self, f'layer{self.layers - 1}')

    def forward(self, hidden_states):
        output_states = hidden_states[:]
        for i in range(self.layers):
            output_states = getattr(self, f'layer{i}')(output_states)

        if self.log_softmax:
            output_states = torch.log_softmax(output_states, dim=-1)
        return output_states


class SampledMultiLayerPerceptron(MultiLayerPerceptron):
    """
    A sampled MLP that can either be used independently or put on top
    of pretrained models (such as BERT) and act as a classifier.
    Args:
        hidden_size (int): the size of each layer
        num_classes (int): number of output classes
        num_layers (int): number of layers
        activation (str): type of activations for layers in between
        log_softmax (bool): whether to add a log_softmax layer before output
    """

    def __init__(
        self,
        hidden_size: int,
        num_classes: int,
        num_layers: int = 2,
        activation: str = 'relu',
        log_softmax: bool = True,
        num_samples: Optional[int] = None,
    ):
        super().__init__(
            hidden_size=hidden_size,
            num_classes=num_classes,
            num_layers=num_layers,
            activation=activation,
            log_softmax=log_softmax,
        )

        if num_samples is None or int(num_samples) < 1:
            raise ValueError(f"For SampledMultiLayerPerceptron, num_samples must be an int > 1. Given: {num_samples}")

        self.num_samples = num_samples

    def forward(self, hidden_states, targets=None, labels=None):
        # If targets are not provided, treat as simple MLP
        if targets is None or labels is None:
            return super().forward(hidden_states)

        # Forward through first N-1 layers
        output_states = hidden_states[:]
        for layer_id in range(self.layers - 1):
            output_states = getattr(self, f'layer{layer_id}')(output_states)

        # Begin compute of sampled softmax of vocab projection
        with torch.no_grad():
            # gather true labels of both targets and labels
            target_label_vocab_ids = torch.unique(torch.cat([targets, labels], dim=1), sorted=True)

            # Remap the targets label ids to new positions of label ids (in the targets_vocab_ids)
            # This is necessary cause the RNNT loss doesnt care about the value, only the position of the ids
            # of the targets tokens. We can skip this step for noise samples cause those are only used for softmax
            # estimation, not for computing actual label.
            # From `https://stackoverflow.com/a/68969697` - bucketize algo.
            t_ids = torch.arange(target_label_vocab_ids.size(0), device='cpu')
            mapping = {k: v for k, v in zip(target_label_vocab_ids.to('cpu'), t_ids)}

            # From `https://stackoverflow.com/questions/13572448`.
            palette, key = zip(*mapping.items())

            t_device = targets.device
            key = torch.tensor(key, device=t_device)
            palette = torch.tensor(palette, device=t_device)

            # This step maps old token id to new token id in broadcasted manner.
            # For example, if original targets tokens were [2, 1, 4, 5, 4, 1]
            # But after computing the unique token set of above we get
            # targets_vocab_ids = [1, 2, 4, 5]  # note: pytorch returns sorted unique values thankfully
            # Then we get the index map of the new vocab ids as:
            # {0: 1, 1: 2, 2: 4, 3: 5}
            # Now we need to map the original targets tokens to new vocab id space
            # So we construct the inverted map as follow :
            # {1: 0, 2: 1, 4: 2, 5: 3}
            # Then remap the original targets tokens to new token ids
            # new_targets = [1, 0, 2, 3, 2, 0]
            index = torch.bucketize(targets.ravel(), palette)
            targets = key[index].reshape(targets.shape)
            targets = targets.to(t_device)

            # Same as above, but for labels
            index = torch.bucketize(labels.ravel(), palette)
            labels = key[index].reshape(labels.shape)
            labels = labels.to(t_device)

        # Get the last layer
        last_layer = self.last_linear_layer

        # Extract out partial weight tensor and bias tensor of just the V_Pos vocabulary from the full joint.
        true_weights = last_layer.weight[target_label_vocab_ids, :]
        true_bias = last_layer.bias[target_label_vocab_ids]

        # Compute the target scores (only of vocab V_Pos)
        target_scores = torch.matmul(output_states, true_weights.transpose(0, 1)) + true_bias

        # Construct acceptance criteria in vocab space, reject all tokens in Intersection(V_Pos, V_Neg)
        with torch.no_grad():
            # Instead of uniform sample, first we create arange V (ignoring blank), then randomly shuffle
            # this range of ids, then subset `n_samples` amount of vocab tokens out of the permuted tensor.
            # This is good because it guarentees that no token will ever be repeated in V_Neg;
            # which dramatically complicates loss calculation.
            # Further more, with this strategy, given a `n_samples` > V + 1; we are guarenteed to get the
            # V_Samples = V (i.e., full vocabulary will be used in such a case).
            # Useful to debug cases where you expect sampled vocab to get exact same training curve as
            # full vocab.
            sample_ids = torch.randperm(n=self.num_classes - 1, device=target_scores.device)[: self.num_samples]

            # We need to compute the intersection(V_Pos, V_Neg), then eliminate the intersection arguments
            # from inside V_Neg.

            # First, compute the pairwise commonality to find index inside `sample_ids` which match the token id
            # inside transcript_vocab_ids.
            # Note: It is important to ignore the hardcoded RNNT Blank token injected at id 0 of the transcript
            # vocab ids, otherwise the blank may occur twice, once for RNNT blank and once as negative sample,
            # doubling the gradient of the RNNT blank token.
            reject_samples = torch.where(target_label_vocab_ids[:, None] == sample_ids[None, :])

            # Let accept samples be a set of ids which is a subset of sample_ids
            # such that intersection(V_Pos, accept_samples) is a null set.
            accept_samples = sample_ids.clone()

            # In order to construct such an accept_samples tensor, first we construct a bool map
            # and fill all the indices where there is a match inside of sample_ids.
            # reject_samples is a tuple (transcript_vocab_position, sample_position) which gives a
            # many to many map between N values of transript and M values of sample_ids.
            # We dont care about transcript side matches, only the ids inside of sample_ids that matched.
            sample_mask = torch.ones_like(accept_samples, dtype=torch.bool)
            sample_mask[reject_samples[1]] = False

            # Finally, compute the subset of tokens by selecting only those sample_ids which had no matches
            accept_samples = accept_samples[sample_mask]

        # Extract out partial weight tensor and bias tensor of just the V_Neg vocabulary from the full joint.
        sample_weights = last_layer.weight[accept_samples, :]
        sample_bias = last_layer.bias[accept_samples]

        # Compute the noise joint scores (only of vocab V_Neg) to be used for softmax
        # The quality of this sample determines the quality of the softmax gradient.
        # We use naive algo broadcasted over batch, but it is more efficient than sample level computation.
        # One can increase `n_samples` for better estimation of rejection samples and its gradient.
        noise_scores = torch.matmul(output_states, sample_weights.transpose(0, 1)) + sample_bias

        # Finally, construct the sampled joint as the V_Sampled = Union(V_Pos, V_Neg)
        # Here, we simply concatenate the two tensors to construct the joint with V_Sampled vocab
        # because before we have properly asserted that Intersection(V_Pos, V_Neg) is a null set.
        output_states = torch.cat([target_scores, noise_scores], dim=-1)

        # Apply log_softmax if needed
        if self.log_softmax:
            output_states = torch.log_softmax(output_states, dim=-1)

        return output_states, targets, labels
