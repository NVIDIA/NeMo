from typing import List

import torch


class ZeroShotModule(torch.nn.Module):
    def __init__(self, hidden_size: int = 768, max_batch_size: int = 16):
        super().__init__()
        self.hidden_size = hidden_size
        self.device = 'cpu'
        self.max_batch_size = max_batch_size
        self.bio_mlp = torch.nn.Linear(self.hidden_size, 300, bias=False)
        self.bio_mlp = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_size, self.hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_size, 3),
            torch.nn.LogSoftmax(dim=-1),
        )
        self.mention_projection_mlp = torch.nn.Linear(self.hidden_size, 300, bias=False)
        self.description_projection_mlp = torch.nn.Linear(self.hidden_size, 300, bias=False)

    def forward(
        self, bio_slot_labels: torch.Tensor, hidden_states: torch.Tensor, entity_type_embeddings: torch.Tensor
    ):
        bio_slot_logits = self.bio_mlp(hidden_states)
        predicted_bio_slot = torch.argmax(bio_slot_logits, dim=-1)

        mention_hidden_states_pad = self.get_entity_embedding_from_hidden_states(bio_slot_labels, hidden_states)
        dot_product_score = self.get_entity_description_score(mention_hidden_states_pad, entity_type_embeddings)
        dot_product_score_log_softmax = torch.log_softmax(dot_product_score, dim=-1)

        predicted_mention_hidden_states_pad = self.get_entity_embedding_from_hidden_states(
            predicted_bio_slot, hidden_states
        )

        predicted_dot_product_score = self.get_entity_description_score(
            predicted_mention_hidden_states_pad, entity_type_embeddings
        )
        predicted_dot_product_score_log_softmax = torch.log_softmax(predicted_dot_product_score, dim=-1)
        return bio_slot_logits, dot_product_score_log_softmax, predicted_dot_product_score_log_softmax

    def get_entity_embedding_from_hidden_states(self, bio_slot_labels, hidden_states):
        batch_size, max_len, hidden_size = hidden_states.shape

        bio_labels_flat = bio_slot_labels.reshape(-1)
        hidden_shapes_flat = hidden_states.reshape((-1, hidden_size))

        # Remove all zeros from the input since we don't use it
        hidden_shapes_flat = hidden_shapes_flat[bio_labels_flat != 0]
        bio_labels_flat = bio_labels_flat[bio_labels_flat != 0]

        # Find indices of 1's since that's where an entity mention begins
        idx = (bio_labels_flat == 1).nonzero().reshape(-1)

        # Find lengths of each BIO entity mention
        mention_lengths = idx[1:] - idx[:-1]

        sum1 = torch.tensor([0], device=self.device)
        if mention_lengths.shape[0] > 0:
            sum1 = torch.sum(mention_lengths, dim=0, keepdim=True)

        # Find the "leftover" length for the last element in the mention_lengths array
        remainder = (bio_labels_flat.shape[0] - sum1).to(self.device)

        # Convert the new mention lengths array to a python list
        mention_lengths_list: List[int] = torch.cat((mention_lengths, remainder)).tolist()

        means_list = []

        # Calculate the final embeddings by first splitting the hidden shapes array and then averaging
        # the embeddings
        for y2_ in hidden_shapes_flat.split(mention_lengths_list):
            mean = torch.zeros((max_len, hidden_size))
            if y2_.shape[0] > 0:
                mean = torch.mean(y2_, dim=0, keepdim=True)
            means_list.append(mean)

        means = torch.cat(means_list)

        output = []
        count_ones: List[int] = torch.sum(bio_slot_labels == 1, dim=1, dtype=torch.int64).tolist()

        # Finally create the output array by padding
        for count in count_ones:
            temp = means[:count]
            means = means[count:]
            output.append(torch.nn.functional.pad(temp, (0, 0, 0, max_len - count)))

        return torch.stack(output).to(self.device)

    def get_entity_description_score(self, mention_hidden_states_pad, entity_type_embeddings):
        projected_description_embedding = self.description_projection_mlp(entity_type_embeddings)
        projected_mention_embedding = self.mention_projection_mlp(mention_hidden_states_pad)
        dot_product_score = torch.matmul(projected_mention_embedding, torch.t(projected_description_embedding))

        return dot_product_score

    def to(self, device):
        self.device = device
        return super().to(device)
