import pytorch_lightning as pl
import torch


class OnnxModule(torch.nn.Module):
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
        mention_hidden_states = []
        orig_batch_size, max_token_len, hidden_states_dim = hidden_states.shape

        # missing = self.max_batch_size - batch_size
        # hidden_states_new = torch.cat((hidden_states, torch.randn(missing, max_token_len, hidden_states_dim,
        #                                                       device=self.device)))
        # bio_slot_labels_new = torch.cat((bio_slot_labels, torch.zeros(missing, max_token_len, device=self.device, dtype=bio_slot_labels.dtype)))

        hidden_states_new = hidden_states[bio_slot_labels.any(dim=1)].to(self.device)
        bio_slot_labels_new = bio_slot_labels[bio_slot_labels.any(dim=1)].to(self.device)

        batch_size, _, _ = hidden_states_new.shape

        for i in range(batch_size):
            one_hidden_states = hidden_states_new[i]
            one_bio_slot_labels = bio_slot_labels_new[i]

            one_hidden_states = one_hidden_states[one_bio_slot_labels != 0]
            one_bio_slot_labels = one_bio_slot_labels[one_bio_slot_labels != 0]

            idx = (one_bio_slot_labels == 1).nonzero().squeeze(1).to('cpu')
            y_split = torch.tensor_split(one_hidden_states, idx)

            mention_states = []
            for j in range(len(y_split)):
                y = y_split[j]
                temp = torch.mean(y, dim=0, keepdim=True)
                mention_states.append(temp)

            mention_states = torch.cat(mention_states, 0)
            mention_states = mention_states[~torch.any(mention_states.isnan(), dim=1)]
            zero_fill = torch.zeros((max_token_len - mention_states.size()[0], hidden_states_dim), device=self.device)
            mention_states = torch.cat((mention_states, zero_fill), 0)
            mention_hidden_states.append(mention_states)

        if mention_hidden_states:
            output = torch.stack(mention_hidden_states)
            zero_fill_batch = torch.zeros(
                (orig_batch_size - output.shape[0], max_token_len, hidden_states_dim), device=self.device
            )
            output = torch.cat((output, zero_fill_batch))
        else:
            output = torch.zeros((orig_batch_size, max_token_len, hidden_states_dim))

        return output.to(self.device)

    def get_entity_description_score(self, mention_hidden_states_pad, entity_type_embeddings):
        projected_description_embedding = self.description_projection_mlp(entity_type_embeddings)
        projected_mention_embedding = self.mention_projection_mlp(mention_hidden_states_pad)
        dot_product_score = torch.matmul(projected_mention_embedding, torch.t(projected_description_embedding))

        return dot_product_score

    def to(self, device):
        self.device = device
        return super().to(device)
