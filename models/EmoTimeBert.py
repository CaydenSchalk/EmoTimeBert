import os
import torch
import torch.nn as nn
from transformers import AutoModel
from models.TemporalTransformer import TemporalTransformer

class EmotionalTimeBert(nn.Module):
    def __init__(self, encoder_name, num_labels, max_time = 8, max_speakers=2):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_name)
        hidden = self.encoder.config.hidden_size

        self.head_emotions = nn.Linear(hidden, num_labels)
        self.time_embed = nn.Embedding(max_time + 1, hidden)
        self.speakers_embed = nn.Embedding(max_speakers + 1, hidden)
        self.temporal_transformer = TemporalTransformer(hidden, 2, 8, 0.1)# num_labels, hidden, False)

        # pause training bert
        # for p in self.encoder.parameters():
        #     p.requires_grad = False

        # for layer in self.encoder.encoder.layer[-2:]:
        #     for p in layer.parameters():
        #         p.requires_grad = True

        for p in self.encoder.parameters():
            p.requires_grad = False

        for layer in self.encoder.encoder.layer[-4:]:
            for p in layer.parameters():
                p.requires_grad = True



    def forward(self, input_ids, attention_mask, timestamps=None, speakers=None, labels=None, utterance_mask=None):
        # bert_output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)

        flat_mask = utterance_mask.view(-1).bool()
        bert_output = self.encoder(
            input_ids=input_ids[flat_mask],
            attention_mask=attention_mask[flat_mask]
        )

        h = bert_output.last_hidden_state[:, 0, :]   # (B*T, H)
        # B, T = timestamps.shape
        # H = h.size(-1)

        B, T = timestamps.shape
        H = h.size(-1)

        h_all = torch.zeros(
            (B * T, H),
            device=h.device,
            dtype=h.dtype
        )

        h_all[flat_mask] = h

        speakers = speakers + 1

        h_t = h_all.view(B, T, H)
        time_vec = self.time_embed(timestamps)
        speakers_vec = self.speakers_embed(speakers)
        Z = h_t + time_vec + speakers_vec
        padding_mask = (labels == -1)# (timestamps == 0) # & (speakers == 0)
        U = self.temporal_transformer(Z, padding_mask)

        logits = self.head_emotions(U)
        return logits