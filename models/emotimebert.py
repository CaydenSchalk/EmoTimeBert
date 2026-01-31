import torch
import torch.nn as nn
from transformers import AutoModel
from models.temporal_transformer import TemporalTransformer
from utils.param_objects import EmpatheticDialogueParams, MELDParams
from enum import Enum

class DatasetMode(Enum):
    ED = 1
    MELD = 2
    BOTH = 3

class EmotionalTimeBert(nn.Module):
    def __init__(self, encoder_name, meld_params : MELDParams = None, empathetic_params : EmpatheticDialogueParams=None): # , meld_params : MELDParams
        super().__init__()

        self.dataset_mode = DatasetMode.ED

        if meld_params is not None and empathetic_params is not None:
            self.dataset_mode = DatasetMode.BOTH
        elif meld_params is not None:
            self.dataset_mode = DatasetMode.MELD

        self.encoder = AutoModel.from_pretrained(encoder_name)
        hidden = self.encoder.config.hidden_size

        if self.dataset_mode == DatasetMode.BOTH:
            self.head_meld = nn.Linear(hidden, meld_params.num_labels)
            self.head_empath = nn.Linear(hidden, empathetic_params.num_labels)
            # add 1 because we're using 0 as padding, so 0 could appear
            self.time_embed = nn.Embedding(max(empathetic_params.max_time, meld_params.max_time) + 1,
                                           hidden, padding_idx=0)
            self.speakers_embed = nn.Embedding(max(empathetic_params.max_speakers, meld_params.max_speakers) + 1,
                                               hidden, padding_idx=0)
        elif self.dataset_mode == DatasetMode.MELD:
            self.head_meld = nn.Linear(hidden, meld_params.num_labels)
            self.speakers_embed = nn.Embedding(meld_params.max_speakers + 1, hidden, padding_idx=0)
            self.time_proj = nn.Linear(1, hidden)
        else:
            self.head_empath = nn.Linear(hidden, empathetic_params.num_labels)
            self.time_embed = nn.Embedding(empathetic_params.max_time, hidden, padding_idx=0)
            self.speakers_embed = nn.Embedding(empathetic_params.max_speakers, hidden, padding_idx=0)


        self.temporal_transformer = TemporalTransformer(hidden, 2, 8, 0.1)# 2, 8, 0.1)# num_labels, hidden, False)
        self.state_gru = nn.GRU(
            input_size=hidden,
            hidden_size=hidden,
            batch_first=True
        )

        self.use_gru = False
        # self.alpha = nn.Parameter(torch.tensor(0.1))


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



    def forward(self, input_ids, attention_mask, timestamps=None, speakers=None, labels=None, utterance_mask=None, task="ED"):
        # bert_output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        if self.dataset_mode == DatasetMode.BOTH:
            # handle swapping between both datasets here
            pass
        elif self.dataset_mode == DatasetMode.MELD:
            flat_mask = utterance_mask.view(-1).bool()
            bert_output = self.encoder(
                input_ids=input_ids[flat_mask],
                attention_mask=attention_mask[flat_mask]
            )

            h = bert_output.last_hidden_state[:, 0, :]  # (B*T, H)
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

            # all
            h_t = h_all.view(B, T, H)
            # normalize to seconds instead of just ms, got an overflow issue without it
            # timestamps = timestamps.float() / 1000.0
            # timestamps = timestamps / 60.0
            time_vec = self.time_proj(timestamps.unsqueeze(-1))
            # time_vec = self.time_proj(timestamps.float().unsqueeze(-1))
            speakers_vec = self.speakers_embed(speakers)
            Z = h_t + time_vec + speakers_vec
            padding_mask = (labels == -1)  # (timestamps == 0) # & (speakers == 0)

            U = self.temporal_transformer(Z, padding_mask)

            H_state, _ = self.state_gru(U)

            alpha = 0.1  # torch.clamp(self.alpha, 0.0, 1.0)
            if self.use_gru:
                U_residual = U + alpha * H_state
            else:
                U_residual = U

            logits = self.head_meld(U_residual)
        else:
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
             # all
            h_t = h_all.view(B, T, H)
            time_vec = self.time_embed(timestamps)
            speakers_vec = self.speakers_embed(speakers)
            Z = h_t + time_vec + speakers_vec
            padding_mask = (labels == -1)# (timestamps == 0) # & (speakers == 0)

            U = self.temporal_transformer(Z, padding_mask)

            H_state, _ = self.state_gru(U)

            alpha = 0.1 # torch.clamp(self.alpha, 0.0, 1.0)
            U_residual = U + alpha * H_state

            logits = self.head_empath(U_residual)



        # U = self.temporal_transformer(Z, padding_mask)
        #
        # logits = self.head_empath(U)
        return logits