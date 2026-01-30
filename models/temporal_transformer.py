import torch.nn as nn

class TemporalTransformer(nn.Module):
    def __init__(self, hidden_size, num_layers, num_heads, dropout):
        super().__init__()

        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=num_layers
        )

    def forward(self, x, padding_mask):
        assert padding_mask.shape[:2] == x.shape[:2]

        return self.encoder(x, src_key_padding_mask=padding_mask)