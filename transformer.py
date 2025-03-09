import torch
import torch.nn as nn


class InputEmbeddings(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model

    def forward(self, x):
        return x * torch.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000, temperature=10000):
        super().__init__()
        """
    d_model => 512
    """
        self.d_model = d_model
        """
    I must do the unsqueeze on position couse
    pe and position must have the same shape

    I have to ADD the pe (positional encoding)
    to the input embeddings
    """
        pe = torch.zeros(max_len, d_model)  # shape: (max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(
            1
        )  # shape: (seq_len, 1)
        div_term = temperature ** (torch.arange(0, d_model, 2).float() / d_model)

        pe[:, 0::2] = torch.sin(position / div_term)
        pe[:, 1::2] = torch.cos(position / div_term)

        pe = pe.unsqueeze(0)  # shape: (1, seq_len, d_model)

        # serve per fare in modo che il modello non modifichi i pesi
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.shape[1], :].requires_grad_(False)
        return x


class LayerNorm(
    nn.Module
):  # here in some implementations, we can inject the positional encoding
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.d_model = d_model
        self.alpha = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias


class FeedForwardBlock(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.fc2(self.dropout(torch.relu(self.fc1(x))))


class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model, h, dropout):
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0, "d_model must be divisible by h"
        self.d_k = d_model // h
        self.dropout = nn.Dropout(dropout)

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

    @staticmethod
    def attention(q, k, v, mask, dropout):
        d_k = q.shape[-1]
        attention_scores = (q @ k.T) / torch.sqrt(d_k)

        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)

        attention_scores = torch.softmax(attention_scores, dim=-1)

        if dropout is not None:  # tipically dropout is 0.2
            attention_scores = dropout(attention_scores)

        return (attention_scores @ v), attention_scores

    def forward(self, q, k, v, mask):
        q = self.w_q(q)
        k = self.w_k(k)
        v = self.w_v(v)

        q = q.view(q.shape[0], -1, self.h, self.d_k).transpose(1, 2)
        k = k.view(k.shape[0], -1, self.h, self.d_k).transpose(1, 2)
        v = v.view(v.shape[0], -1, self.h, self.d_k).transpose(1, 2)

        x, self.attention_scores = MultiHeadAttentionBlock.attention(
            q, k, v, mask, self.dropout
        )

        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.d_model)

        return self.w_o(x)


class ResidualConnection(nn.Module):
    def __init__(self, d_model, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNorm(d_model)

    def forward(self, x, sublayer):
        return x + self.dropout(self.norm(sublayer(x)))


class EncoderBlock(nn.Module):
    def __init__(self, d_model, self_attention_block, feed_fordward_block, dropout):
        super().__init__()
        self.d_model = d_model
        self.self_attention_block = self_attention_block
        self.feed_fordward_block = feed_fordward_block
        self.residual_connections = nn.ModuleList(
            [ResidualConnection(d_model, dropout) for _ in range(2)]
        )

    def forward(self, x, src_mask=None):
        x = self.residual_connections[0](
            x, lambda x: self.self_attention_block(x, x, x, src_mask)
        )
        return self.residual_connections[1](x, self.feed_fordward_block)


class Encoder(nn.Module):
    def __init__(self, d_model, layers):
        super().__init__()
        self.layers = layers
        self.d_model = d_model
        self.norm = LayerNorm(d_model)

    def forward(self, x, src_mask=None):
        for layer in self.layers:
            x = layer(x, src_mask)
        return self.norm(x)


class DecoderBlock(nn.Module):
    def __init__(
        self,
        d_model,
        self_attention_block,
        cross_attention_block,
        feed_forward_block,
        dropout,
    ):
        super().__init__()
        self.d_model = d_model
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.cross_attention_block = cross_attention_block
        self.residual_connections = nn.ModuleList(
            [ResidualConnection(d_model, dropout) for _ in range(3)]
        )

    def forward(self, x, encoder_output, src_mask=None):
        x = self.residual_connections[0](
            x, lambda x: self.self_attention_block(x, x, x)
        )
        x = self.residual_connections[1](
            x,
            lambda x: self.cross_attention_block(
                x, encoder_output, encoder_output, src_mask
            ),
        )
        return self.residual_connections[2](x, self.feed_forward_block)


class Decoder(nn.Module):
    def __init__(self, d_model, layers):
        super().__init__()
        self.layers = layers
        self.norm = LayerNorm(d_model)

    def forward(self, x, encoder_output, src_mask=None):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask)
        return self.norm(x)


class Heads(nn.Module):
    def __init__(self, d_model, num_classes):
        super().__init__()
        self.class_embeddings = nn.Linear(d_model, num_classes + 1)
        self.bbox_embeddings = MLP(d_model, d_model, 4, 3)

    def forward(self, x):
        return {
            "pred_logits": self.class_embeddings(x),
            "pred_boxes": self.bbox_embeddings(x).sigmoid(),
        }


class MLP(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(in_channels, hidden_channels))

        for _ in range(num_layers - 2):
            self.layers.append(nn.Linear(hidden_channels, hidden_channels))

        self.layers.append(nn.Linear(hidden_channels, out_channels))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = nn.functional.relu(layer(x)) if i < len(self.layers) - 1 else layer(x)

        return x


class DETR(nn.Module):
    def __init__(self, backbone, transformer, num_classes, num_queries, d_model=256):
        super().__init__()
        self.backbone = backbone  # ResNet backbone already implemented
        self.transformer = transformer

        self.input_proj = nn.Conv2d(backbone.out_channels, d_model, kernel_size=1)

        self.position_encoding = PositionalEncoding(d_model)

        self.query_embed = nn.Embedding(num_queries, d_model)

        self.class_embed = nn.Linear(d_model, num_classes + 1)  # +1 for no-object class
        self.bbox_embed = MLP(d_model, d_model, 4, 3)  # 4 for (x, y, w, h)

    def forward(self, x):
        features = self.backbone(x)

        h = self.input_proj(features)

        bs, c, h_size, w_size = h.shape
        h = h.flatten(2).permute(0, 2, 1)  # (batch_size, h*w, d_model)

        pos = self.position_encoding(h)

        query_embed = self.query_embed.weight.unsqueeze(0).repeat(bs, 1, 1)

        h = self.transformer(h, query_embed, pos)

        outputs_class = self.class_embed(h)
        outputs_coord = self.bbox_embed(h).sigmoid()  # normalize to [0, 1]

        return {"pred_logits": outputs_class, "pred_boxes": outputs_coord}


class DETRTransformer(nn.Module):
    def __init__(
        self,
        d_model=256,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
    ):
        super().__init__()

        encoder_layers = []
        for _ in range(num_encoder_layers):
            encoder_self_attn = MultiHeadAttentionBlock(d_model, nhead, dropout)
            feed_forward = FeedForwardBlock(d_model, dim_feedforward, dropout)
            encoder_layers.append(
                EncoderBlock(d_model, encoder_self_attn, feed_forward, dropout)
            )

        self.encoder = Encoder(d_model, encoder_layers)

        decoder_layers = []
        for _ in range(num_decoder_layers):
            decoder_self_attn = MultiHeadAttentionBlock(d_model, nhead, dropout)
            decoder_cross_attn = MultiHeadAttentionBlock(d_model, nhead, dropout)
            feed_forward = FeedForwardBlock(d_model, dim_feedforward, dropout)
            decoder_layers.append(
                DecoderBlock(
                    d_model,
                    decoder_self_attn,
                    decoder_cross_attn,
                    feed_forward,
                    dropout,
                )
            )

        self.decoder = Decoder(d_model, decoder_layers)

    def forward(self, src, query_embed, pos_embed):
        src = src + pos_embed

        # Encode
        memory = self.encoder(src)

        # Initialize decoder input with object queries
        tgt = torch.zeros_like(query_embed)

        # Decode
        hs = self.decoder(tgt, memory)

        return hs


def build_detr(backbone, num_classes=91, num_queries=100):
    """
    Build a complete DETR model given a backbone (ResNet)
    """
    transformer = DETRTransformer(
        d_model=256,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
    )

    model = DETR(
        backbone=backbone,
        transformer=transformer,
        num_classes=num_classes,
        num_queries=num_queries,
        d_model=256,
    )

    # Initialize parameters
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return model
