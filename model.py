import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import D_MODEL, N_ENCODER_LAYERS, N_DECODER_LAYERS, N_HEADS, FF_DIM, DROPOUT, MAX_TEXT_LEN, MAX_SPEC_LEN, N_FREQS, N_CHANNELS


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:x.size(1)]


class TextEncoder(nn.Module):
    def __init__(self, vocab_size, d_model=D_MODEL, n_layers=N_ENCODER_LAYERS,
                 n_heads=N_HEADS, ff_dim=FF_DIM, dropout=DROPOUT, max_len=MAX_TEXT_LEN):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len)
        self.d_model = d_model
        layer = nn.TransformerEncoderLayer(d_model, n_heads, ff_dim, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers)

    def forward(self, tokens, mask=None):
        x = self.embed(tokens) * math.sqrt(self.d_model)
        x = self.pos_enc(x)
        return self.encoder(x, src_key_padding_mask=mask)


class SpecDecoder(nn.Module):
    def __init__(self, d_model=D_MODEL, n_layers=N_DECODER_LAYERS,
                 n_heads=N_HEADS, ff_dim=FF_DIM, dropout=DROPOUT, max_len=MAX_SPEC_LEN):
        super().__init__()
        self.frame_size = N_FREQS * N_CHANNELS
        self.prenet = nn.Sequential(
            nn.Linear(self.frame_size, d_model),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(0.5),
        )
        self.pos_enc = PositionalEncoding(d_model, max_len)
        self.start_frame = nn.Parameter(torch.randn(1, 1, self.frame_size) * 0.01)
        layer = nn.TransformerDecoderLayer(d_model, n_heads, ff_dim, dropout, batch_first=True)
        self.decoder = nn.TransformerDecoder(layer, n_layers)
        self.output_proj = nn.Linear(d_model, self.frame_size)
        self.eos_head = nn.Linear(d_model, 2)

    def forward(self, memory, spec_frames, tgt_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        x = self.prenet(spec_frames)
        x = self.pos_enc(x)
        x = self.decoder(x, memory, tgt_mask=tgt_mask,
                         tgt_key_padding_mask=tgt_key_padding_mask,
                         memory_key_padding_mask=memory_key_padding_mask)
        frame_pred = self.output_proj(x)
        eos_logits = self.eos_head(x)
        return frame_pred, eos_logits


def _causal_mask(sz, device):
    return torch.triu(torch.full((sz, sz), float("-inf"), device=device), diagonal=1)


class DirectTTS(nn.Module):
    def __init__(self, vocab_size, d_model=D_MODEL,
                 n_enc_layers=N_ENCODER_LAYERS, n_dec_layers=N_DECODER_LAYERS,
                 n_heads=N_HEADS, ff_dim=FF_DIM, dropout=DROPOUT,
                 max_text_len=MAX_TEXT_LEN, max_spec_len=MAX_SPEC_LEN):
        super().__init__()
        self.encoder = TextEncoder(vocab_size, d_model, n_enc_layers, n_heads, ff_dim, dropout, max_text_len)
        self.decoder = SpecDecoder(d_model, n_dec_layers, n_heads, ff_dim, dropout, max_spec_len)

    def forward(self, tokens, spec_frames, token_mask=None, frame_mask=None):
        memory = self.encoder(tokens, mask=token_mask)
        T = spec_frames.size(1)
        tgt_mask = _causal_mask(T, spec_frames.device)
        frame_pred, eos_logits = self.decoder(memory, spec_frames, tgt_mask=tgt_mask,
                                              tgt_key_padding_mask=frame_mask,
                                              memory_key_padding_mask=token_mask)
        return frame_pred, eos_logits

    @torch.no_grad()
    def generate(self, tokens, eos_threshold=0.5, max_frames=MAX_SPEC_LEN):
        self.eval()
        B = tokens.size(0)
        memory = self.encoder(tokens)

        frame = self.decoder.start_frame.expand(B, -1, -1)

        for _ in range(max_frames):
            T = frame.size(1)
            tgt_mask = _causal_mask(T, tokens.device)
            frame_pred, eos_logits = self.decoder(memory, frame, tgt_mask=tgt_mask)
            next_frame = frame_pred[:, -1:, :]
            frame = torch.cat([frame, next_frame], dim=1)

            eos_prob = torch.softmax(eos_logits[:, -1, :], dim=-1)
            if eos_prob[0, 1] > eos_threshold:
                break

        spec = frame[:, 1:, :]
        return spec
