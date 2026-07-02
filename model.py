import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import (
    D_MODEL, N_ENCODER_LAYERS, N_DECODER_LAYERS, N_HEADS, FF_DIM, DROPOUT,
    MAX_TEXT_LEN, MAX_SPEC_LEN, N_FREQS, N_CHANNELS,
    DUR_HIDDEN, DUR_KERNEL, DUR_DROPOUT, MIN_DURATION,
)
from mas import monotonic_alignment_search, extract_durations


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
    """AR decoder kept for the alignment pass — returns hidden states for MAS."""
    def __init__(self, d_model=D_MODEL, n_layers=N_DECODER_LAYERS,
                 n_heads=N_HEADS, ff_dim=FF_DIM, dropout=DROPOUT, max_len=MAX_SPEC_LEN):
        super().__init__()
        self.frame_size = N_FREQS * N_CHANNELS
        self.prenet = nn.Sequential(
            nn.Linear(self.frame_size, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
        )
        self.pos_enc = PositionalEncoding(d_model, max_len)
        self.start_frame = nn.Parameter(torch.randn(1, 1, self.frame_size) * 0.01)
        self.mask_token = nn.Parameter(torch.randn(1, 1, self.frame_size) * 0.01)
        layer = nn.TransformerDecoderLayer(d_model, n_heads, ff_dim, dropout, batch_first=True)
        self.decoder = nn.TransformerDecoder(layer, n_layers)
        self.output_proj = nn.Linear(d_model, self.frame_size)
        self.eos_head = nn.Linear(d_model, 2)
        self.confidence_head = nn.Linear(d_model, 1)

    def forward(self, memory, spec_frames, tgt_mask=None, mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        x = self.prenet(spec_frames)
        x = self.pos_enc(x)
        x = self.decoder(x, memory, tgt_mask=tgt_mask,
                         tgt_key_padding_mask=tgt_key_padding_mask,
                         memory_key_padding_mask=memory_key_padding_mask)
        frame_pred = self.output_proj(x)
        eos_logits = self.eos_head(x)
        confidence = torch.sigmoid(self.confidence_head(x))
        return frame_pred, eos_logits, confidence, x


def _causal_mask(sz, device):
    return torch.triu(torch.full((sz, sz), float("-inf"), device=device), diagonal=1)


class AlignmentScorer(nn.Module):
    """Bilinear scorer that computes alignment logits for MAS."""
    def __init__(self, d_model):
        super().__init__()
        self.proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, decoder_states, encoder_outputs):
        enc = self.proj(encoder_outputs)
        scores = torch.bmm(decoder_states, enc.transpose(1, 2))
        return scores / math.sqrt(decoder_states.size(-1))


class DurationPredictor(nn.Module):
    """Small conv-net predicting per-token frame durations (FastSpeech 2 style)."""
    def __init__(self, d_model, hidden=DUR_HIDDEN, kernel=DUR_KERNEL, dropout=DUR_DROPOUT):
        super().__init__()
        self.conv1 = nn.Conv1d(d_model, hidden, kernel, padding=kernel // 2)
        self.conv2 = nn.Conv1d(hidden, hidden, kernel, padding=kernel // 2)
        self.proj = nn.Linear(hidden, 1)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(hidden)

    def forward(self, x, x_mask=None):
        x = x.transpose(1, 2)
        x = self.dropout(F.relu(self.conv1(x)))
        x = x.transpose(1, 2)
        x = self.norm1(x)
        x = x.transpose(1, 2)
        x = self.dropout(F.relu(self.conv2(x)))
        x = x.transpose(1, 2)
        x = self.norm2(x)
        log_dur = self.proj(x).squeeze(-1)
        if x_mask is not None:
            log_dur = log_dur.masked_fill(x_mask, 0.0)
        return log_dur.exp()


class DurationDecoder(nn.Module):
    """Autoregressive decoder — predicts frames from expanded phone sequence.
    Uses TransformerEncoder with causal self-attention.
    Memory (expanded phone sequence) is injected via element-wise addition,
    avoiding the O(n²) memory cost of diagonal cross-attention."""
    def __init__(self, d_model=D_MODEL, n_layers=N_DECODER_LAYERS,
                 n_heads=N_HEADS, ff_dim=FF_DIM, dropout=DROPOUT, max_len=MAX_SPEC_LEN):
        super().__init__()
        self.frame_size = N_FREQS * N_CHANNELS
        self.prenet = nn.Sequential(
            nn.Linear(self.frame_size, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
        )
        self.pos_enc = PositionalEncoding(d_model, max_len)
        self.start_frame = nn.Parameter(torch.randn(1, 1, self.frame_size) * 0.01)
        self.memory_proj = nn.Linear(d_model, d_model)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, ff_dim, dropout, batch_first=True)
        self.layers = nn.TransformerEncoder(layer, n_layers)
        self.output_proj = nn.Linear(d_model, self.frame_size)
        self.eos_head = nn.Linear(d_model, 2)

    def forward(self, memory, tgt_frames, tgt_mask=None, memory_mask=None,
                memory_key_padding_mask=None, tgt_key_padding_mask=None):
        x = self.prenet(tgt_frames)
        x = self.pos_enc(x)
        mem = self.memory_proj(memory)
        x = x + mem
        x = self.layers(x, mask=tgt_mask, src_key_padding_mask=tgt_key_padding_mask)
        frame_pred = self.output_proj(x)
        eos_logits = self.eos_head(x)
        return frame_pred, eos_logits


class DirectTTS(nn.Module):
    def __init__(self, vocab_size, d_model=D_MODEL,
                 n_enc_layers=N_ENCODER_LAYERS, n_dec_layers=N_DECODER_LAYERS,
                 n_heads=N_HEADS, ff_dim=FF_DIM, dropout=DROPOUT,
                 max_text_len=MAX_TEXT_LEN, max_spec_len=MAX_SPEC_LEN):
        super().__init__()
        self.encoder = TextEncoder(vocab_size, d_model, n_enc_layers, n_heads, ff_dim, dropout, max_text_len)
        self.decoder = SpecDecoder(d_model, n_dec_layers, n_heads, ff_dim, dropout, max_spec_len)
        self.alignment_scorer = AlignmentScorer(d_model)
        self.duration_predictor = DurationPredictor(d_model)
        self.duration_decoder = DurationDecoder(d_model, n_dec_layers, n_heads, ff_dim, dropout, max_spec_len)
        self.max_spec_len = max_spec_len

    def forward(self, tokens, spec_frames, token_mask=None, frame_mask=None,
                spec_lens=None):
        B = tokens.size(0)
        device = tokens.device
        T_spec = spec_frames.size(1)

        # ── Encode (in graph — trains encoder for both dur_pred and DD losses) ──
        memory = self.encoder(tokens, mask=token_mask)

        # ── Pass 1: AR alignment pass (detached — no gradients through MAS) ──
        with torch.no_grad():
            start = self.decoder.start_frame.expand(B, -1, -1)
            dec_in = torch.cat([start, spec_frames[:, :-1]], dim=1)
            tgt_mask = _causal_mask(T_spec, device)
            ar_preds, ar_eos, _, hidden = self.decoder(
                memory, dec_in, tgt_mask=tgt_mask,
                tgt_key_padding_mask=frame_mask,
                memory_key_padding_mask=token_mask)

            scores = self.alignment_scorer(hidden, memory)
            alignment = monotonic_alignment_search(
                scores, token_mask=token_mask, frame_mask=frame_mask)
            durations = extract_durations(alignment, token_mask, MIN_DURATION)

        # ── Expand (in graph — DD loss trains encoder) ──
        expanded, expand_mask, _ = self._expand(memory, durations, token_mask)

        # ── Duration predictor (in graph) ──
        dur_pred = self.duration_predictor(memory, token_mask)

        # ── Pass 2: DurationDecoder (in graph, causal self-attn, element-wise memory) ──
        T_expanded = expanded.size(1)
        T_common = min(T_spec, T_expanded)
        expanded = expanded[:, :T_common]
        expand_mask = expand_mask[:, :T_common]

        dd_start = self.duration_decoder.start_frame.expand(B, -1, -1)
        dd_tgt = torch.cat([dd_start, spec_frames[:, :T_common - 1]], dim=1)
        dd_tgt_mask = _causal_mask(T_common, device)

        dd_preds, dd_eos = self.duration_decoder(
            expanded, dd_tgt, tgt_mask=dd_tgt_mask,
            tgt_key_padding_mask=expand_mask)

        return {
            "ar_preds": ar_preds,
            "ar_eos": ar_eos,
            "dd_preds": dd_preds,
            "dd_eos": dd_eos,
            "durations": durations,
            "dur_pred": dur_pred,
            "alignment": alignment,
        }

    def _expand(self, memory, durations, token_mask):
        B, T_text, D = memory.shape
        device = memory.device

        if token_mask is not None:
            valid_tokens = ~token_mask
        else:
            valid_tokens = torch.ones(B, T_text, dtype=torch.bool, device=device)

        dur_valid = durations.masked_fill(~valid_tokens, 0)
        total_lens = dur_valid.sum(dim=1).long()
        max_len = min(total_lens.max().item(), self.max_spec_len)

        expand_mask = torch.ones(B, max_len, dtype=torch.bool, device=device)
        gather_idx = torch.zeros(B, max_len, dtype=torch.long, device=device)

        for b in range(B):
            pos = 0
            for j in range(T_text):
                if not valid_tokens[b, j]:
                    continue
                dur = int(durations[b, j].item())
                end = min(pos + dur, max_len)
                gather_idx[b, pos:end] = j
                pos += dur
                if pos >= max_len:
                    break
            expand_mask[b, :pos] = False

        expanded = torch.gather(
            memory.unsqueeze(1).expand(-1, max_len, -1, -1),
            2,
            gather_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, D)
        ).squeeze(2)

        return expanded, expand_mask, gather_idx

    @torch.no_grad()
    def generate(self, tokens, eos_threshold=0.5, max_frames=None):
        self.eval()
        with torch.no_grad():
            B = tokens.size(0)
            device = tokens.device
            max_frames = max_frames or self.max_spec_len

            memory = self.encoder(tokens)

            dur_pred = self.duration_predictor(memory)
            durations = dur_pred.round().long().clamp(min=MIN_DURATION)

            expanded, _, _ = self._expand(memory, durations, token_mask=None)
            T_expanded = expanded.size(1)
            if T_expanded > max_frames:
                expanded = expanded[:, :max_frames]
                T_expanded = max_frames

            if T_expanded == 0:
                return torch.zeros(B, 0, self.duration_decoder.frame_size, device=device)

            generated = [self.duration_decoder.start_frame.expand(B, -1, -1)]

            for step in range(T_expanded):
                tgt = torch.cat(generated, dim=1)
                tgt_len = tgt.size(1)
                tgt_mask = _causal_mask(tgt_len, device)

                frame_pred, eos_logits = self.duration_decoder(
                    expanded[:, :tgt_len], tgt, tgt_mask=tgt_mask)

                next_frame = frame_pred[:, -1:, :]
                generated.append(next_frame)

                eos_prob = torch.softmax(eos_logits, dim=-1)
                if (eos_prob[:, -1, 1] > eos_threshold).any():
                    break

            result = torch.cat(generated[1:], dim=1)
        return result
