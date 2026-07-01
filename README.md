# ⤳ DirecTTS

Direct phase-spectrogram neural text-to-speech with duration-aligned encoder-decoder.

Maps text to raw phase spectrogram frames (768×2 = 1536-dim per frame) via a FastSpeech 2-style architecture: text encoder → duration predictor → frame expansion → causal DurationDecoder. Audio is reconstructed via the inverse phase transform — no vocoder, no intermediate representations.

## ⤳ Architecture

```
Text → IPA phonemes → TextEncoder(2-layer transformer) → memory
                                                           ├──→ [training only] AlignmentScorer + MAS → durations from AR decoder (detached)
                                                           ├──→ DurationPredictor(2×Conv1D) → predicted durations
                                                           └──→ Expand by duration → DurationDecoder(4-layer causal TransformerEncoder + element-wise memory) → frames
                                                                                                                       ↓
                                                                                                                   eos_head → stop probability
```

- **TextEncoder**: 2-layer transformer encoder (D_MODEL=256, 8 heads)
- **AlignmentScorer**: bilinear scorer projecting encoder outputs; alignment logits fed into Monotonic Alignment Search (MAS, Viterbi DP) to extract per-token frame durations from the AR decoder's hidden states
- **SpecDecoder** (AR, training-only): 4-layer transformer decoder with Tacotron 2-style PreNet, used only for the detached alignment pass (no gradients through MAS)
- **DurationPredictor**: 2×Conv1D with ReLU, LayerNorm, dropout → predicts per-token log-durations, trained via MSE against MAS-extracted durations
- **DurationDecoder**: 4-layer TransformerEncoder with **causal self-attention mask** — encoder memory is injected via element-wise addition (not cross-attention), avoiding O(n²) diagonal attention cost; serves as the main decoder during both training and inference
- **EOS heads**: binary classifiers on both SpecDecoder and DurationDecoder outputs to predict the final frame, enabling variable-length generation
- **Confidence head**: auxiliary sigmoid head on SpecDecoder
- **Positional encoding**: sinusoidal, applied to both encoder and decoder inputs

## ⤳ Dataset

Slovak speech from [SlovakSpeech](https://huggingface.co/datasets/neurlang/slovakspeech_female_dataset) corpus, 48 kHz mono WAV files listed in `dataset/dataset.tsv` in `path<TAB>text<TAB>text` format. Training supports `--limit N` to restrict sample count and `--finetune checkpoint.pt` to continue from a prior run.

Text is phonemized to IPA via [Pygoruut](https://pypi.org/project/pygoruut/) before tokenization. The tokenizer builds a character-level vocabulary from all unique IPA characters across the dataset.

A synthetic dataset generator (`make_synthetic_dataset.py`) is also available for debugging — it renders letter sequences as pure tones at distinct frequencies, letting you verify that the model composes symbols rather than memorizing utterances.

## ⤳ Training

```bash
./train.sh
```

Or directly:
```bash
uv run --with torch --with torchvision --with soundfile --with numpy --with scipy --with pillow --with pypng --with phase-spectrogram --reinstall-package torch --reinstall-package phase-spectrogram --with pygoruut --with tqdm python3 train.py
```

Training features:
- **Two-pass forward**: (1) detached AR SpecDecoder → AlignmentScorer → MAS → durations; (2) DurationPredictor + DurationDecoder (in-graph, trains encoder via both losses)
- DurationDecoder MSE loss on frame prediction (1536-dim phase vectors) — **main loss**
- DurationDecoder cross-entropy loss on EOS prediction
- Duration predictor MSE loss on log-durations (0.1× weight)
- AR auxiliary MSE + EOS losses (0.1× weight)
- Per-batch padding + masking
- Gradient clipping (CLIP_GRAD_NORM=1.0)
- Checkpoint saves model weights + tokenizer for standalone inference
- Generate sample audio every 100 epochs (and at epoch 1 and last) for monitoring
- `--finetune checkpoint.pt` to reload model + tokenizer and continue training
- `--limit N` to train on a subset of the dataset

Default hyperparameters in `config.py`:
| Parameter | Value |
|-----------|-------|
| D_MODEL | 256 |
| N_ENCODER_LAYERS | 2 |
| N_DECODER_LAYERS | 4 |
| N_HEADS | 8 |
| FF_DIM | 1024 |
| DROPOUT | 0.1 |
| LR | 3e-4 |
| BATCH_SIZE | 3 |
| EPOCHS | 10000 |
| MAX_TEXT_LEN | 1200 |
| MAX_SPEC_LEN | 2700 |
| SAMPLE_RATE | 48000 |
| DUR_HIDDEN | 256 |
| DUR_KERNEL | 3 |
| DUR_DROPOUT | 0.5 |
| DUR_LOSS_WEIGHT | 0.1 |
| AR_LOSS_WEIGHT | 0.1 |
| MIN_DURATION | 1 |
| CLIP_GRAD_NORM | 1.0 |

## ⤳ Inference

```bash
./generate.sh "text to speak"
```

Or directly:
```bash
uv run --with pygoruut --with torch --with soundfile --with phase-spectrogram python3 generate.py "text to speak" trained_model.pt
```

The inference pipeline:
1. Phonemize input text to IPA via Pygoruut
2. Encode IPA tokens using tokenizer from checkpoint
3. Predict per-token durations via DurationPredictor → round → clamp
4. Expand encoder memory by predicted durations (each token replicated for its frame count)
5. Autoregressive generation via DurationDecoder (causal self-attention, element-wise memory addition) with EOS-based stopping
6. Reshape flat 1536-dim frame vectors to 2-channel (768×2) array
7. Inverse phase transform → audio waveform → WAV file

## ⤳ Requirements

- [uv](https://docs.astral.sh/uv/) (Python package manager)
- PyTorch (automatically resolved by `uv run --with torch`)
- [phase-spectrogram](https://pypi.org/project/phase-spectrogram/)
- [Pygoruut](https://pypi.org/project/pygoruut/) — Slovak IPA phonemization
- soundfile, numpy, scipy, tqdm
- torchvision, pillow, pypng (for training)
- matplotlib (for diagnostic scripts)

All dependencies are resolved on-the-fly by `uv run --with ...` — no virtual environment setup needed.

## ⤳ Files

| File | Purpose |
|------|---------|
| `config.py` | Hyperparameters and paths |
| `model.py` | PositionalEncoding, TextEncoder, SpecDecoder (with PreNet, EOS, confidence), AlignmentScorer, DurationPredictor, DurationDecoder, DirectTTS |
| `mas.py` | Monotonic Alignment Search (Viterbi DP) and duration extraction |
| `dataset.py` | TextTokenizer, TTSDataset (WAV loading, IPA phonemization via Pygoruut, phase spec conversion) |
| `train.py` | Training loop with pad_collate, 2-pass forward, MAS alignment, duration/DurationDecoder/AR losses, checkpointing, `--finetune` and `--limit` CLI |
| `generate.py` | Inference from checkpoint using duration predictor + DurationDecoder |
| `reconstruct.py` | `spec_to_wav()` helper — inverse phase transform |
| `make_synthetic_dataset.py` | Generates synthetic tone-based dataset for debugging compositionality |
| `diagnose_attention.py` | Captures and plots cross-attention weight heatmaps from DurationDecoder |
| `diagnose_durations.py` | Prints MAS-extracted vs predicted durations per token |
| `train.sh` / `generate.sh` | Convenience shell wrappers |
