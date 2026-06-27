# DirecTTS
## [zero vocoder, TTS transformer only]

Direct phase-spectrogram neural text-to-speech with encoder-decoder transformer.

Maps text to raw phase spectrogram frames (768×2 per frame) in a single autoregressive pass, then reconstructs audio via the inverse phase transform — no vocoder, no intermediate representations.

## Architecture

```
Text → IPA phonemes → TextEncoder(6-layer transformer) → memory
                                                              ↓
Spec frames ← output_proj ← SpecDecoder(4-layer transformer) ← prenet ← teacher-forced / autoregressive frames
                                  ↓
                              eos_head → stop probability
```

- **TextEncoder**: 6-layer transformer encoder (D_MODEL=256, 8 heads)
- **SpecDecoder**: 4-layer transformer decoder with Tacotron 2-style PreNet (Linear→ReLU→Dropout→Linear→ReLU→Dropout at 0.5 rate)
- **PostNet**: none — PreNet alone prevents repeated speech by forcing decoder cross-attention over frame-level memorization
- **EOS head**: binary classifier on decoder outputs to predict the final frame, enabling variable-length generation
- **Positional encoding**: sinusoidal, applied to both encoder and decoder inputs

## Dataset

Slovak speech from [SlovakSpeech](https://huggingface.co/datasets/neurlang/slovakspeech_female_dataset) corpus, 48 kHz mono WAV files listed in `dataset/dataset.tsv` in `path<TAB>text<TAB>text` format.

Text is phonemized to IPA via [Pygoruut](https://pypi.org/project/pygoruut/) before tokenization. The tokenizer builds a character-level vocabulary from all unique IPA characters across the dataset.

## Training

```bash
./train.sh
```

Or directly:
```bash
uv run --with torch --with soundfile --with phase-spectrogram --with pygoruut --with tqdm python3 train.py
```

Training features:
- Teacher forcing with frame-shifted decoder input
- MSE loss on frame prediction (1536-dim phase vectors)
- Cross-entropy loss on EOS prediction (0.1× weight)
- Per-batch padding + masking
- Checkpoint saves model weights + tokenizer for standalone inference
- Generate sample audio every 100 epochs for monitoring

Default hyperparameters in `config.py`:
| Parameter | Value |
|-----------|-------|
| D_MODEL | 256 |
| N_ENCODER_LAYERS | 6 |
| N_DECODER_LAYERS | 4 |
| N_HEADS | 8 |
| FF_DIM | 1024 |
| DROPOUT | 0.1 |
| LR | 3e-4 |
| BATCH_SIZE | 7 |
| EPOCHS | 10000 |
| MAX_SPEC_LEN | 200 |
| SAMPLE_RATE | 48000 |

## Inference

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
3. Autoregressive generation with EOS-based stopping
4. Reshape flat phase frames to 2-channel array
5. Inverse phase transform → audio waveform → WAV file

## Requirements

- [uv](https://docs.astral.sh/uv/) (Python package manager)
- PyTorch (automatically resolved by `uv run --with torch`)
- [phase-spectrogram](https://pypi.org/project/phase-spectrogram/)
- [Pygoruut](https://pypi.org/project/pygoruut/) — Slovak IPA phonemization
- soundfile, numpy, scipy, tqdm

All dependencies are resolved on-the-fly by `uv run --with ...` — no virtual environment setup needed.

## Files

| File | Purpose |
|------|---------|
| `config.py` | Hyperparameters and paths |
| `model.py` | PositionalEncoding, TextEncoder, SpecDecoder (with PreNet), DirectTTS |
| `dataset.py` | TextTokenizer, TTSDataset (WAV loading, IPA phonemization, spec conversion) |
| `train.py` | Training loop with pad_collate, loss computation, checkpointing |
| `generate.py` | Inference from checkpoint |
| `reconstruct.py` | `spec_to_wav()` helper — inverse phase transform |
| `train.sh` / `generate.sh` | Convenience shell wrappers |
