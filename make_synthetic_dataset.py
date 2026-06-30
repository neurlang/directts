"""
Generates a synthetic dataset/wavs48khz/ + dataset/dataset.tsv where each
"word" is built from letters a, b, c (...) and each letter is rendered as a
fixed-duration pure tone at a distinct, known frequency. This lets you train
through your REAL pipeline (dataset.py phonemization + spec extraction +
train.sh/generate.sh + reconstruct.py) while still knowing the exact expected
acoustic content of every training and probe word -- so you can later decode
generated audio back to a symbol sequence and check it's actually composing,
not parroting.

Run from your directts repo root:
    python make_synthetic_dataset.py

Then point train.sh at the generated dataset/dataset.tsv (it should already
match config.py's DATA_TSV / WAVS_DIR paths) and train as usual.

NOTE: text goes through your real phonemizer (Pygoruut, Slovak), so the
token sequence the model actually sees won't be a clean 1:1 map to letters --
that's expected and fine. What matters is that different input words still
produce correspondingly different, position-correct tone sequences in the
generated audio. If you want letter-level token control instead, you'll need
to bypass phonemization for these synthetic rows (see comment near the
bottom).
"""

import itertools
import os
import numpy as np
import soundfile as sf

from config import SAMPLE_RATE, DATA_TSV, WAVS_DIR

# ---------------------------------------------------------------------------
ALPHABET = ["a", "b", "s"]
FREQ_HZ = {"a": 300.0, "b": 600.0, "s": 900.0}   # widely separated pure tones
SYMBOL_DURATION_S = 0.15      # 150ms per letter
FADE_S = 0.01                 # 10ms raised-cosine fade in/out, avoids clicks
SILENCE_BETWEEN_S = 0.0       # set >0 if you want a gap between letters

TRAIN_MAX_LEN = 3             # train on all 1,2,3-letter combos of ALPHABET
PROBE_WORDS = ["ababa", "abbba", "aabbss", "ssbbaa", "absab", "babab"]
WRITE_PROBES_AS_REFERENCE_ONLY = True   # probes NOT added to dataset.tsv


def make_tone(freq_hz: float, duration_s: float, sample_rate: int, fade_s: float) -> np.ndarray:
    n = int(duration_s * sample_rate)
    t = np.arange(n) / sample_rate
    wave = 0.5 * np.sin(2 * np.pi * freq_hz * t).astype(np.float32)

    fade_n = int(fade_s * sample_rate)
    if fade_n > 0 and fade_n * 2 < n:
        ramp = 0.5 * (1 - np.cos(np.pi * np.arange(fade_n) / fade_n))
        wave[:fade_n] *= ramp
        wave[-fade_n:] *= ramp[::-1]
    return wave


def word_to_audio(word: str, sample_rate: int) -> np.ndarray:
    parts = []
    silence = np.zeros(int(SILENCE_BETWEEN_S * sample_rate), dtype=np.float32)
    for i, ch in enumerate(word):
        parts.append(make_tone(FREQ_HZ[ch], SYMBOL_DURATION_S, sample_rate, FADE_S))
        if SILENCE_BETWEEN_S > 0 and i < len(word) - 1:
            parts.append(silence)
    return np.concatenate(parts)


def build_train_words(max_len: int) -> list[str]:
    words = set()
    for L in range(1, max_len + 1):
        for combo in itertools.product(ALPHABET, repeat=L):
            words.add("".join(combo))
    words -= set(PROBE_WORDS)
    return sorted(words)


def main():
    wavs_dir = WAVS_DIR
    tsv_path = DATA_TSV
    os.makedirs(wavs_dir, exist_ok=True)
    os.makedirs(os.path.dirname(tsv_path) or ".", exist_ok=True)

    train_words = build_train_words(TRAIN_MAX_LEN)
    print(f"Generating {len(train_words)} training wavs into {wavs_dir}/ at {SAMPLE_RATE} Hz")

    rows = []
    for word in train_words:
        audio = word_to_audio(word, SAMPLE_RATE)
        fname = f"syn_{word}.wav"
        fpath = os.path.join(wavs_dir, fname)
        sf.write(fpath, audio, SAMPLE_RATE, subtype="PCM_16")
        # dataset.tsv format per README: path<TAB>text<TAB>text
        rows.append(f"{fpath}\t{word}\t{word}")

    with open(tsv_path, "w", encoding="utf-8") as f:
        f.write("\n".join(rows) + "\n")
    print(f"Wrote {len(rows)} rows to {tsv_path}")

    if WRITE_PROBES_AS_REFERENCE_ONLY:
        probe_ref_path = os.path.join(os.path.dirname(tsv_path) or ".", "probe_words_reference.tsv")
        probe_rows = []
        for word in PROBE_WORDS:
            audio = word_to_audio(word, SAMPLE_RATE)
            fpath = os.path.join(wavs_dir, f"probe_{word}.wav")
            sf.write(fpath, audio, SAMPLE_RATE, subtype="PCM_16")  # ground-truth reference, NOT for training
            probe_rows.append(f"{fpath}\t{word}\t{word}")
        with open(probe_ref_path, "w", encoding="utf-8") as f:
            f.write("\n".join(probe_rows) + "\n")
        print(f"Wrote {len(probe_rows)} held-out probe reference wavs "
              f"(ground truth only, NOT in {tsv_path}) -> {probe_ref_path}")
        print(f"Probe words (use these with generate.sh after training): {PROBE_WORDS}")

    print("\nFREQ_HZ mapping (needed later to decode generated audio):")
    for k, v in FREQ_HZ.items():
        print(f"  {k!r}: {v} Hz")
    print(f"SYMBOL_DURATION_S={SYMBOL_DURATION_S}  (used to estimate expected frame counts)")


if __name__ == "__main__":
    main()
