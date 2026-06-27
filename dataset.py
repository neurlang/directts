import torch
from torch.utils.data import Dataset
import soundfile as sf
import numpy as np
from phase import Phase

from config import SAMPLE_RATE, N_FREQS, N_CHANNELS, DATA_TSV, WAVS_DIR


class TextTokenizer:
    def __init__(self, texts=None):
        self.pad_token = "<pad>"
        self.bos_token = "<bos>"
        self.eos_token = "<eos>"
        self.unk_token = "<unk>"
        specials = [self.pad_token, self.bos_token, self.eos_token, self.unk_token]

        if texts is not None:
            chars = sorted(set("".join(texts)))
            self.vocab = {s: i for i, s in enumerate(specials + chars)}
        else:
            self.vocab = {s: i for i, s in enumerate(specials)}

        self.idx_to_token = {i: s for s, i in self.vocab.items()}
        self.vocab_size = len(self.vocab)

    def encode(self, text, add_special_tokens=True):
        ids = [self.vocab.get(c, self.vocab[self.unk_token]) for c in text]
        if add_special_tokens:
            ids = [self.vocab[self.bos_token]] + ids + [self.vocab[self.eos_token]]
        return torch.tensor(ids, dtype=torch.long)

    def decode(self, ids):
        return "".join(self.idx_to_token.get(i, self.unk_token) for i in ids.tolist())


class TTSDataset(Dataset):
    def __init__(self, tsv_path=DATA_TSV, wavs_dir=WAVS_DIR, sr=SAMPLE_RATE):
        self.phase = Phase(sample_rate=sr)
        self.sr = sr
        self.wavs_dir = wavs_dir

        with open(tsv_path, "r") as f:
            lines = [line.strip().split("\t") for line in f if line.strip()]

        self.paths = []
        self.texts = []
        for parts in lines:
            wav_path = parts[0]
            text = parts[1] if len(parts) > 1 else ""
            self.paths.append(wav_path)
            self.texts.append(text)

        self.tokenizer = TextTokenizer(self.texts)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        wav_path = self.paths[idx]
        text = self.texts[idx]

        audio, sr = sf.read(wav_path)
        assert sr == self.sr, f"Expected sr={self.sr}, got {sr} in {wav_path}"

        spec = self.phase.to_phase(audio)
        T = spec.shape[0] // N_FREQS
        spec = spec.reshape(T, N_FREQS, N_CHANNELS)

        spec = torch.from_numpy(spec).float()
        tokens = self.tokenizer.encode(text)

        return {
            "spec": spec,
            "spec_len": spec.shape[0],
            "text": text,
            "tokens": tokens,
            "tokens_len": len(tokens),
            "path": wav_path,
        }
