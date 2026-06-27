import soundfile as sf
import numpy as np
from phase import Phase
import torch

from dataset import TTSDataset
from config import SAMPLE_RATE, N_FREQS, N_CHANNELS


def spec_to_wav(spec, output_path, sr=SAMPLE_RATE):
    phase = Phase(sample_rate=sr)
    spec = spec.reshape(-1, 2)
    audio = phase.from_phase(spec)
    sf.write(output_path, audio, sr)


if __name__ == "__main__":
    ds = TTSDataset()
    for i in range(len(ds)):
        item = ds[i]
        spec = item["spec"]
        spec_to_wav(spec.numpy(), f"reconstructed_{i}.wav")
        print(f"reconstructed_{i}.wav — '{item['text']}' — spec shape {tuple(spec.shape)}")
