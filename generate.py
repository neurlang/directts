import sys
import torch
from model import DirectTTS
from dataset import TTSDataset
from reconstruct import spec_to_wav


def main():
    text = sys.argv[1] if len(sys.argv) > 1 else "ano"
    checkpoint = sys.argv[2] if len(sys.argv) > 2 else "trained_model.pt"

    ds = TTSDataset()
    model = DirectTTS(ds.tokenizer.vocab_size)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    model.eval()

    tokens = ds.tokenizer.encode(text).unsqueeze(0)
    gen_spec = model.generate(tokens)
    frames = gen_spec.shape[1]

    out_path = f"gen_{text}.wav"
    spec_to_wav(gen_spec.squeeze(0).numpy(), out_path)
    print(f"Generated {out_path} — {frames} frames, {ds.tokenizer.encode(text).tolist()}")


if __name__ == "__main__":
    main()
