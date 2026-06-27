import sys
import torch
from model import DirectTTS
from dataset import TTSDataset
from reconstruct import spec_to_wav
from pygoruut.pygoruut import Pygoruut


def main():
    text = sys.argv[1] if len(sys.argv) > 1 else "ano"
    checkpoint_path = sys.argv[2] if len(sys.argv) > 2 else "trained_model.pt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pg = Pygoruut(writeable_bin_dir="")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    tokenizer = ckpt["tokenizer"]

    model = DirectTTS(tokenizer.vocab_size).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    ipa = str(pg.phonemize(language='Slovak', sentence=text))

    tokens = tokenizer.encode(ipa).unsqueeze(0).to(device)
    gen_spec = model.generate(tokens)
    frames = gen_spec.shape[1]

    out_path = f"gen_{text}.wav"
    spec_to_wav(gen_spec.squeeze(0).cpu().numpy(), out_path)
    print(f"Generated {out_path} — {frames} frames, IPA: {ipa}, tokens: {tokenizer.encode(ipa).tolist()}")
    print(f"Pygoruut version {pg.version}")

if __name__ == "__main__":
    main()
