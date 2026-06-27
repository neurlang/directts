import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
import soundfile as sf
from phase import Phase

from model import DirectTTS
from dataset import TTSDataset
from config import LR, EPOCHS, CLIP_GRAD_NORM, SAMPLE_RATE, BATCH_SIZE
from tqdm import tqdm

def pad_collate(batch):
    specs = [item["spec"] for item in batch]
    tokens = [item["tokens"] for item in batch]
    texts = [item["text"] for item in batch]
    paths = [item["path"] for item in batch]

    max_T = max(s.size(0) for s in specs)
    max_L = max(t.size(0) for t in tokens)

    spec_padded = torch.stack([
        F.pad(s, (0, 0, 0, 0, 0, max_T - s.size(0)))
        for s in specs
    ])
    token_padded = torch.stack([
        F.pad(t, (0, max_L - t.size(0)))
        for t in tokens
    ])
    token_mask = token_padded == 0

    frame_mask = torch.zeros(len(batch), max_T, dtype=torch.bool)
    for i, s in enumerate(specs):
        frame_mask[i, s.size(0):] = True

    return {
        "spec": spec_padded,
        "spec_lens": torch.tensor([s.size(0) for s in specs]),
        "tokens": token_padded,
        "token_mask": token_mask,
        "frame_mask": frame_mask,
        "text": texts,
        "path": paths,
    }


def generate_sample(model, text, tokenizer, output_path, device):
    model.eval()
    tokens = tokenizer.encode(text).unsqueeze(0).to(device)
    gen_spec = model.generate(tokens)
    spec_np = gen_spec.squeeze(0).reshape(-1, 2).cpu().numpy()
    phase = Phase(sample_rate=SAMPLE_RATE)
    audio = phase.from_phase(spec_np)
    sf.write(output_path, audio, SAMPLE_RATE)
    model.train()


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = TTSDataset()
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=pad_collate)

    model = DirectTTS(ds.tokenizer.vocab_size).to(device)
    optim = AdamW(model.parameters(), lr=LR)

    print(f"Training on {len(ds)} samples. Model params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Samples: {ds.texts}")
    print("\nGround truth IPA variants:")
    for text, variants in zip(ds.texts, ds.ipa_texts):
        sample_ipa = next(iter(variants))
        tokens = ds.tokenizer.encode(sample_ipa)
        print(f'  "{text}" → {len(variants)} variant(s): {sorted(variants)}')
        print(f'    tokens: {tokens.tolist()}')
    print(f"\n{'Epoch':>6}  {'Frame':>8}  {'EOS':>8}  {'Total':>8}")
    print("-" * 36)

    for epoch in range(1, EPOCHS + 1):
        for batch in tqdm(loader):
            spec = batch["spec"].to(device)
            tokens = batch["tokens"].to(device)
            token_mask = batch["token_mask"].to(device)
            frame_mask = batch["frame_mask"].to(device)

            B, T, nF, C = spec.shape
            spec_flat = spec.view(B, T, -1)

            start = model.decoder.start_frame.expand(B, -1, -1)
            dec_in = torch.cat([start, spec_flat[:, :-1]], dim=1)

            frame_pred, eos_logits = model(tokens, dec_in, token_mask=token_mask, frame_mask=frame_mask)

            loss_frame = F.mse_loss(frame_pred, spec_flat, reduction="none")
            loss_frame = loss_frame[~frame_mask.unsqueeze(-1).expand_as(loss_frame)].mean()

            eos_target = torch.zeros(B, T, dtype=torch.long, device=spec.device)
            for b in range(B):
                eos_target[b, batch["spec_lens"][b] - 1] = 1
            loss_eos = F.cross_entropy(eos_logits.permute(0, 2, 1), eos_target, reduction="none")
            loss_eos = loss_eos[~frame_mask].mean()

            loss = loss_frame + 0.1 * loss_eos

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD_NORM)
            optim.step()

        if epoch == 1 or epoch % 100 == 0 or epoch == EPOCHS:
            for text in tqdm(ds.texts[:5] if len(ds.texts) >= 5 else ds.texts):
                safe = "".join(c if c.isalnum() else "_" for c in text)[:30]
                wav_path = f"train_{safe}_epoch_{epoch}.wav"
                generate_sample(model, text, ds.tokenizer, wav_path, device)
            print(f"{epoch:>6}  {loss_frame.item():>8.4f}  {loss_eos.item():>8.4f}  {loss.item():>8.4f}")

    torch.save({"model": model.state_dict(), "tokenizer": ds.tokenizer}, "trained_model.pt")
    print(f"\nFinal frame loss: {loss_frame.item():.6f}, EOS loss: {loss_eos.item():.4f}")
    print("Model saved to trained_model.pt")


if __name__ == "__main__":
    train()
