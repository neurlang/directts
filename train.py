import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
import soundfile as sf
from phase import Phase

from model import DirectTTS
from dataset import TTSDataset
from config import LR, EPOCHS, CLIP_GRAD_NORM, SAMPLE_RATE, BATCH_SIZE, N_FREQS, N_CHANNELS
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


def generate_sample(model, text, tokenizer, output_path, device, pygoruut, language):
    text = str(pygoruut.phonemize(language=language, sentence=text))
    model.eval()
    tokens = tokenizer.encode(text).unsqueeze(0).to(device)
    gen_spec = model.generate(tokens)
    spec_np = gen_spec.squeeze(0).reshape(-1, 2).cpu().numpy()
    phase = Phase(sample_rate=SAMPLE_RATE)
    audio = phase.from_phase(spec_np)
    sf.write(output_path, audio, SAMPLE_RATE)
    model.train()


def train(finetune_checkpoint=None, limit=-1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = TTSDataset(limit=limit)

    if finetune_checkpoint:
        checkpoint = torch.load(finetune_checkpoint, map_location=device, weights_only=False)
        ds.tokenizer = checkpoint["tokenizer"]
        print(f"Loaded checkpoint from {finetune_checkpoint}")
    else:
        checkpoint = None

    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=pad_collate)

    model = DirectTTS(ds.tokenizer.vocab_size).to(device)
    if finetune_checkpoint:
        model.load_state_dict(checkpoint["model"])
        print("Model weights loaded for fine-tuning")

    optim = AdamW(model.parameters(), lr=LR)

    print(f"Training on {len(ds)} samples. Model params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Samples: {ds.texts}")
    print("\nGround truth IPA variants:")
    for text, variants in zip(ds.texts, ds.ipa_texts):
        sample_ipa = next(iter(variants))
        tokens = ds.tokenizer.encode(sample_ipa)
        print(f'  "{text}" → {len(variants)} variant(s): {sorted(variants)}')
        print(f'    tokens: {tokens.tolist()}')
    print(f"\n{'Epoch':>6}  {'Frame':>8}  {'EOS':>8}  {'Conf':>8}  {'Total':>8}")
    print("-" * 46)

    for epoch in range(1, EPOCHS + 1):
        ar_rate = max(0.1, 0.4 - epoch / EPOCHS * 0.3)
        for batch in tqdm(loader):
            spec = batch["spec"].to(device)
            tokens = batch["tokens"].to(device)
            token_mask = batch["token_mask"].to(device)
            frame_mask = batch["frame_mask"].to(device)

            B, T, nF, C = spec.shape
            spec_flat = spec.view(B, T, -1)

            use_ar = torch.rand(1).item() < ar_rate
            frame_pred, eos_logits, mask, confidence = model(
                tokens, spec_flat, token_mask=token_mask,
                frame_mask=frame_mask, use_ar=use_ar)

            if use_ar:
                loss_frame = F.mse_loss(
                    frame_pred[~frame_mask.unsqueeze(-1).expand_as(frame_pred)],
                    spec_flat[~frame_mask.unsqueeze(-1).expand_as(spec_flat)])
                loss_conf = torch.tensor(0.0, device=device)
            else:
                valid_loss = ~frame_mask & mask
                loss_frame = F.mse_loss(
                    frame_pred[valid_loss],
                    spec_flat[valid_loss])

                # confidence target: high where prediction is accurate, low where it isn't
                # detach frame_pred so confidence head doesn't interfere with frame prediction gradients
                with torch.no_grad():
                    per_frame_err = F.mse_loss(
                        frame_pred.detach(), spec_flat, reduction='none'
                    ).mean(dim=-1, keepdim=True)           # (B, T, 1)
                    target_conf = torch.exp(-per_frame_err) # 1.0 = perfect, ~0 = bad
                # only supervise on valid (non-padding) frames
                valid_conf = ~frame_mask                    # (B, T)
                loss_conf = F.mse_loss(
                    confidence[valid_conf].squeeze(-1),
                    target_conf.squeeze(-1)[valid_conf])

            eos_target = torch.zeros(B, T, dtype=torch.long, device=spec.device)
            for b in range(B):
                eos_target[b, batch["spec_lens"][b] - 1] = 1
            loss_eos = F.cross_entropy(eos_logits.permute(0, 2, 1), eos_target, reduction="none")
            loss_eos = loss_eos[~frame_mask].mean()

            loss = loss_frame + 0.1 * loss_eos + 0.05 * loss_conf

            optim.zero_grad()

            if epoch == 1 or epoch % 100 == 0 or epoch == EPOCHS:
                with torch.no_grad():
                    loss_per_sample_raw = ((frame_pred - spec_flat)**2).mean(dim=[1,2])
                    diff = (frame_pred - spec_flat)**2
                    mask_expanded = (~frame_mask).unsqueeze(-1)
                    loss_per_sample_masked = (diff * mask_expanded).sum(dim=[1,2]) / mask_expanded.sum(dim=[1,2]).clamp(min=1)
                    for b in range(min(4, B)):
                        print(f"  sample {b}: raw MSE={loss_per_sample_raw[b]:.6f}  masked MSE={loss_per_sample_masked[b]:.6f}  n_valid={mask_expanded[b].sum().item()}")

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD_NORM)
            optim.step()

        if epoch == 1 or epoch % 100 == 0 or epoch == EPOCHS:
            for text in tqdm(ds.texts[:5] if len(ds.texts) >= 5 else ds.texts):
                safe = "".join(c if c.isalnum() else "_" for c in text)[:30]
                wav_path = f"train_{safe}_epoch_{epoch}.wav"
                generate_sample(model, text, ds.tokenizer, wav_path, device, ds.pygoruut, ds.language)
            print(f"{epoch:>6}  {loss_frame.item():>8.4f}  {loss_eos.item():>8.4f}  {loss_conf.item():>8.4f}  {loss.item():>8.4f}")
            torch.save({"model": model.state_dict(), "tokenizer": ds.tokenizer}, f"checkpoint_model_{epoch}.pt")
    
    torch.save({"model": model.state_dict(), "tokenizer": ds.tokenizer}, "trained_model.pt")
    print(f"\nFinal frame loss: {loss_frame.item():.6f}, EOS loss: {loss_eos.item():.4f}")
    print("Model saved to trained_model.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--finetune", type=str, default=None,
                        help="Path to checkpoint to fine-tune from (resets epoch to 0)")
    parser.add_argument("--limit", type=int, default=-1,
                        help="Dataset file count limit")
    args = parser.parse_args()
    train(finetune_checkpoint=args.finetune, limit=args.limit)
