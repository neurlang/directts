import gc
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
from config import DUR_LOSS_WEIGHT, AR_LOSS_WEIGHT
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
    with torch.no_grad():
        tokens = tokenizer.encode(text).unsqueeze(0).to(device)
        gen_spec = model.generate(tokens)
        spec_np = gen_spec.squeeze(0).reshape(-1, 2).cpu().numpy()
        del gen_spec, tokens
    phase = Phase(sample_rate=SAMPLE_RATE)
    audio = phase.from_phase(spec_np)
    sf.write(output_path, audio, SAMPLE_RATE)
    model.train()
    gc.collect()
    torch.cuda.empty_cache()


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
    print(f"\n{'Epoch':>6}  {'DD-Frm':>8}  {'DD-EOS':>8}  {'Dur':>8}  {'AR-Frm':>8}  {'AR-EOS':>8}  {'Total':>8}")
    print("-" * 60)

    for epoch in range(1, EPOCHS + 1):
        for batch in tqdm(loader):
            spec = batch["spec"].to(device)            # (B, T, nF, C)
            tokens = batch["tokens"].to(device)
            token_mask = batch["token_mask"].to(device)
            frame_mask = batch["frame_mask"].to(device)
            spec_lens = batch["spec_lens"].to(device)

            B, T, nF, C = spec.shape
            spec_flat = spec.view(B, T, -1)            # (B, T, N_FREQS*N_CHANNELS)

            # ── Forward ──
            out = model(tokens, spec_flat, token_mask=token_mask,
                        frame_mask=frame_mask, spec_lens=spec_lens)

            # ── DurationDecoder losses (main) ──
            dd_preds = out["dd_preds"]                # (B, max_len, frame_size)
            dd_eos = out["dd_eos"]                    # (B, max_len, 2)

            # Build mask matching the expanded length
            max_len_dd = dd_preds.size(1)
            dd_mask = torch.ones(B, max_len_dd, dtype=torch.bool, device=device)
            for b in range(B):
                dd_mask[b, :spec_lens[b]] = False

            dd_loss_frame = F.mse_loss(
                dd_preds[~dd_mask],
                spec_flat[:, :max_len_dd][~dd_mask])

            eos_target = torch.zeros(B, max_len_dd, dtype=torch.long, device=device)
            for b in range(B):
                eos_target[b, min(spec_lens[b], max_len_dd) - 1] = 1
            dd_loss_eos = F.cross_entropy(dd_eos.permute(0, 2, 1), eos_target, reduction="none")
            dd_loss_eos = dd_loss_eos[~dd_mask].mean()

            # ── Duration predictor loss ──
            dur_pred = out["dur_pred"]                # (B, T_text)
            durations = out["durations"]               # (B, T_text)
            dur_loss = F.mse_loss(
                dur_pred[~token_mask].log(),
                durations[~token_mask].float().log())

            # ── AR losses (auxiliary) ──
            ar_preds = out["ar_preds"]
            ar_eos = out["ar_eos"]

            ar_loss_frame = F.mse_loss(
                ar_preds[~frame_mask.unsqueeze(-1).expand_as(ar_preds)],
                spec_flat[~frame_mask.unsqueeze(-1).expand_as(spec_flat)])

            ar_eos_target = torch.zeros(B, T, dtype=torch.long, device=device)
            for b in range(B):
                ar_eos_target[b, spec_lens[b] - 1] = 1
            ar_loss_eos = F.cross_entropy(ar_eos.permute(0, 2, 1), ar_eos_target, reduction="none")
            ar_loss_eos = ar_loss_eos[~frame_mask].mean()

            # ── Total ──
            loss = (
                dd_loss_frame + dd_loss_eos
                + DUR_LOSS_WEIGHT * dur_loss
                + AR_LOSS_WEIGHT * (ar_loss_frame + ar_loss_eos)
            )

            optim.zero_grad()

            if epoch == 1 or epoch % 100 == 0 or epoch == EPOCHS:
                with torch.no_grad():
                    diff = (dd_preds - spec_flat[:, :max_len_dd, :]) ** 2
                    mask_expanded = (~dd_mask).unsqueeze(-1)
                    loss_per_sample = (diff * mask_expanded).sum(dim=[1, 2]) / mask_expanded.sum(dim=[1, 2]).clamp(min=1)
                    for b in range(min(4, B)):
                        print(f"  sample {b}: DD-MSE={loss_per_sample[b]:.6f}  n_valid={mask_expanded[b].sum().item()}")

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD_NORM)
            optim.step()

        # Save loss scalars before freeing tensors
        epoch_dd_frame = dd_loss_frame.item()
        epoch_dd_eos = dd_loss_eos.item()
        epoch_dur = dur_loss.item()
        epoch_ar_frame = ar_loss_frame.item()
        epoch_ar_eos = ar_loss_eos.item()
        epoch_loss = loss.item()

        # Free GPU tensors from last batch before eval generation
        del spec, tokens, token_mask, frame_mask, spec_lens
        del spec_flat, dd_preds, dd_eos, dd_mask, eos_target
        del dur_pred, durations, ar_preds, ar_eos, ar_eos_target
        del out
        del loss, dd_loss_frame, dd_loss_eos, dur_loss, ar_loss_frame, ar_loss_eos
        try:
            del diff, loss_per_sample, mask_expanded
        except NameError:
            pass

        # Free gradients
        optim.zero_grad(set_to_none=True)

        # Offload optimizer state to CPU to free GPU memory
        optim_sd = optim.state_dict()
        for group in optim.param_groups:
            for p in group['params']:
                optim.state[p].clear()

        gc.collect()
        torch.cuda.empty_cache()

        if True:
            torch.save({"model": model.state_dict(), "tokenizer": ds.tokenizer}, f"checkpoint_model_{epoch}.pt")
            for text in tqdm(ds.texts[:5] if len(ds.texts) >= 5 else ds.texts):
                safe = "".join(c if c.isalnum() else "_" for c in text)[:30]
                wav_path = f"train_{safe}_epoch_{epoch}.wav"
                generate_sample(model, text, ds.tokenizer, wav_path, device, ds.pygoruut, ds.language)
                gc.collect()
                torch.cuda.empty_cache()
            print(f"{epoch:>6}  {epoch_dd_frame:>8.4f}  {epoch_dd_eos:>8.4f}  "
                  f"{epoch_dur:>8.4f}  {epoch_ar_frame:>8.4f}  "
                  f"{epoch_ar_eos:>8.4f}  {epoch_loss:>8.4f}")

        # Restore optimizer state back to GPU
        optim.load_state_dict(optim_sd)

    torch.save({"model": model.state_dict(), "tokenizer": ds.tokenizer}, "trained_model.pt")
    print(f"\nFinal DD frame loss: {epoch_dd_frame:.6f}, DD EOS loss: {epoch_dd_eos:.4f}, "
          f"Dur loss: {epoch_dur:.4f}")
    print("Model saved to trained_model.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--finetune", type=str, default=None,
                        help="Path to checkpoint to fine-tune from (resets epoch to 0)")
    parser.add_argument("--limit", type=int, default=-1,
                        help="Dataset file count limit")
    args = parser.parse_args()
    train(finetune_checkpoint=args.finetune, limit=args.limit)
