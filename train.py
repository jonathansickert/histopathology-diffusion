import argparse
import copy
import json
from pathlib import Path

import torch
import torch.nn as nn
from torchvision.utils import save_image

from dataset import HistopathologyDataset
from model import DiffusionModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EMA_DECAY = 0.999


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--out-dir", default="checkpoint")
    parser.add_argument("--sample-every", type=int, default=500)
    parser.add_argument("--fid-every", type=int, default=500)
    parser.add_argument("--sample-steps", type=int, default=20)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()

def corrupt(x, amount):
    """Corrupt the input `x` by mixing it with noise according to `amount`.
    Returns the noisy image and the noise used."""
    noise = torch.randn_like(x)
    amount = amount.view(-1, 1, 1, 1)  # Sort shape so broadcasting works
    return x * (1 - amount) + noise * amount, noise


@torch.no_grad()
def update_ema(ema, model, iteration):
    # Warm up the decay so early EMA weights aren't dominated by the random init.
    decay = min(EMA_DECAY, (1 + iteration) / (10 + iteration))
    for e, p in zip(ema.parameters(), model.parameters()):
        e.lerp_(p, 1 - decay)


def main():
    args = parse_args()
    print("Using device:", device)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    samples_dir = out / "samples"
    samples_dir.mkdir(exist_ok=True)
    ckpt_path = out / "diffusion.pt"
    with open(out / "config.json", "w") as f:
        json.dump(vars(args), f, indent=1)

    train_ds = HistopathologyDataset()
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    model = DiffusionModel()
    model = model.to(device)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    fixed_noise = torch.randn((64, 3, 128, 128), device=device)
    start_epoch = 0
    iteration = 0
    losses = []
    fid_history = []
    best_fid = float("inf")

    if args.resume and ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        ema.load_state_dict(ckpt.get("ema", ckpt["model"]))
        optimizer.load_state_dict(ckpt["optimizer"])
        fixed_noise = ckpt["fixed_noise"].to(device)
        start_epoch = ckpt["epoch"]
        iteration = ckpt["iteration"]
        losses = ckpt["losses"]
        fid_history = ckpt.get("fid_history", [])
        best_fid = min((f for _, _, f in fid_history), default=float("inf"))
        print(f"Resumed from epoch {start_epoch}, iteration {iteration}")


    for epoch in range(start_epoch, args.epochs):
        model.train()

        for x in train_loader:
            x = x.to(device)
            noise_amount = torch.rand(x.shape[0]).to(device)
            noisy_x, noise = corrupt(x, noise_amount)

            pred = model(noisy_x, noise_amount)
            loss = loss_fn(pred, noise - x)  # velocity target

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            update_ema(ema, model, iteration)
            losses.append(loss.item())
            
            iteration += 1
            if iteration % args.sample_every == 0:
                samples = ema.sample(fixed_noise, steps=args.sample_steps)
                save_image(samples, samples_dir / f"iter_{iteration:07d}.png", nrow=8, normalize=True, value_range=(-1, 1))


        n = len(train_loader)
        print(f"epoch {epoch + 1}/{args.epochs} iter {iteration} Loss {sum(losses[-n:]) / n:.3f}")

        state = {
            "model": model.state_dict(),
            "ema": ema.state_dict(),
            "optimizer": optimizer.state_dict(),
            "fixed_noise": fixed_noise,
            "epoch": epoch + 1,
            "iteration": iteration,
            "args": vars(args),
            "losses": losses,
            "fid_history": fid_history,
        }
        torch.save(state, ckpt_path)


if __name__ == "__main__":
    main()
