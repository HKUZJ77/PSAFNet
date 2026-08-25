"""Train PSAFNet on the POSDiag 25%-FOV R-channel reconstruction task."""

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from psafnet import POSDiagDataset, PSAFNet, load_psafnet_checkpoint
from util.loss_functions import PSAFNetLoss


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, help="Path to the POSDiag directory")
    parser.add_argument("--output-dir", default="runs/posdiag_fov25")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default=None, help="For example: cuda, cuda:0, or cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--initial-weights", default=None)
    parser.add_argument("--no-pretrained-backbone", action="store_true")
    parser.add_argument("--train-split-file", default=None)
    parser.add_argument("--val-split-file", default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    return parser.parse_args()


def resolve_device(requested):
    if requested:
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_batch(batch, device):
    return (
        batch["image"].to(device, dtype=torch.float32, non_blocking=True),
        batch["target"].to(device, dtype=torch.float32, non_blocking=True),
        batch["template"].to(device, dtype=torch.float32, non_blocking=True),
        batch["region_index"].to(device, dtype=torch.long, non_blocking=True),
    )


@torch.no_grad()
def validate(model, loader, loss_function, device):
    model.eval()
    total_loss = 0.0
    total_samples = 0

    for batch in loader:
        image, target, template, region_index = move_batch(batch, device)
        prediction = model(image, template, region_index).unsqueeze(1)
        loss, _ = loss_function(prediction, target)
        total_loss += loss.item() * image.shape[0]
        total_samples += image.shape[0]

    return total_loss / max(total_samples, 1)


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = resolve_device(args.device)

    train_dataset = POSDiagDataset(
        args.data_root,
        split="train",
        split_file=args.train_split_file,
        max_samples=args.max_train_samples,
    )
    val_dataset = POSDiagDataset(
        args.data_root,
        split="val",
        split_file=args.val_split_file,
        max_samples=args.max_val_samples,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = PSAFNet(pretrained_backbone=not args.no_pretrained_backbone)
    if args.initial_weights:
        load_psafnet_checkpoint(model, args.initial_weights, map_location="cpu", strict=True)
    model = model.to(device=device)

    loss_function = PSAFNetLoss(perceptual_weight=0.1, emd_weight=0.1, num_bins=256)
    loss_function = loss_function.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=20,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")

    print("Dataset: POSDiag, FOV: 25%, coordinate: RUV -> R")
    print("Device: {} | train: {} | val: {}".format(device, len(train_dataset), len(val_dataset)))

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        seen_samples = 0
        progress = tqdm(train_loader, desc="Epoch {}/{}".format(epoch, args.epochs), unit="batch")

        for batch in progress:
            image, target, template, region_index = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)

            prediction = model(image, template, region_index).unsqueeze(1)
            loss, components = loss_function(prediction, target)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            batch_size = image.shape[0]
            running_loss += loss.item() * batch_size
            seen_samples += batch_size
            progress.set_postfix(
                loss="{:.5f}".format(loss.item()),
                l1="{:.5f}".format(components["smooth_l1"].item()),
                emd="{:.5f}".format(components["emd"].item()),
            )

        train_loss = running_loss / max(seen_samples, 1)
        val_loss = validate(model, val_loader, loss_function, device)
        scheduler.step(val_loss)
        print(
            "Epoch {}: train_loss={:.6f}, val_loss={:.6f}, lr={:.3e}".format(
                epoch, train_loss, val_loss, optimizer.param_groups[0]["lr"]
            )
        )

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "best_loss": best_loss,
                    "dataset": "POSDiag",
                    "fov": "25%",
                    "coordinate": "RUV-to-R",
                    "model_state_dict": model.state_dict(),
                },
                str(output_dir / "best_model.pth"),
            )
            print("Saved {}".format(output_dir / "best_model.pth"))


if __name__ == "__main__":
    main()
