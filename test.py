"""Run PSAFNet inference on the POSDiag 25%-FOV split."""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from psafnet import POSDiagDataset, PSAFNet, load_psafnet_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, help="Path to the POSDiag directory")
    parser.add_argument("--weights", required=True, help="PSAFNet .pth checkpoint")
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--split-file", default=None)
    parser.add_argument("--output-dir", default="results/posdiag_fov25")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--device", default=None, help="For example: cuda, cuda:0, or cpu")
    parser.add_argument("--max-samples", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    dataset = POSDiagDataset(
        args.data_root,
        split=args.split,
        split_file=args.split_file,
        max_samples=args.max_samples,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = PSAFNet(pretrained_backbone=False)
    metadata = load_psafnet_checkpoint(model, args.weights, map_location="cpu", strict=True)
    model = model.to(device=device)
    model.eval()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    losses = []

    with torch.inference_mode():
        for batch in tqdm(loader, desc="Testing", unit="batch"):
            image = batch["image"].to(device, dtype=torch.float32, non_blocking=True)
            target = batch["target"].to(device, dtype=torch.float32, non_blocking=True)
            template = batch["template"].to(device, dtype=torch.float32, non_blocking=True)
            region_index = batch["region_index"].to(device, dtype=torch.long, non_blocking=True)

            prediction = model(image, template, region_index).unsqueeze(1)
            if not torch.isfinite(prediction).all():
                raise FloatingPointError("Model prediction contains NaN or Inf values")

            per_sample_loss = F.smooth_l1_loss(
                prediction,
                target,
                reduction="none",
            ).flatten(start_dim=1).mean(dim=1)
            losses.extend(per_sample_loss.cpu().tolist())

            predictions = prediction[:, 0].float().cpu().numpy()
            regions = batch["region_index"].cpu().tolist()
            for array, case_id, region in zip(predictions, batch["case_id"], regions):
                output_path = output_dir / "case_{}_region_{:02d}.npy".format(case_id, region)
                np.save(str(output_path), array.astype(np.float32))

    print("Loaded checkpoint metadata: {}".format(metadata))
    print("Saved {} predictions to {}".format(len(dataset), output_dir))
    print("Mean Smooth L1: {:.6f}".format(float(np.mean(losses))))


if __name__ == "__main__":
    main()
