# PSAFNet

Official PyTorch implementation of **Polar Subarea-Aware Fusion Net for Posterior Eyeball Shape Reconstruction**.

This compact release supports the POSDiag 25%-FOV configuration. PSAFNet takes a local OCT/iOCT scan and an anatomical template as input to reconstruct the complete and high-fidelity retinal morphology. The released model uses a siamese hybrid backbone with a fully 2D multi-channel representation, called Ocular Shape Map (OSM), therefore converting this 3D reconstruction task into a dense completion task.

Paper: [IEEE Transactions on Medical Imaging](https://doi.org/10.1109/TMI.2025.3642381)

## Installation

Python 3.9 or later is recommended. The release was tested with Python 3.9.21, PyTorch 2.6.0, torchvision 0.21.0, NumPy 1.26.4, OpenCV 4.6.0, and timm 0.4.12.

```bash
python -m venv .venv
pip install -r requirements.txt
```

Activate the virtual environment before installation if required by your operating system. Training may download ImageNet weights for the hybrid backbone and VGG-19 perceptual network on first use. Testing with the released checkpoint does not download pretrained backbone weights.

## Weights

Download `psafnet_posdiag_fov25_r.pth` from the repository's [GitHub Releases](https://github.com/HKUZJ77/PSAFNet/releases).

- Size: 952,888,084 bytes (0.887 GiB)

## Data preparation

Dataset files are not included in this repository. Arrange POSDiag as follows:

```text
POSDiag/
├── global_mc_img/
│   └── <case_id>_glob_mcimg.npy
├── subarea_input/
│   └── <case_id>_<region_id>subarea.npy
├── template/
│   └── template12_mcimg.npy
└── txt/subarea_txt/
    ├── train.txt
    ├── val.txt
    └── test.txt
```

Each split line contains:

```text
<input_filename> <case_id> <region_id>
```

The source NPY arrays are channel-first float32 OSMs with shape `(6, H, W)` and values normalized to approximately `[0, 1]`. Channels 3--5 form the RUV input, channel 3 is the complete R-channel target, and `region_id` is an integer from 0 to 4 or 0 to 24 according to SFEM congfiguration. The data loader resizes arrays to `384 x 352`.

## Training

```bash
python train.py --data-root /path/to/POSDiag --output-dir runs/posdiag_fov25 --epochs 200 --batch-size 20
```

Training uses `Smooth L1 + 0.1 * perceptual loss + 0.1 * histogram EMD`. Batch size 20 requires a high-memory GPU; reduce `--batch-size` when necessary.

## Testing

```bash
python test.py --data-root /path/to/POSDiag --weights /path/to/psafnet_posdiag_fov25_r.pth --split test --output-dir results/posdiag_fov25
```

Predictions are saved as float32 NPY files. The script also reports mean loss against the available ground truth.

## Citation

```bibtex
@article{zhang2025psafnet,
  title   = {Polar Subarea-Aware Fusion Net for Posterior Eyeball Shape Reconstruction},
  author  = {Zhang, Jiaqi and Wu, Xiuzhe and Liu, Jiahui and Zou, Chunyu and Nie, Fengze and Sun, Zicheng and Qi, Xiaojuan and Liu, Jiang},
  journal = {IEEE Transactions on Medical Imaging},
  year    = {2025},
  doi     = {10.1109/TMI.2025.3642381}
}
```

## License

PSAFNet is released under the MIT License. The implementation contains code derived from the MIT-licensed [DPT project](https://github.com/isl-org/DPT); see `THIRD_PARTY_NOTICES.md` and `licenses/DPT-LICENSE`.
