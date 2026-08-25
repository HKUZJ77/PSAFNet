# GitHub Release asset

Upload the following inference checkpoint as a GitHub Release asset. The `.pth`
file is intentionally excluded from Git history.

| File | Dataset | Input | Target | Embedding grid | Backbone |
|---|---|---|---|---|---|
| `psafnet_posdiag_fov25_r.pth` | POSDiag | 25%-FOV RUV OSM | Complete R-channel OSM | 24 x 22 | ResNet-50/ViT hybrid |

- Size: 952,888,084 bytes (0.887 GiB)
- SHA256: `A990599DEBB4DF0D62C35231D19EDD4D876EFC2059A2EEF728873A3B32DA4BA4`

The checkpoint contains model parameters and release metadata only. It does not
contain optimizer states, scheduler states, local paths, or dataset files.
