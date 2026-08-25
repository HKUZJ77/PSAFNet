"""POSDiag 25%-FOV data loader used by the public PSAFNet scripts."""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


NETWORK_HEIGHT = 384
NETWORK_WIDTH = 352
RUV_CHANNELS = (3, 4, 5)
R_CHANNEL = (3,)


@dataclass(frozen=True)
class _Record:
    input_name: str
    case_id: str
    region_index: int


def _load_osm(path, channels):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError("OSM file not found: {}".format(path))

    array = np.load(str(path), allow_pickle=False)
    if array.ndim != 3:
        raise ValueError("Expected a channel-first 3D OSM at {}, got {}".format(path, array.shape))
    if max(channels) >= array.shape[0]:
        raise ValueError("OSM at {} does not contain channels {}".format(path, channels))

    array = array[list(channels)].astype(np.float32, copy=False)
    array = np.transpose(array, (1, 2, 0))
    array = cv2.resize(
        array,
        (NETWORK_WIDTH, NETWORK_HEIGHT),
        interpolation=cv2.INTER_AREA,
    )
    if array.ndim == 2:
        array = array[:, :, np.newaxis]
    array = np.transpose(array, (2, 0, 1))
    return torch.from_numpy(np.ascontiguousarray(array)).float()


class POSDiagDataset(Dataset):
    """Load RUV inputs and the complete R-channel target for 25%-FOV POSDiag."""

    def __init__(
        self,
        data_root,
        split="train",
        split_file=None,
        template_path=None,
        max_samples=None,
    ):
        self.data_root = Path(data_root)
        split = {"valid": "val", "validation": "val"}.get(split, split)
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be one of: train, val, test")

        if split_file is None:
            split_file = self.data_root / "txt" / "subarea_txt" / "{}.txt".format(split)
        else:
            split_file = Path(split_file)
        if not split_file.is_file():
            raise FileNotFoundError("Split file not found: {}".format(split_file))

        self.records = []
        with split_file.open("r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                fields = raw_line.split()
                if not fields:
                    continue
                if len(fields) != 3:
                    raise ValueError(
                        "{}:{} must contain input_name, case_id, and region_index".format(
                            split_file, line_number
                        )
                    )
                region_index = int(fields[2])
                if not 0 <= region_index < 5:
                    raise ValueError(
                        "{}:{} has invalid 25%-FOV region index {}".format(
                            split_file, line_number, region_index
                        )
                    )
                self.records.append(_Record(fields[0], fields[1], region_index))

        if max_samples is not None:
            self.records = self.records[:max_samples]
        if not self.records:
            raise ValueError("No samples were found in {}".format(split_file))

        if template_path is None:
            template_path = self.data_root / "template" / "template12_mcimg.npy"
        self.template = _load_osm(template_path, RUV_CHANNELS)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        input_path = self.data_root / "subarea_input" / record.input_name
        normalized_case_id = str(int(record.case_id))
        target_path = (
            self.data_root
            / "global_mc_img"
            / "{}_glob_mcimg.npy".format(normalized_case_id)
        )

        return {
            "image": _load_osm(input_path, RUV_CHANNELS),
            "target": _load_osm(target_path, R_CHANNEL),
            "template": self.template,
            "region_index": torch.tensor(record.region_index, dtype=torch.long),
            "case_id": normalized_case_id,
            "input_name": record.input_name,
        }
