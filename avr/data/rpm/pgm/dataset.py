import os
import re
from abc import ABC
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from avr.data.dataset import DatasetSplit, DEFAULT_DATASET_SPLITS
from avr.data.rpm.pgm.rule_encoder import PgmRuleEncoder, SparsePgmRuleEncoder
from avr.data.transform import resize, select_n_answers, shuffle_answers, to_tensor


class PgmDataset(Dataset, ABC):
    FILEPATH_PATTERN = re.compile(r"PGM_([\w.]+)_(\w+)_(\d+).npz")

    def __init__(
            self,
            dataset_root_dir: str = '.',
            splits: List[DatasetSplit] = DEFAULT_DATASET_SPLITS,
            image_size: int = 160,
            do_shuffle_answers: bool = False,
            rule_encoder: PgmRuleEncoder = SparsePgmRuleEncoder(),
            num_answers: int = 2
    ):
        self.dataset_root_dir = dataset_root_dir
        self.split_names = [s.value for s in splits]
        self.filenames = self._list_filenames(dataset_root_dir, splits)
        self.image_size = image_size
        self.do_shuffle_answers = do_shuffle_answers
        self.rule_encoder = rule_encoder
        self.num_answers = num_answers

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int, torch.Tensor]:
        with np.load(self.filenames[idx], allow_pickle=True) as data:
            images = data['image']
            h, w, c = images.shape
            images = np.ascontiguousarray(images.reshape(c, h, w))
            images = np.stack([
                resize(image, self.image_size, self.image_size)
                for image in images
            ])
            context, answers = np.split(images, 2)

            target = data['target']
            if self.do_shuffle_answers:
                answers, target = shuffle_answers(answers, target)
            answers, target = select_n_answers(answers, target, self.num_answers)

            context = to_tensor(context).unsqueeze(dim=1)
            answers = to_tensor(answers).unsqueeze(dim=1)
            rules = self.rule_encoder.encode(data)
            return context, answers, target, rules

    def _list_filenames(self, data_dir: str, splits: List[DatasetSplit]):
        split_names = [s.value for s in splits]
        filenames = []
        for f in os.listdir(data_dir):
            split = self._split_filename(f)
            if split is not None and split["dataset_split"] in split_names:
                filenames.append(os.path.join(data_dir, f))
        return filenames

    def _split_filename(self, filename: str):
        match = re.match(self.FILEPATH_PATTERN, filename)
        if match:
            return {
                "generalisation_split": match.group(1),
                "dataset_split": match.group(2),
                "id": match.group(3),
            }
        else:
            return None
