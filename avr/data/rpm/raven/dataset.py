import glob
import os
import re
from abc import ABC
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from avr.data.dataset import DatasetSplit, DEFAULT_DATASET_SPLITS
from avr.data.rpm.raven.configuration import RavenConfiguration, ALL_RAVEN_CONFIGURATIONS
from avr.data.rpm.raven.rule_encoder import RavenRuleEncoder, SparseRavenRuleEncoder
from avr.data.transform import resize, select_n_answers, shuffle_answers, to_tensor


class RavenDataset(Dataset, ABC):
    FILEPATH_PATTERN = re.compile(r".*/(\w+)/RAVEN_(\d+)_(\w+).npz")

    def __init__(
            self,
            dataset_root_dir: str = '.',
            configurations: List[RavenConfiguration] = ALL_RAVEN_CONFIGURATIONS,
            splits: List[DatasetSplit] = DEFAULT_DATASET_SPLITS,
            image_size: int = 160,
            do_shuffle_answers: bool = False,
            rule_encoder: RavenRuleEncoder = SparseRavenRuleEncoder(),
            num_answers: int = 2,
    ):
        self.dataset_root_dir = dataset_root_dir
        self.configuration_names = [c.value for c in configurations]
        self.split_names = [s.value for s in splits]
        self.filenames = self._get_filenames()
        self.image_size = image_size
        self.do_shuffle_answers = do_shuffle_answers
        self.rule_encoder = rule_encoder
        self.num_answers = num_answers

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int, torch.Tensor]:
        with np.load(self.filenames[idx]) as data:
            images = data['image']
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

    def __len__(self) -> int:
        return len(self.filenames)

    def _get_filenames(self):
        filenames = []
        for configuration in self.configuration_names:
            filename_pattern = os.path.join(self.dataset_root_dir, configuration, '*.npz')
            filename_pattern = os.path.expanduser(filename_pattern)
            configuration_filenames = glob.glob(filename_pattern)
            for f in configuration_filenames:
                if self._should_contain_filename(f):
                    filenames.append(f)
        return filenames

    def _should_contain_filename(self, filename: str):
        filename = self._split_filename(filename)
        return filename['configuration'] in self.configuration_names and filename['split'] in self.split_names

    def _split_filename(self, filename: str):
        match = re.match(self.FILEPATH_PATTERN, filename)
        return {'configuration': match.group(1), 'id': match.group(2), 'split': match.group(3)}
