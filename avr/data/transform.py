from typing import Tuple

import cv2 as cv
import numpy as np
import torch


def to_tensor(image: np.array) -> torch.Tensor:
    image = image.astype("float32") / 255.0
    return torch.tensor(image)


def shuffle_answers(panels: np.array, target: int) -> Tuple[np.array, int]:
    indices = list(range(len(panels)))
    np.random.shuffle(indices)
    return panels[indices], indices.index(target)


def select_n_answers(answers: np.array, target: int, n: int) -> Tuple[np.array, int]:
    n = min(n, len(answers))
    indices = list(range(len(answers)))
    indices = np.delete(indices, target)
    np.random.shuffle(indices)
    indices = np.concatenate([indices[: n - 1], [target]])
    np.random.shuffle(indices)
    return answers[indices], list(indices).index(target)


def resize(image: np.array, height: int, width: int) -> np.array:
    return cv.resize(np.asarray(image, dtype=float), dsize=(width, height), interpolation=cv.INTER_AREA)
