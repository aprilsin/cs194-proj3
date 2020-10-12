import os
from pathlib import Path
from typing import Union
import pickle
import skimage.io as io
from skimage.util import img_as_float

import numpy as np

ToImgArray = Union[os.PathLike, np.ndarray]
ZeroOneFloatArray = np.ndarray
UbyteArray = np.ndarray

ToPoints = Union[os.PathLike, np.ndarray]
Triangle = np.ndarray


def to_img_arr(x: ToImgArray) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return img_as_float(x).clip(0, 1)
    elif isinstance(x, (str, Path, os.PathLike)):
        x = Path(x)
        if x.suffix in (".jpeg", ".jpg"):
            img = io.imread(x)
            img = img_as_float(img)
            assert_img_type(img)
            return img
        else:
            raise ValueError(f"Didn't expect type {type(x)}")


def to_points(x: ToPoints) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    elif isinstance(x, (str, Path, os.PathLike)):
        x = Path(x)
        if x.suffix in (".pkl", ".p"):
            return pickle.load(open(x, "rb"))
        else:
            raise ValueError(f"Didn't expect type {type(x)}")


def assert_img_type(img: np.ndarray) -> None:
    """ Check image data type """
    assert img.dtype == "float64", img.dtype
    assert np.max(img) <= 1.0 and np.min(img) >= 0.0, (np.min(img), np.max(img))
    assert np.ndim(img) == 3


def assert_is_triangle(triangle: np.ndarray) -> None:
    """ Check image data type """
    assert triangle.shape == (3, 2), triangle.shape