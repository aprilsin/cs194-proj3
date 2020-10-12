import os
import pickle
from pathlib import Path
from typing import Union, Sequence
import numpy as np
import skimage.io as io
from skimage.util import img_as_float
from constants import *

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


def load_points_from_asf(file_name) -> np.ndarray:
    asf = open(file_name, "r")
    lines_read = asf.readlines()
    num_pts = int(lines_read[9])
    lines = []
    for i in range(16, num_pts + 16):
        lines.append(lines_read[i])

    points = []
    for line in lines:
        data = line.split(" \t")
        points.append((float(data[2]), float(data[3])))
    points.append((0.0, 0.0))
    points.append((1.0, 0.0))
    points.append((0.0, 1.0))
    points.append((1.0, 1.0))
    points = np.array(points)
    points[:, 0] *= POP_HEIGHT
    points[:, 1] *= POP_WIDTH

    # points = np.genfromtxt(
    #     file_name,
    #     dtype="float",
    #     comments="#",
    #     skip_header=1,
    #     skip_footer=1,
    #     usecols=(2, 3),
    # )
    # points.append([0.0, 0.0])
    # points.append([1.0, 0.0])
    # points.append([0.0, 1.0])
    # points.append([1.0, 1.0])
    return points


def to_points(x: ToPoints) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    elif isinstance(x, (str, Path, os.PathLike)):
        x = Path(x)
        if x.suffix in (".pkl", ".p"):
            points = pickle.load(open(x, "rb"))
            assert_points(points)
            return points
        elif x.suffix == ".asf":
            points = load_points_from_asf(x)
            assert_points(points)
            return points
        else:
            raise ValueError(f"Didn't expect type {type(x)}")


def assert_img_type(img: np.ndarray) -> bool:
    """ Check image data type """
    assert img.dtype == "float64", img.dtype
    assert np.max(img) <= 1.0 and np.min(img) >= 0.0, (np.min(img), np.max(img))
    assert np.ndim(img) == 3
    return True


def assert_is_triangle(triangle: np.ndarray) -> bool:
    """ Check image data type """
    assert triangle.shape == (3, 2), triangle.shape
    assert (triangle >= 0).all()
    return True


def assert_points(points: np.ndarray) -> bool:
    assert isinstance(points, np.ndarray)
    assert (points >= 0).all()
    assert points.shape[1] == 2
    return True
