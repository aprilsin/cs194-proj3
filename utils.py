import argparse
import math
import os
import pickle
import re
import sys
from pathlib import Path
from typing import Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import skimage as sk
import skimage.io as io
from scipy.interpolate import interp2d
from scipy.spatial import Delaunay
from skimage import transform
from skimage.util import img_as_float, img_as_ubyte

from constants import *
from my_types import *

#######################
#      Alignment      #
#######################


def find_centers(p1, p2) -> Tuple[int, int]:
    cx = int(np.round(np.mean([p1[0], p2[0]])))
    cy = int(np.round(np.mean([p1[1], p2[1]])))
    return cx, cy


def align_img(
    img: ToImgArray,
    points: Optional[ToImgArray] = None,
    target_h=DEFAULT_HEIGHT,
    target_w=DEFAULT_WIDTH,
) -> np.ndarray:

    img = to_img_arr(img)
    if points is None:
        print("Please select the eyes for alignment.")
        points = pick_points(img, 2)
    points = to_points(points)
    left_eye, right_eye = points[0], points[1]
    print(left_eye, right_eye)
    # rescale
    actual_eye_len = np.sqrt(
        (right_eye[1] - left_eye[1]) ** 2 + (right_eye[0] - left_eye[0]) ** 2
    )
    diff = abs(actual_eye_len - DEFAULT_EYE_LEN) / DEFAULT_EYE_LEN
    print(actual_eye_len)
    scale = DEFAULT_EYE_LEN / actual_eye_len

    if diff > 0.2:
        assert not np.isnan(img).any()
        scaled = transform.rescale(
            img,
            scale=scale,
            preserve_range=True,
            multichannel=True,
            mode=PAD_MODE,
        ).clip(0, 1)
    else:
        scaled = img
    assert_img_type(scaled)

    # do crop
    scaled_h, scaled_w = scaled.shape[0], scaled.shape[1]
    col_center, row_center = find_centers(left_eye * scale, right_eye * scale)
    row_center += 50

    col_shift = int(target_w // 2)
    row_shift = int(target_h // 2)

    col_start = col_center - col_shift
    col_end = col_center + col_shift
    row_start = row_center - row_shift
    row_end = row_center + row_shift

    rpad_before, rpad_after, cpad_before, cpad_after = 0, 0, 0, 0
    if target_h % 2 != 0:
        rpad_after = 1
    if target_w % 2 != 0:
        cpad_after = 1

    if row_start < 0:
        rpad_before += abs(row_start)
        row_start = 0
        row_end += rpad_before
    if row_end > scaled_h:
        rpad_after += row_end - scaled_h
    if col_start < 0:
        cpad_before += abs(col_start)
        col_start = 0
        col_end += cpad_before
    if col_end > scaled_w:
        cpad_after += col_end - scaled_w
    padded = np.pad(
        scaled,
        pad_width=((rpad_before, rpad_after), (cpad_before, cpad_after), (0, 0)),
        mode=PAD_MODE,
    )
    assert row_start >= 0 and row_end >= 0 and col_start >= 0 and col_end >= 0
    if target_h % 2 != 0:
        row_end += 1
    if target_w % 2 != 0:
        col_end += 1
    aligned = padded[row_start:row_end, col_start:col_end, :]

    assert aligned.shape[0] == DEFAULT_HEIGHT and aligned.shape[1] == DEFAULT_WIDTH
    assert_img_type(aligned)
    return aligned, points


#######################
#   INPUT AND OUPUT   #
#######################


def pick_points(img: ToImgArray, num_pts: int, APPEND_CORNERS=True) -> np.ndarray:
    """
    Returns an array of points for one image with ginput
    """
    img = to_img_arr(img)
    print(f"Please select {num_pts} points in image.")
    plt.imshow(img)
    points = plt.ginput(num_pts, timeout=0)
    plt.close()

    if APPEND_CORNERS:
        points.extend(
            [(0, 0), (0, img.shape[1]), (img.shape[0], 0), (img.shape[0], img.shape[1])]
        )
    print(f"Picked {num_pts} points successfully.")
    return np.array(points)


def save_points(points: np.ndarray, name: os.PathLike) -> None:
    """
    Saves points as Pickle
    """
    name = Path(name)
    pickle_name = name.with_suffix(".pkl")
    pickle.dump(points, open(pickle_name, "wb"))


def load_points(name: os.PathLike) -> np.ndarray:
    """
    Loads an array of points saved as Pickle
    """
    name = Path(name)
    pickle_name = name.with_suffix(".pkl")
    return pickle.load(open(pickle_name, "rb"))


def match_img_size(im1: np.ndarray, im2: np.ndarray):
    # Make images the same size
    h1, w1, c1 = im1.shape
    h2, w2, c2 = im2.shape
    assert c1 == c2

    if h1 < h2:
        im2 = im2[int(np.floor((h2 - h1) / 2.0)) : -int(np.ceil((h2 - h1) / 2.0)), :, :]
    elif h1 > h2:
        im1 = im1[int(np.floor((h1 - h2) / 2.0)) : -int(np.ceil((h1 - h2) / 2.0)), :, :]
    if w1 < w2:
        im2 = im2[:, int(np.floor((w2 - w1) / 2.0)) : -int(np.ceil((w2 - w1) / 2.0)), :]
    elif w1 > w2:
        im1 = im1[:, int(np.floor((w1 - w2) / 2.0)) : -int(np.ceil((w1 - w2) / 2.0)), :]
    assert im1.shape == im2.shape
    return im1, im2
