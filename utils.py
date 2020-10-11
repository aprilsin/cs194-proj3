import math
import pickle
import re
from os import path
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import skimage as sk
import skimage.io as io
from skimage import transform, util
from skimage.util import img_as_float, img_as_ubyte

data = Path("input")
data.mkdir(parents=True, exist_ok=True)

DEFAULT_HEIGHT = 800
DEFAULT_WIDTH = 600
DEFAULT_EYE_LEN = DEFAULT_WIDTH * 0.25
PAD_MODE = "edge"

NUM_POINTS = 11

#######################
#   INPUT AND OUPUT   #
#######################


def get_points(img: np.ndarray, num_pts: int, APPEND_CORNERS=True) -> np.ndarray:
    """
    Returns an array of points for one image with ginput
    """
    print(f"Please select {num_pts} points in image.")

    plt.imshow(img)
    points = plt.ginput(num_pts)
    plt.close()

    if APPEND_CORNERS:
        points.append((0, 0))
        points.append((0, img.shape[1]))
        points.append((img.shape[0], 0))
        points.append((img.shape[0], img.shape[1]))

    return np.array(points)


def save_points(img_name, points) -> None:
    """
    Saves points as Pickle
    """
    pickle_name = re.split("\.", img_name)[0] + ".p"
    pickle.dump(points, open(pickle_name, "wb"))


def load_points(img_name) -> np.ndarray:
    """
    Loads an array of points saved as Pickle
    """
    pickle_name = re.split("\.", img_name)[0] + ".p"
    assert path.exists(pickle_name)
    return pickle.load(open(pickle_name, "rb"))


def read_img(img_name) -> np.ndarray:
    """
    Input Image
    """
    im_path = img_name
    img = io.imread(im_path)
    img = img_as_float(img)
    assert img.dtype == "float64"
    return img


def check_img_type(img) -> None:
    """ Check image data type """
    assert img.dtype == "float64"
    print(np.max(img), np.min(img))
    assert np.max(img) <= 1.0 and np.min(img) >= 0.0


#######################
#      Alignment      #
#######################


def find_centers(p1, p2):  # -> Tuple[int, int]:
    cx = int(np.round(np.mean([p1[0], p2[0]])))
    cy = int(np.round(np.mean([p1[1], p2[1]])))
    return cx, cy


def align(
    img: np.ndarray, points: np.ndarray, target_h=DEFAULT_HEIGHT, target_w=DEFAULT_WIDTH
) -> np.ndarray:

    left_eye, right_eye, top, bottom = points[0], points[1], points[2], points[3]

    # rescale
    actual_eye_len = np.sqrt(
        (right_eye[1] - left_eye[1]) ** 2 + (right_eye[0] - left_eye[0]) ** 2
    )
    diff = abs(actual_eye_len - DEFAULT_EYE_LEN) / DEFAULT_EYE_LEN
    scale = DEFAULT_EYE_LEN / actual_eye_len
    if diff > 0.1:
        scaled = transform.rescale(
            img,
            scale=scale,
            preserve_range=True,
            multichannel=True,
            mode=PAD_MODE,
        )
    else:
        scaled = img

    scaled_h, scaled_w = scaled.shape[0], scaled.shape[1]
    # do crop
    col_center, row_center = find_centers(left_eye * scale, right_eye * scale)
    row_center += 50

    col_shift = int(target_w // 2)
    row_shift = int(target_h // 2)

    col_start = col_center - col_shift
    col_end = col_center + col_shift
    row_start = row_center - row_shift
    row_end = row_center + row_shift

    rpad_before, rpad_after, cpad_before, cpad_after = 0, 0, 0, 0
    if row_start < 0:
        rpad_before = abs(row_start)
        row_start = 0
        row_end += rpad_before
    if row_end > scaled_h:
        rpad_after = row_end - scaled_h
    if col_start < 0:
        cpad_before = abs(col_start)
        col_start = 0
        col_end += cpad_before
    if col_end > scaled_w:
        cpad_after = col_end - scaled_w

    padded = np.pad(
        scaled,
        pad_width=((rpad_before, rpad_after), (cpad_before, cpad_after), (0, 0)),
        mode=PAD_MODE,
    )

    assert row_start >= 0 and row_end >= 0 and col_start >= 0 and col_end >= 0
    cropped = padded[row_start:row_end, col_start:col_end, :]
    assert cropped.shape[0] == DEFAULT_HEIGHT and cropped.shape[1] == DEFAULT_WIDTH
    check_img_type(cropped)
    return cropped


#######################
#    SET UP IMAGE     #
#######################


def setup_img(img_name):
    im_arr = read_img(img_name)
    pickle_name = re.split("\.", img_name)[0] + ".p"
    if path.exists(pickle_name):
        points = load_points(img_name)
    else:
        points = get_points(im_arr, NUM_POINTS)
        save_points(img_name, points)
    return align(im_arr, points)