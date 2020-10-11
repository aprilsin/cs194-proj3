import argparse
import math
import pickle
import re
import sys
from pathlib import Path
from os import path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import skimage as sk
import skimage.io as io
from scipy.interpolate import interp2d
from scipy.spatial import Delaunay
from skimage import transform
from skimage.util import img_as_float, img_as_ubyte

DEFAULT_HEIGHT = 700
DEFAULT_WIDTH = 600
DEFAULT_EYE_LEN = DEFAULT_WIDTH * 0.35
PAD_MODE = "edge"

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

    left_eye, right_eye = points[0], points[1]

    # rescale
    actual_eye_len = np.sqrt(
        (right_eye[1] - left_eye[1]) ** 2 + (right_eye[0] - left_eye[0]) ** 2
    )
    diff = abs(actual_eye_len - DEFAULT_EYE_LEN) / DEFAULT_EYE_LEN
    scale = DEFAULT_EYE_LEN / actual_eye_len
    if diff > 0.12:
        scaled = transform.rescale(
            img,
            scale=scale,
            preserve_range=True,
            multichannel=True,
            mode=PAD_MODE,
        )
    else:
        scaled = img

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
    cropped = padded[row_start:row_end, col_start:col_end, :]
    assert cropped.shape[0] == DEFAULT_HEIGHT and cropped.shape[1] == DEFAULT_WIDTH
    return cropped


def check_img_type(img) -> None:
    """ Check image data type """
    assert img.dtype == "float64"
    assert np.max(img) <= 1.0 and np.min(img) >= 0.0


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


def load_points(img_name, for_alignment=False) -> np.ndarray:
    """
    Loads an array of points saved as Pickle
    """
    if for_alignment:
        # pickle_name = PICKLE_DIR / Path(img_name + "_align" + ".p")
        pickle_name = re.split("\.", img_name)[0] + "_align" + ".p"
    else:
        # pickle_name = PICKLE_DIR / (img_name + ".p")
        pickle_name = re.split("\.", img_name)[0] + ".p"
    assert path.exists(pickle_name)
    return pickle.load(open(pickle_name, "rb"))


def read_img(img_name) -> np.ndarray:
    """
    Input Image
    """
    # im_path = DATA_DIR / (img_name + ".jpg")
    # im_path = DATA_DIR + img_name + ".jpg"
    img = io.imread(img_name)
    img = img_as_float(img)
    assert img.dtype == "float64"
    return img


def align_img(img_name: str):
    im_arr = read_img(img_name)
#     pickle_name = re.split("\.", img_name)[0] + "_align" + ".p"
#     if path.exists(pickle_name):
#         points = pickle.load(open(pickle_name, "rb"))
#     else:
    print("Please select the eyes for alignment.")
    points = get_points(im_arr, 2)
#         pickle_name = re.split("\.", img_name)[0] + "_align" + ".p"
#         save_points(pickle_name, points)
    aligned_img_name = re.split("\.", img_name)[0] + "_align" + ".jpg"
    aligned_im_arr = align(im_arr, points)
    print(aligned_im_arr.dtype)
    io.imsave(
        aligned_img_name,
        img_as_ubyte(aligned_im_arr),
        format="jpg",
    )


def shape_vector_exist(image_name):
    return path.exists(re.split("\.", image_name)[0] + ".p")


def define_shape_vector(img_name: str):
    im_arr = read_img(img_name)
    pickle_name = re.split("\.", img_name)[0] + ".p"
    if not path.exists(pickle_name):
        points = utils.get_points(img_name, NUM_POINTS)
        save_points(pickle_name, points)