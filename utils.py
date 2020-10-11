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

DEFAULT_HEIGHT = 575
DEFAULT_WIDTH = 547
DEFAULT_EYE_LEN = DEFAULT_WIDTH * 0.25
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
        print(0,img.min(),img.max())
        scaled = transform.rescale(
            img,
            scale=scale,
            preserve_range=True,
            multichannel=True,
            mode=PAD_MODE,
        ).clip(0, 1)
        print(1, scaled.min(),scaled.max())
    else:
        scaled = img
        print(2, scaled.min,scaled.max())
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
    cropped = padded[row_start:row_end, col_start:col_end, :]
    assert cropped.shape[0] == DEFAULT_HEIGHT and cropped.shape[1] == DEFAULT_WIDTH
    assert_img_type(cropped)
    return cropped


def assert_img_type(img) -> None:
    """ Check image data type """
    assert img.dtype == "float64", img.dtype
    assert np.max(img) <= 1.0 and np.min(img) >= 0.0, (np.min(img), np.max(img))


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
    print(f'range as ubyte: {img.min(), img.max()}')
    img = img_as_float(img)
    print(f'range as float: {img.min(), img.max()}')
    assert_img_type(img)
    return img


def align_img(img_name: str):
    im_arr = read_img(img_name)
    pickle_name = re.split("\.", img_name)[0] + "_align" + ".p"
    if path.exists(pickle_name):
        points = pickle.load(open(pickle_name, "rb"))
    else:
        print("Please select the eyes for alignment.")
        points = get_points(im_arr, 2)
        pickle_name = re.split("\.", img_name)[0] + "_align" + ".p"
        save_points(pickle_name, points)
    aligned_img_name = re.split("\.", img_name)[0] + "_align" + ".jpg"
    print('max', im_arr.max())
    aligned_im_arr = align(im_arr, points)
    assert_img_type(aligned_im_arr)
    io.imsave(
        aligned_img_name,
        img_as_ubyte(aligned_im_arr),
        format="jpg",
    )
    assert_img_type(aligned_im_arr)
    return aligned_im_arr
    
def match_img_size(im1:np.ndarray, im2:np.ndarray):
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
    print("image shapes: ", im1.shape, im2.shape)
    assert im1.shape == im2.shape
    return im1, im2


def shape_vector_exist(image_name):
    return path.exists(re.split("\.", image_name)[0] + ".p")


def define_shape_vector(img_name: str):
    im_arr = read_img(img_name)
    pickle_name = re.split("\.", img_name)[0] + ".p"
    if not path.exists(pickle_name):
        points = utils.get_points(img_name, NUM_POINTS)
        save_points(pickle_name, points)