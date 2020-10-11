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


def check(img) -> None:
    """ Check image data type """
    assert img.dtype == "float64"
    assert np.max(img) == 1.0 and np.min(img) == 0.0


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
    print("scaled shape ", scaled.shape)
    # do crop
    col_center, row_center = find_centers(left_eye * scale, right_eye * scale)

    col_shift = int(target_w // 2)  # + DEFAULT_HEIGHT * 0.12
    row_shift = int(target_h // 2)  # + DEFAULT_WIDTH * 0.1

    col_start = col_center - col_shift
    col_end = col_center + col_shift
    row_start = row_center - row_shift
    row_end = row_center + row_shift

    print("center ", col_center, row_center)
    print("shift ", col_shift, row_shift)
    print("before pad index ", row_start, row_end, col_start, col_end)

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

    print("padded shape ", padded.shape)
    print("index ", row_start, row_end, col_start, col_end)
    assert row_start >= 0 and row_end >= 0 and col_start >= 0 and col_end >= 0
    cropped = padded[row_start:row_end, col_start:col_end, :]
    # check(scaled)
    print("cropped shape ", cropped.shape)
    return scaled

    # def crop_img(img: np.ndarray, pts) -> np.ndarray:
    #     left, right, top, down = pts
    #     eye_len = np.sqrt((right[1] - left[1]) ** 2 + (right[0] - left[0]) ** 2)

    #     col_start = cx - DEFAULT_WIDTH * scale // 2
    #     col_end = cx + DEFAULT_WIDTH * scale // 2
    #     row_start = cy - DEFAULT_HEIGHT * scale // 2
    #     row_end = cy + DEFAULT_HEIGHT * scale // 2

    #     print(row_start, row_end, col_start, col_end)
    #     return img[row_start:row_end, col_start:col_end, :]

    # def recenter(im, r, c):
    #     R, C, _ = im.shape
    #     rpad = (int)(np.abs(2 * r + 1 - R))
    #     cpad = (int)(np.abs(2 * c + 1 - C))
    #     return np.pad(
    #         im,
    #         [
    #             (0 if r > (R - 1) / 2 else rpad, 0 if r < (R - 1) / 2 else rpad),
    #             (0 if c > (C - 1) / 2 else cpad, 0 if c < (C - 1) / 2 else cpad),
    #             (0, 0),
    #         ],
    #         mode=PAD_MODE,
    #     )

    # def align_image_centers(im1, im2, pts):
    #     p1, p2, p3, p4 = pts
    #     h1, w1, c1 = im1.shape
    #     h2, w2, c2 = im2.shape

    #     cx1, cy1 = find_centers(p1, p2)
    #     cx2, cy2 = find_centers(p3, p4)

    #     im1 = recenter(im1, cy1, cx1)
    #     im2 = recenter(im2, cy2, cx2)
    #     return im1, im2

    # def rescale_images(im1, im2, pts):
    #     p1, p2, p3, p4 = pts
    #     print("65:", im1.dtype, im2.dtype)
    #     # distance formula
    #     len1 = np.sqrt((p2[1] - p1[1]) ** 2 + (p2[0] - p1[0]) ** 2)
    #     len2 = np.sqrt((p4[1] - p3[1]) ** 2 + (p4[0] - p3[0]) ** 2)

    #     dscale = len2 / len1
    #     if dscale < 1:
    #         im1 = transform.rescale(
    #             im1, dscale, preserve_range=True, multichannel=True, mode=PAD_MODE
    #         )
    #     else:
    #         im2 = transform.rescale(
    #             im2, 1.0 / dscale, preserve_range=True, multichannel=True, mode=PAD_MODE
    #         )
    #     #     print("image shapes: ", im1.shape, im2.shape)
    #     return im1, im2

    # def match_img_size(im1, im2):
    #     # Make images the same size
    #     h1, w1, c1 = im1.shape
    #     h2, w2, c2 = im2.shape
    #     assert c1 == c2

    #     if h1 < h2:
    #         im2 = im2[int(np.floor((h2 - h1) / 2.0)) : -int(np.ceil((h2 - h1) / 2.0)), :, :]
    #     elif h1 > h2:
    #         im1 = im1[int(np.floor((h1 - h2) / 2.0)) : -int(np.ceil((h1 - h2) / 2.0)), :, :]
    #     if w1 < w2:
    #         im2 = im2[:, int(np.floor((w2 - w1) / 2.0)) : -int(np.ceil((w2 - w1) / 2.0)), :]
    #     elif w1 > w2:
    #         im1 = im1[:, int(np.floor((w1 - w2) / 2.0)) : -int(np.ceil((w1 - w2) / 2.0)), :]
    #     print("image shapes: ", im1.shape, im2.shape)
    #     assert im1.shape == im2.shape
    #     return im1, im2

    # def align_images(im1, im2, im1_pts, im2_pts):
    #     """ Align two images given two points for alignment """
    #     h1, w1, c1 = im1.shape
    #     h2, w2, c2 = im2.shape
    #     print("113:", im1.dtype, im2.dtype)
    #     assert c1 == c2
    #     print("asserted channels")
    #     pts = [im1_pts[0], im1_pts[1], im2_pts[0], im2_pts[1]]
    #     im1, im2 = align_image_centers(im1, im2, pts)
    #     im1, im2 = rescale_images(im1, im2, pts)
    #     im1, im2 = match_img_size(im1, im2)
    #     return im1, im2


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