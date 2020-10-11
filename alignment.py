# import numpy as np
# import skimage as sk
# import skimage.io as io
# from skimage.util import img_as_ubyte, img_as_float
# from pathlib import Path
# import matplotlib.pyplot as plt
# import math
# import numpy as np
# import matplotlib.pyplot as plt
# from skimage import util, transform
# from typing import Tuple

# this code is for aligning two colored images with 3 channels
import utils

DEFAULT_HEIGHT = 575
DEFAULT_WIDTH = 547
DEFAULT_EYE_LEN = DEFAULT_WIDTH * 0.35
PAD_MODE='edge'

""" Align Center """
def find_centers(p1, p2) -> Tuple[int, int]:
    cx = int(np.round(np.mean([p1[0], p2[0]])))
    cy = int(np.round(np.mean([p1[1], p2[1]])))
    return cx, cy

#######################
# Aligning One Image #
#######################

def align_img(points: np.ndarray, img: np.ndarray) -> np.ndarray:
    left, right, top, bottom = utils.load_points()
    
    # rescale
    eye_len = np.sqrt((right[1] - left[1]) ** 2 + (right[0] - left[0]) ** 2)
    actual = eye_len
    diff = abs(actual - DEFAULT_EYE_LEN) / DEFAULT_EYE_LEN
    if diff > 0.1:
        scaled = transform.rescale(img, scale=DEFAULT_EYE_LEN / actual, preserve_range=True, multichannel=True, mode=PAD_MODE)
    else:
        scaled = img
    
    # crop
    col_start = cx - DEFAULT_WIDTH * scale // 2
    col_end = cx + DEFAULT_WIDTH * scale // 2
    row_start = cy - DEFAULT_HEIGHT * scale // 2
    row_end = cy + DEFAULT_HEIGHT * scale // 2
    cropped = img[row_start:row_end, col_start:col_end, :]
    
    return cropped
    
def crop_img(img: np.ndarray, pts) -> np.ndarray:
    left, right, top, down = pts
    eye_len = np.sqrt((right[1] - left[1]) ** 2 + (right[0] - left[0]) ** 2)
    
    col_start = cx - DEFAULT_WIDTH * scale // 2
    col_end = cx + DEFAULT_WIDTH * scale // 2
    row_start = cy - DEFAULT_HEIGHT * scale // 2
    row_end = cy + DEFAULT_HEIGHT * scale // 2
    
    print(row_start, row_end, col_start,col_end)
    return img[row_start:row_end, col_start:col_end, :]

#######################
# Aligning Two Images #
#######################

def recenter(im, r, c):
    R, C, _ = im.shape
    rpad = (int)(np.abs(2 * r + 1 - R))
    cpad = (int)(np.abs(2 * c + 1 - C))
    return np.pad(
        im,
        [
            (0 if r > (R - 1) / 2 else rpad, 0 if r < (R - 1) / 2 else rpad),
            (0 if c > (C - 1) / 2 else cpad, 0 if c < (C - 1) / 2 else cpad),
            (0, 0),
        ],
        mode=PAD_MODE,
    )


def align_image_centers(im1, im2, pts):
    p1, p2, p3, p4 = pts
    h1, w1, c1 = im1.shape
    h2, w2, c2 = im2.shape

    cx1, cy1 = find_centers(p1, p2)
    cx2, cy2 = find_centers(p3, p4)

    im1 = recenter(im1, cy1, cx1)
    im2 = recenter(im2, cy2, cx2)
    return im1, im2


""" Scaling and Rotation """


def rescale_images(im1, im2, pts):
    p1, p2, p3, p4 = pts
    print("65:", im1.dtype, im2.dtype)
    # distance formula
    len1 = np.sqrt((p2[1] - p1[1]) ** 2 + (p2[0] - p1[0]) ** 2)
    len2 = np.sqrt((p4[1] - p3[1]) ** 2 + (p4[0] - p3[0]) ** 2)

    dscale = len2 / len1
    if dscale < 1:
        im1 = transform.rescale(im1, dscale, preserve_range=True, multichannel=True, mode=PAD_MODE)
    else:
        im2 = transform.rescale(im2, 1.0 / dscale, preserve_range=True, multichannel=True, mode=PAD_MODE)
#     print("image shapes: ", im1.shape, im2.shape)
    return im1, im2


def rotate_im1(im1, im2, pts):
    p1, p2, p3, p4 = pts
    theta1 = math.atan2(-(p2[1] - p1[1]), (p2[0] - p1[0]))
    theta2 = math.atan2(-(p4[1] - p3[1]), (p4[0] - p3[0]))
    dtheta = theta2 - theta1
    im1 = sktr.rotate(im1, dtheta * 180 / np.pi)
    return im1, dtheta


def match_img_size(im1, im2):
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


""" Master Function """

def align_images(im1, im2, im1_pts, im2_pts):
    h1, w1, c1 = im1.shape
    h2, w2, c2 = im2.shape
    print("113:", im1.dtype, im2.dtype)
    assert c1 == c2
    print("asserted channels")
    pts = [im1_pts[0], im1_pts[1], im2_pts[0], im2_pts[1]]
    im1, im2 = align_image_centers(im1, im2, pts)
    im1, im2 = rescale_images(im1, im2, pts)
#     im1, angle = rotate_im1(im1, im2, pts)
    im1, im2 = match_img_size(im1, im2)
    return im1, im2