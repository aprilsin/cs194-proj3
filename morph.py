# morphing sequence

import math
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import scipy
import skimage as sk
import skimage.io as io
from skimage import util
from skimage.util import img_as_float, img_as_ubyte

import utils

NUM_POINTS = 10
NUM_FRAMES = 10

def set_up_img(img_name):
    img = utils.read_img(img_name)
    morph_pts = utils.get_point(img, NUM_POINTS)

    #     img, morph_pts = align_img(img, morph_pts)
    utils.save_points()

def get_affine_mat(
    initial_tri: list, target: list
) -> np.ndarray:
    
    # A*T = B
    # T = A-1 * B
    # return invA * B
    A = np.matrix([triangle[:,0], triangle[:, 1], [1, 1, 1]])
    B = np.matrix([target[:,0], target[:, 1], [1, 1, 1]])
    return lin.inv(A) * B


def cross_dissolve():
    pass


def morph():
    # for each timeframe
    pass