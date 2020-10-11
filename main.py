import argparse
import math
import sys
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import skimage as sk
import skimage.io as io
from scipy.interpolate import interp2d
from scipy.spatial import Delaunay
from skimage import transform
from skimage.util import img_as_float, img_as_ubyte

import morph
import utils

data = Path("input")
data.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    intro = "Project 3 for CS 194-26: Face Morphing\n"
    parser = argparse.ArgumentParser(intro)
    parser.add_argument("im1_path", type=str, help="image 1 for morphing")
    parser.add_argument("im2_path", type=str, help="image 2 for morphing")
    args = parser.parse_args()

    # im2_name = args.im1_path.stem

    im1 = utils.setup_img(args.im1_path)
    # print(type(im1))
    # im2 = utils.setup_img(im2_name)

    # im1_pts = utils.load_points(im1_name)
    # im2_pts = utils.load_points(im2_name)

    # mid_pts = avg_points(im1_pts, im2_pts)
    # triangulation = delaunay(mid_pts)
    # for triangle in triangulation.simplices:
    # get points in original image
    # warp the triangle by applying affine transformation
    # pass
    # triangle = triangulation.simplices[0]
    # vertices = mid_pts[triangle]
    # plt.imshow(triangle_mask(im1, vertices))
    # plt.imshow(im1)
    # plot_tri_mesh(im1, im1_pts, triangulation)
    # plt.show()
