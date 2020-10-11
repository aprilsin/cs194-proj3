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

#######################
#   TRIANGULATIONS    #
#######################


def avg_points(im1_pts: np.ndarray, im2_pts: np.ndarray, alpha=0.5) -> np.ndarray:
    """
    Compute the (weighted) average points of correspondence
    """
    assert len(im1_pts) == len(im2_pts)
    final_points = []
    for i in range(len(im1_pts)):
        final_points.append((alpha * im1_pts[i] + (1 - alpha) * im2_pts[i]))
    return np.array(final_points)


def delaunay(points):
    return Delaunay(points)


def plot_tri_mesh(img: np.ndarray, points: np.ndarray, triangulation) -> None:
    """
    Displays the triangular mesh of an image
    """
    plt.imshow(img)
    plt.triplot(points[:, 0], points[:, 1], triangulation.simplices)
    plt.plot(points[:, 0], points[:, 1], "o")
    plt.show()


def warp_img(
    img: np.ndarray,
    img_pts: np.ndarray,
    target: np.ndarray,
    triangulation: Delaunay,
):
    for triangle in triangulation.simplices:
        pass


def triangle_mask(img, triangle_vertices: np.ndarray) -> np.ndarray:
    assert triangle_vertices.shape == (3, 2)  # three points, each with x y coordinates
    """ Returns a mask for one triangle """
    mask = np.zeros_like(img, dtype=np.float)
    rr, cc = sk.draw.polygon(
        triangle_vertices[:, 0], triangle_vertices[:, 1], shape=img.shape
    )
    mask[rr, cc] = 1.0
    utils.check_img_type(mask)
    return mask


def main(im1_name, im2_name, num_pts):
    # import image
    im1 = utils.read_img(im1_name)
    im2 = utils.read_img(im2_name)


if __name__ == "__main__":
    im1_name = sys.argv[1]
    im2_name = sys.argv[2]
    #     num_pts = sys.argv[3]
    im1 = utils.setup_img(im1_name)
    im2 = utils.setup_img(im2_name)
    im1_pts = utils.load_points(im1_name)
    im2_pts = utils.load_points(im2_name)
    mid_pts = avg_points(im1_pts, im2_pts)
    triangulation = delaunay(mid_pts)
    for triangle in triangulation.simplices:
        # get points in original image
        # warp the triangle by applying affine transformation
        pass
    # triangle = triangulation.simplices[0]
    # vertices = mid_pts[triangle]
    # plt.imshow(triangle_mask(im1, vertices))
    # plt.imshow(im1)
    plot_tri_mesh(im1, im1_pts, triangulation)
    plt.show()