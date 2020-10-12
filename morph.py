# morphing sequence

import math
from pathlib import Path
from typing import Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import skimage as sk
import skimage.io as io
from scipy.interpolate import interp2d
from scipy.spatial import Delaunay
from skimage import transform
from skimage.util import img_as_float, img_as_ubyte

import utils

NUM_FRAMES = 10
NUM_POINTS = 41
ToImgArray = Union[Path, np.ndarray]

#######################
#    DEFINE SHAPES    #
#######################


def define_shape_vector(img: ToImgArray) -> np.ndarray:
    img = utils.to_img_arr(img)
    points = utils.pick_points(img, NUM_POINTS)
    return points


def avg_points(im1_pts: np.ndarray, im2_pts: np.ndarray, alpha=0.5) -> np.ndarray:
    """
    Compute the (weighted) average points of correspondence
    """
    assert len(im1_pts) == len(im2_pts)
    avg_pts = []
    for i in range(len(im1_pts)):
        avg_pts.append((alpha * im1_pts[i] + (1 - alpha) * im2_pts[i]))
    return np.array(avg_pts)


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


###############################
#   WARP AND CROSS DISSOLVE   #
###############################


def get_triangle_pixels(img, triangle_vertices: np.ndarray):
    """ Returns the coordinates of pixels within triangle for an image """
    rr, cc = sk.draw.polygon(
        triangle_vertices[:, 0], triangle_vertices[:, 1], shape=img.shape
    )
    return rr, cc


def create_triangle_mask(img, triangle_vertices: np.ndarray) -> np.ndarray:
    assert triangle_vertices.shape == (3, 2)  # three points, each with x y coordinates
    """ Returns a mask for one triangle """
    mask = np.zeros_like(img, dtype=np.float)
    rr, cc = get_triangle_pixels(img, triangle_vertices)
    mask[rr, cc] = 1.0
    utils.check_img_type(mask)
    return mask


def get_affine_mat(start: list, target: list) -> np.ndarray:
    # A*T = B
    # T = A-1 * B
    # return invA * B
    A = np.array([start[:, 0], start[:, 1], [1, 1, 1]])
    B = np.array([target[:, 0], target[:, 1], [1, 1, 1]])
    return np.linalg.inv(A) * B


def inverse_affine(img, affine_mat, transformed_coordinates):
    """ Returns the values of pixels from original image. """
    rr, cc = transformed_coordinates


def warp_img(
    img: np.ndarray,
    img_pts: np.ndarray,
    target: np.ndarray,
    triangulation: Delaunay,
):
    for triangle in triangulation.simplices:
        mask = create_triangle_mask(img, img_pts[triangle])
        pixels = inverse_affine()
        pass


def cross_dissolve():
    pass


def compute_middle_object(im1, im2, im1_pts, im2_pts, alpha=0.5):
    mid_pts = avg_points(im1_pts, im2_pts, alpha)
    triangulation = delaunay(mid_pts)
    im1_warped = warp_img(im1, im1_pts, mid_pts, triangulation)
    im2_warped = warp_img(im2, im2_pts, mid_pts, triangulation)
    final_img = cross_dissolve(im1_warped, im2_warped)


def compute_morph_video():
    # for each timeframe
    pass