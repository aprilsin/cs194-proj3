# morphing sequence

import math
from pathlib import Path
from typing import Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import skimage as sk
import skimage.io as io
from scipy.interpolate import RectBivariateSpline, interp2d
from scipy.spatial import Delaunay
from skimage import transform
from skimage.util import img_as_float, img_as_ubyte

import utils
from constants import *
from my_types import *

#######################
#    DEFINE SHAPES    #
#######################


def define_shape_vector(img: ToImgArray) -> np.ndarray:
    img = utils.to_img_arr(img)
    points = utils.pick_points(img, NUM_POINTS)
    return points


def weighted_avg(im1_pts: np.ndarray, im2_pts: np.ndarray, alpha=0.5) -> np.ndarray:
    """
    Compute the (weighted) average points of correspondence
    """
    assert len(im1_pts) == len(im2_pts)
    return alpha * im1_pts + (1 - alpha) * im2_pts


def delaunay(points):
    return Delaunay(points)


def points_from_delaunay(points, triangulation):
    return points[triangulation.simplices]


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


def get_affine_mat(start: Triangle, target: Triangle) -> np.ndarray:
    # A*T = B
    # T = A-1 * B
    # return invA * B
    assert_is_triangle(start)
    assert_is_triangle(target)
    A = np.vstack((start[:, 0], start[:, 1], [1, 1, 1]))
    B = np.vstack((target[:, 0], target[:, 1], [1, 1, 1]))
    return np.linalg.inv(A) * B


def get_triangle_pixels(
    triangle_vertices: np.ndarray, shape=(DEFAULT_HEIGHT, DEFAULT_WIDTH)
):
    """ Returns the coordinates of pixels within triangle for an image """
    rr, cc = sk.draw.polygon(
        triangle_vertices[:, 0], triangle_vertices[:, 1], shape=shape
    )
    return rr, cc


def create_triangle_mask(
    triangle_vertices: np.ndarray, shape=(DEFAULT_HEIGHT, DEFAULT_WIDTH)
) -> np.ndarray:
    # assert triangle_vertices.shape == (3, 2)  # three points, each with x y coordinates
    """ Returns a mask for one triangle """
    mask = np.zeros(shape, dtype=np.float)
    rr, cc = get_triangle_pixels(triangle_vertices)
    mask[rr, cc] = 1.0
    utils.assert_img_type(mask)
    return mask


def inverse_affine(img, img_triangle_vertices, target_triangle_vertices):
    """ Returns the coordinates of pixels from original image. """
    assert_is_triangle(img_triangle_vertices)
    assert_is_triangle(target_triangle_vertices)

    affine_mat = get_affine_mat(img_triangle_vertices, target_triangle_vertices)
    # inverse of affine is just the transpose
    inverse = np.linalg.inv(affine_mat)
    rr, cc = get_triangle_pixels(target_triangle_vertices)
    return rr, cc


def warp_img(
    img: np.ndarray,
    img_pts: np.ndarray,
    target_pts: np.ndarray,
    triangulation: Delaunay,
):
    warped = np.zeros_like(img)
    # num_triangles, _, _ = triangulation.simplices
    for triangle in points_from_delaunay(target_pts, triangulation.simplices):
        print("simplices ", triangle)

        target_vertices = points_from_delaunay(target_pts, triangulation)
        print("vertices ", target_vertices.shape)
        img_vertices = points_from_delaunay(img_pts, triangulation)
        target_mask = create_triangle_mask(target_vertices, img.shape)

        # do inverse warping
        rr, cc = inverse_affine(img, img_vertices, target_vertices)
        interp_func = RectBivariateSpline(rr, cc, img)
        warped[rr, cc] = interp_func(rr, cc, grid=False)
    return warped


def cross_dissolve(warped_im1, warped_im2, alpha=0.5):
    utils.assert_img_type(warped_im1)
    utils.assert_img_type(warped_im2)

    result = np.zeros_like(warped_im1)
    for channel in range(utils.NUM_CHANNELS):
        result[:, :, channel] = weighted_avg(warped_im1, warped_im2)
    return result


def compute_middle_object(im1, im2, im1_pts, im2_pts, alpha=0.5):
    mid_pts = weighted_avg(im1_pts, im2_pts, alpha)
    triangulation = delaunay(mid_pts)
    im1_warped = warp_img(im1, im1_pts, mid_pts, triangulation)
    im2_warped = warp_img(im2, im2_pts, mid_pts, triangulation)
    # final_img = cross_dissolve(im1_warped, im2_warped)
    return im1_warped


import copy
import time

from matplotlib import animation


def compute_morph_video(
    im1, im2, im1_pt2, im2_pts, out_path, num_frames=NUM_FRAMES, boomerang=True
):
    # for each timeframe
    frames = []
    fig = plt.figure(frameon=False)
    ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
    ax.set_axis_off()
    fig.add_axes(ax)
    for i in range(num_frames):
        alpha = 1 / num_frames * i
        strt = time.time()
        curr_frame = compute_middle_object(im1, im2, im1_pt2, im2_pts, alpha)
        print("Frame morph time:", time.time() - strt)
        frames.append()
        # im = plt.imshow(curr_frame)
        # mov.append([im])

    if boomerang:
        reversed_frames = copy.copy(frames)
        reversed_frames.reverse()
        frames += reversed_frames

    # create video from frames
    h, w, c = im1.shape
    ratio = w / h
    fig.set_size_inches(int(h / 50), int(w / 50), 10)
    ani = animation.ArtistAnimation(
        fig, frames, interval=1000, blit=True, repeat_delay=0
    )
    Writer = animation.writers["ffmpeg"]
    writer = Writer(fps=10, metadata=dict(artist="Me"), bitrate=1800)
    ani.save(out_path, writer=writer)
    return frames