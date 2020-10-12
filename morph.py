# morphing sequence

import math
from pathlib import Path
from typing import Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import skimage as sk
import skimage.io as io
from scipy import interpolate
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


def weighted_avg(im1_pts: np.ndarray, im2_pts: np.ndarray, alpha) -> np.ndarray:
    """
    Compute the (weighted) average points of correspondence
    """
    assert len(im1_pts) == len(im2_pts), (len(im1_pts), len(im2_pts))
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
    # B = T * A
    # T = B * A^-1
    assert_is_triangle(start)
    assert_is_triangle(target)
    A = np.vstack((start[:, 0], start[:, 1], [1, 1, 1]))
    B = np.vstack((target[:, 0], target[:, 1], [1, 1, 1]))
    return B @ np.linalg.inv(A)


def get_triangle_pixels(
    triangle_vertices: np.ndarray, shape=(DEFAULT_HEIGHT, DEFAULT_WIDTH)
):
    """ Returns the coordinates of pixels within triangle for an image """
    assert_is_triangle(triangle_vertices)
    rr, cc = sk.draw.polygon(
        triangle_vertices[:, 0], triangle_vertices[:, 1], shape=shape
    )
    return rr, cc


def create_triangle_mask(
    triangle_vertices: np.ndarray, shape=(DEFAULT_HEIGHT, DEFAULT_WIDTH)
) -> np.ndarray:
    # assert triangle_vertices.shape == (3, 2)  # three points, each with x y coordinates
    """ Returns a mask for one triangle """
    assert_is_triangle(triangle_vertices)

    mask = np.zeros(shape, dtype=np.float)
    rr, cc = get_triangle_pixels(triangle_vertices)
    mask[rr, cc] = 1.0
    utils.assert_img_type(mask)
    return mask


def inverse_affine(img, img_triangle_vertices, target_triangle_vertices):
    """ Returns the coordinates of pixels from original image. """
    assert_img_type(img)
    assert_is_triangle(img_triangle_vertices)
    assert_is_triangle(target_triangle_vertices)

    affine_mat = get_affine_mat(img_triangle_vertices, target_triangle_vertices)
    # inverse of affine is just the transpose
    inverse = np.linalg.inv(affine_mat)
    rr, cc = get_triangle_pixels(target_triangle_vertices)
    target_points = np.vstack([rr, cc, np.ones(len(rr))])
    src_points = inverse @ target_points
    return src_points[0, :], src_points[1, :]


def warp_img(
    img: np.ndarray,
    img_pts: np.ndarray,
    target_pts: np.ndarray,
    triangulation: Delaunay,
)->np.ndarray:
    assert_img_type(img)
    assert_points(img_pts)
    assert_points(target_pts)

    warped = np.zeros_like(img)
    # num_triangles, _, _ = triangulation.simplices
    for simplex in triangulation.simplices:

        target_vertices = target_pts[simplex]
        img_vertices = img_pts[simplex]
        assert_is_triangle(target_vertices)
        assert_is_triangle(img_vertices)

        # do inverse warping
        target_rr, target_cc = get_triangle_pixels(target_vertices, img.shape)
        src_rr, src_cc = inverse_affine(img, img_vertices, target_vertices)
        src_rr, src_cc = np.int32(np.floor(src_rr) - 1), np.int32(np.floor(src_cc) - 10)
        print(max(target_rr), max(target_cc))
        print(max(src_rr), max(src_cc))
        warped[target_rr, target_cc] = img[src_rr, src_cc]
    return warped


def cross_dissolve(warped_im1, warped_im2, alpha):
    assert_img_type(warped_im1)
    assert_img_type(warped_im2)
    return weighted_avg(warped_im1, warped_im2, alpha=alpha)


def compute_middle_object(
    im1: ToImgArray, im2: ToImgArray, im1_pts: ToPoints, im2_pts: ToPoints, alpha
):
    im1 = to_img_arr(im1)
    im2 = to_img_arr(im2)
    im1_pts = to_points(im1_pts)
    im2_pts = to_points(im2_pts)

    mid_pts = weighted_avg(im1_pts, im2_pts, alpha=alpha)
    triangulation = delaunay(mid_pts)
    im1_warped = warp_img(im1, im1_pts, mid_pts, triangulation)
    im2_warped = warp_img(im2, im2_pts, mid_pts, triangulation)
    middle_img = cross_dissolve(im1_warped, im2_warped, alpha=alpha)
    return middle_img, triangulation


import copy
import time

from matplotlib import animation


def compute_morph_video(
    im1, im2, im1_pts, im2_pts, out_path, num_frames=NUM_FRAMES, fps=10, boomerang=True
):
    im1 = to_img_arr(im1)
    im2 = to_img_arr(im2)
    im1_pts = to_points(im1_pts)
    im2_pts = to_points(im2_pts)

    # for each timeframe
    frames = []
    fig = plt.figure(frameon=False)
    ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
    ax.set_axis_off()
    fig.add_axes(ax)
    alphas = np.linspace(0, 1, num_frames)

    for i, alpha in enumerate(alphas, start=1):
        start = time.time()
        curr_frame = compute_middle_object(im1, im2, im1_pts, im2_pts, 1 - alpha)
        print(f"Frame {i} morph time with alpha {alpha}:", time.time() - start)
        frames.append(curr_frame)
        # im = plt.imshow(curr_frame)
        # mov.append([im])

    if boomerang:
        reversed_frames = copy.deepcopy(frames)
        reversed_frames.reverse()
        frames += reversed_frames

    # create video from frames

    metadata = {"title": "Morph Video", "artist": "Me = April Sin"}
    writer = animation.FFMpegWriter(fps=fps, metadata=metadata, bitrate=1800)
    with writer.saving(fig, outfile=out_path, dpi=100):
        for frame in frames:
            plt.imshow(frame)
            writer.grab_frame()
    return frames