# morphing sequence

import copy
import math
import time
from pathlib import Path
from typing import Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import skimage as sk
import skimage.io as io
from matplotlib import animation
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

    assert_is_triangle(start)
    assert_is_triangle(target)
    # print(start.T.shape, start[:, 0])
    A = np.vstack((start[:, 0], start[:, 1], [1, 1, 1]))
    # print(A)
    # print(A.shape)
    try:
        inv = np.linalg.inv(A)
    except:
        return
    B = np.vstack((target[:, 0], target[:, 1], [1, 1, 1]))
    # print(B)
    # B = T * A
    # T = B * A^-1
    T = B @ inv
    return T
    # return np.linalg.inv(T) # TODO why anoter inverse? because we inversed the input


# Vanessa
def compute_affine(tri1_pts, tri2_pts):
    # print(tri1_pts.T.shape, tri1_pts[:, 0])
    source = np.vstack((tri1_pts.T, [1, 1, 1]))
    # print(source)
    target = np.vstack((tri2_pts.T, [1, 1, 1]))
    # print(target)

    # T = inv(target * inv(source))
    A = np.dot(target, np.linalg.inv(source))
    inverse_A = np.linalg.inv(A)
    # print(inverse_A)
    return inverse_A


def get_triangle_pixels(
    triangle_vertices: np.ndarray, shape=(DEFAULT_HEIGHT, DEFAULT_WIDTH)
):
    """
    Returns the coordinates of pixels within triangle for an image
    """
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
    rr, cc = get_triangle_pixels(triangle_vertices, shape)
    mask[rr, cc] = 1.0
    utils.assert_img_type(mask)
    return mask


def inverse_affine(img, img_triangle_vertices, target_triangle_vertices):
    """ Returns the coordinates of pixels from original image. """
    assert_img_type(img)
    assert_is_triangle(img_triangle_vertices)
    assert_is_triangle(target_triangle_vertices)

    # affine_mat = get_affine_mat(img_triangle_vertices, target_triangle_vertices)
    # # inverse of affine is just the transpose
    # # inverse = np.linalg.inv(affine_mat) # TODO why don't I need to do this?
    # rr, cc = get_triangle_pixels(target_triangle_vertices, img.shape)
    # target_points = np.vstack([rr, cc, np.ones(len(rr))])
    # src_points = affine_mat @ target_points
    # return src_points

    # img = im
    # img_triangle_vertices = im_t_points[i]
    # target_triangle_vertices = avg_t_points[i]
    affine_mat = get_affine_mat(img_triangle_vertices, target_triangle_vertices)
    # inverse of affine is just the transpose
    # inverse = np.linalg.inv(affine_mat)
    rr, cc = get_triangle_pixels(target_triangle_vertices, img.shape)
    print(img.shape)
    # print(rr, cc)
    target_points = np.vstack([rr, cc, np.ones(len(rr))])
    # print(target_points)
    src_points = affine_mat @ target_points
    print(src_points)
    # transformed = np.around(src_points)
    transformed = src_points
    return transformed


def warp_img(
    img: np.ndarray,
    img_pts: np.ndarray,
    target_pts: np.ndarray,
    triangulation: Delaunay,
) -> np.ndarray:
    assert_img_type(img)
    assert_points(img_pts)
    assert_points(target_pts)
    h, w, c = img.shape
    warped = np.zeros_like(img)
    # num_triangles, _, _ = triangulation.simplices

    # Interpolation functions
    # f_red = interpolate.RectBivariateSpline(
    #     range(img.shape[0]), range(img.shape[1]), img[:, :, 0]
    # )
    # f_green = interpolate.RectBivariateSpline(
    #     range(img.shape[0]), range(img.shape[1]), img[:, :, 1]
    # )
    # f_blue = interpolate.RectBivariateSpline(
    #     range(img.shape[0]), range(img.shape[1]), img[:, :, 2]
    # )

    for simplex in triangulation.simplices:

        target_vertices = target_pts[simplex]
        img_vertices = img_pts[simplex]
        assert_is_triangle(target_vertices)
        assert_is_triangle(img_vertices)

        # do inverse warping
        h, w, _ = img.shape
        target_rr, target_cc = get_triangle_pixels(target_vertices, shape=(w, h))
        print(target_rr.shape, target_cc.shape)
        src_rr, src_cc = inverse_affine(img, img_vertices, target_vertices)

        # assert not DEFAULT_HEIGHT in [max(target_rr), max(src_rr)]
        # assert not DEFAULT_WIDTH in [max(target_cc), max(src_cc)]

        src_rr, src_cc = (
            utils.ifloor(src_rr).clip(0, h - 1),
            utils.ifloor(src_cc).clip(0, w - 1),
        )
        # Transform points to the source image domain
        #         transformed = np.around(affine_mats[i] @ np.vstack((rr, cc, np.ones(len(rr))))).astype(int)
        transformed = src_rr, src_cc
        # Interpolate
        # warped[target_cc, target_rr, 0] = f_red.ev(transformed[1], transformed[0])
        # warped[target_cc, target_rr, 1] = f_green.ev(transformed[1], transformed[0])
        # warped[target_cc, target_rr, 2] = f_blue.ev(transformed[1], transformed[0])
        # TODO remove this
        break
    #         warped[target_rr, target_cc] = img[src_rr, src_cc]

    #     result = np.flip(transform.rotate(warped, -90), axis=1)
    warped = np.clip(warped, 0.0, 1.0)
    assert_img_type(warped)
    return warped


## From Venessa Lin ###

# Warp images to shape
def warp_image_to(im, im_points, avg_points, del_tri: Delaunay, vanessa=True):
    assert isinstance(del_tri, Delaunay)

    del_tri = del_tri.simplices.copy()

    x, y, _ = im.shape
    # Points of triangles
    im_t_points = im_points[del_tri].copy()
    avg_t_points = avg_points[del_tri].copy()
    # im_t_points = points_from_delaunay(im_points, del_tri)
    # avg_t_points = points_from_delaunay(avg_points, del_tri)

    # Affine transformations
    affine_mats = []

    # Affine transformations for triangles
    for i in range(len(del_tri)):
        affine_mats.append(compute_affine(im_t_points[i], avg_t_points[i]))
    # Create warped image
    new_im = np.zeros(im.shape)

    # Interpolation functions
    f_red = interpolate.RectBivariateSpline(
        range(im.shape[0]), range(im.shape[1]), im[:, :, 0]
    )
    f_green = interpolate.RectBivariateSpline(
        range(im.shape[0]), range(im.shape[1]), im[:, :, 1]
    )
    f_blue = interpolate.RectBivariateSpline(
        range(im.shape[0]), range(im.shape[1]), im[:, :, 2]
    )

    for i in range(len(affine_mats)):
        # Mask
        if vanessa:
            rr, cc = sk.draw.polygon(
                avg_t_points[i].T[0], avg_t_points[i].T[1], shape=(y, x)
            )
        else:
            rr, cc = get_triangle_pixels(avg_t_points[i], shape=(y, x))
        # THIS IS GOOD
        # print(rr, cc)
        # rr1, cc1 = sk.draw.polygon(
        #     avg_t_points[i].T[0], avg_t_points[i].T[1], shape=(y, x)
        # )
        # rr2, cc2 = get_triangle_pixels(avg_t_points[i], shape=(y, x))
        # print(np.equal(rr1, rr2).all(), np.equal(cc1, cc2).all())

        # Transform points to the source image domain
        if vanessa:
            target_points = np.vstack(
                (rr, cc, np.ones(len(rr)))
            )  # append 1 to all rows?
            transformed = np.around(affine_mats[i] @ target_points).astype(int)

        else:
            transformed = inverse_affine(im, im_t_points[i], avg_t_points[i])

            img = im
            img_triangle_vertices = im_t_points[i]
            target_triangle_vertices = avg_t_points[i]
            affine_mat = get_affine_mat(img_triangle_vertices, target_triangle_vertices)
            # inverse of affine is just the transpose
            # inverse = np.linalg.inv(affine_mat)
            rr, cc = get_triangle_pixels(target_triangle_vertices, img.shape)
            print(img.shape)
            # print(rr, cc)
            target_points = np.vstack([rr, cc, np.ones(len(rr))])
            # print(target_points)
            src_points = affine_mat @ target_points
            print(src_points)
            # transformed = np.around(src_points)
            # transformed = src_points #TODO

        # print(transformed[0].shape)

        # Interpolate
        new_im[cc, rr, 0] = f_red.ev(transformed[1], transformed[0])
        new_im[cc, rr, 1] = f_green.ev(transformed[1], transformed[0])
        new_im[cc, rr, 2] = f_blue.ev(transformed[1], transformed[0])

        # TODO remove this
        break

    return new_im


def cross_dissolve(warped_im1, warped_im2, alpha):
    assert_img_type(warped_im1)
    assert_img_type(warped_im2)
    return weighted_avg(warped_im1, warped_im2, alpha=alpha)


def compute_middle_object(
    im1: ToImgArray,
    im2: ToImgArray,
    im1_pts: ToPoints,
    im2_pts: ToPoints,
    alpha,
    vanessa=True,
):
    im1 = to_img_arr(im1)
    im2 = to_img_arr(im2)
    im1_pts = to_points(im1_pts)
    im2_pts = to_points(im2_pts)

    mid_pts = weighted_avg(im1_pts, im2_pts, alpha=alpha)
    triangulation = delaunay(mid_pts)
    if vanessa:
        im1_warped = warp_image_to(im1, im1_pts, mid_pts, triangulation)
        im2_warped = warp_image_to(im2, im2_pts, mid_pts, triangulation)
    else:
        im1_warped = warp_img(im1, im1_pts, mid_pts, triangulation)
        im2_warped = warp_img(im2, im2_pts, mid_pts, triangulation)

    middle_img = cross_dissolve(im1_warped, im2_warped, alpha=alpha)
    # middle_img = transform.rotate(middle_img, -90)
    return middle_img, mid_pts, triangulation


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
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
    fig.set_size_inches(int(im1.shape[1] / 50), int(im1.shape[0] / 50))
    alphas = np.linspace(0, 1, num_frames)
    # fig = plt.figure(frameon=False)
    # plt.axis("off")
    for i, alpha in enumerate(alphas, start=1):
        start = time.time()
        curr_frame, _, _ = compute_middle_object(im1, im2, im1_pts, im2_pts, 1 - alpha)
        print(f"Frame {i} morph time with alpha {alpha}:", time.time() - start)
        frames.append(curr_frame)

    if boomerang:
        reversed_frames = copy.deepcopy(frames)
        reversed_frames.reverse()
        frames += reversed_frames

    # create video from frames

    metadata = {"title": "Morph Video", "artist": "Me = April Sin"}
    writer = animation.FFMpegWriter(fps=fps, metadata=metadata, bitrate=1800)
    with writer.saving(fig, outfile=out_path, dpi=100):
        for frame in frames:
            assert_img_type(frame)
            plt.imshow(frame)
            writer.grab_frame()
    return frames
