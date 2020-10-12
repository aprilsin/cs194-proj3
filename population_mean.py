import morph
import numpy as np
from my_types import *
from constants import POP_HEIGHT, POP_WIDTH
import utils


def fix_pop_pts(points: np.ndarray) -> np.ndarray:
    assert_points(points)
    corners = np.array([[0.0, 0.0], [0.0, 0.1], [0.0, 1.0], [1.0, 1.0]])
    np.append(points, corners)
    points[:, 0] *= POP_HEIGHT
    points[:, 1] *= POP_WIDTH
    return points


def compute_population_mean(
    population_imgs: Sequence[ToImgArray], population_pts: Sequence[ToPoints]
) -> np.ndarray:

    assert len(population_imgs) == len(population_pts)
    images = []
    points = []
    for img, pts in zip(population_imgs, population_pts):
        pts = to_points(pts)
        pts = fix_pop_pts(pts)
        img = to_img_arr(img)
        img, _ = utils.align_img(
            img, pts, left_idx=19, right_idx=27, SUPPRESS_DISPLAY=True
        )
        images.append(img)
        points.append(pts)
    assert_img_type(images[0])
    mean_pts = np.mean(points, axis=0)
    triangulation = morph.delaunay(mean_pts)

    num_imgs = len(images)
    alpha = 1 / num_imgs

    mean_img = np.zeros_like(images[0])
    for img, pts in zip(images, points):
        warped = morph.warp_img(img, pts, mean_pts, triangulation)
        mean_img += alpha * warped

    assert_points(mean_pts)
    assert_img_type(mean_img)
    return mean_img, mean_pts, triangulation


def caricature(img, mean_img, img_pts, mean_pts):
    assert_img_type(img)
    assert_img_type(mean_img)
    assert_points(img_pts)
    assert_points(mean_pts)

    unique_qualities = img_pts - mean_pts
    cari_pts = img_pts + unique_qualities
    return morph.warp_img(img, img_pts, cari_pts)
