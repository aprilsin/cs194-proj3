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


def pop_align(img, points, left_idx, right_idx):
    r, c = utils.find_centers(points[left_idx], points[right_idx])
    return utils.recenter(img, r, c)


def compute_population_mean(
    pop_imgs: Sequence[ToImgArray], pop_pts: Sequence[ToPoints]
) -> np.ndarray:

    assert len(pop_imgs) == len(pop_pts)
    for i, (img, pts) in enumerate(zip(pop_imgs, pop_pts)):
        pop_pts[i] = fix_pop_pts(to_points(pts))
        pop_imgs[i] = to_img_arr(img)

    # left = np.array([pt[19] for pt in pop_pts])  # shape [(2,)]

    # right = np.array([pt[27] for pt in pop_pts])  # shape [(2,)]

    tmp = np.array([(l + r) / 2 for l, r in zip(left, right)])
    cr, cc = tmp[:, 0], tmp[:, 1]

    for img in pop_imgs:
        # img = recenter(img, r, c)
    #     img = pop_align(img, pts, left_idx=19, right_idx=27)
    #     images.append(img)
    #     points.append(pts)
    # assert_img_type(images[0])
    # mean_pts = np.mean(points, axis=0)
    # triangulation = morph.delaunay(mean_pts)

    # num_imgs = len(images)
    # alpha = 1 / num_imgs

    # mean_img = np.zeros_like(images[0])
    # for img, pts in zip(images, points):
    #     warped = morph.warp_img(img, pts, mean_pts, triangulation)
    #     mean_img += alpha * warped

    # assert_points(mean_pts)
    # assert_img_type(mean_img)
    # return mean_img, mean_pts, triangulation


def caricature(img, mean_img, img_pts, mean_pts, alpha):
    assert_img_type(img)
    assert_img_type(mean_img)
    assert_points(img_pts)
    assert_points(mean_pts)

    unique_qualities = img_pts - mean_pts
    cari_pts = img_pts + alpha * unique_qualities
    return morph.warp_img(img, img_pts, cari_pts)
