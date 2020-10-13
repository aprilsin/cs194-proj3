import morph
import numpy as np
from my_types import *
import utils


POP_HEIGHT = 480
POP_WIDTH = 640


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
    imgs, pts = np.stack([to_img_arr(img) for img in pop_imgs]), np.stack(
        [fix_pop_pts(to_points(p)) for p in pop_pts]
    )

    assert len(imgs) == len(pts), (len(imgs), len(pts))

    mean_shape = np.mean(pts, axis=0)
    print(mean_shape.min(), mean_shape.max(), mean_shape.shape)

    triangulation = morph.delaunay(mean_shape)
    warped_imgs = []
    for img, pt in zip(imgs, pts):
        try:
            w = morph.warp_img(img, pt, mean_shape, triangulation)
            warped_imgs.append(w)
        except np.linalg.LinAlgError:
            continue
    warped_imgs = np.stack(warped_imgs)
    alpha = 1 / len(warped_imgs)
    assert alpha >= 0 and alpha <= 1, alpha

    mean_img = (alpha * warped_imgs).sum(axis=0)
    assert_img_type(mean_img)
    return mean_img, mean_shape, triangulation, warped_imgs


def caricature(img, mean_img, img_pts, mean_pts, alpha):
    assert_img_type(img)
    assert_img_type(mean_img)
    assert_points(img_pts)
    assert_points(mean_pts)

    unique_qualities = img_pts - mean_pts
    cari_pts = img_pts + alpha * unique_qualities
    return morph.warp_img(img, img_pts, cari_pts)
