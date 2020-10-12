import morph
from my_types import *


def compute_population_mean(
    population_imgs: Sequence[ToImgArray], population_pts: Sequence[ToPoints]
) -> np.ndarray:

    assert len(population_imgs) == len(population_pts)
    population_imgs = [to_img_arr(img) for img in population_imgs]
    population_pts = [to_points(pts) for pts in population_pts]

    mean_img = np.zeros_like(population_imgs[0])
    num_imgs = len(population_imgs)
    alpha = 1 / num_imgs
    for img in population_imgs:
        mean_img += alpha * img
    mean_pts = np.mean(population_pts, axis=0)
    assert_points(mean_pts)
    assert_img_type(mean_img)
    return mean_img


def caricature(img, mean_img, img_pts, mean_pts):
    assert_img_type(img)
    assert_img_type(mean_img)
    assert_points(img_pts)
    assert_points(mean_pts)

    unique_qualities = img_pts - mean_pts
    return img + unique_qualities
