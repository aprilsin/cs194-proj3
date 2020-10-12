import morph
from my_types import *


def compute_population_mean(
    population_imgs: np.ndarray, population_pts: np.ndarray
) -> np.ndarray:
    assert len(population_imgs) == len(population_pts)
    assert all(assert_img_type(img) for img in population_imgs)

    mean_img = population_imgs[0]
    # FIXME
    mean_pts = np.mean(population_pts, axis=0)
    assert_points(mean_pts)
    return mean_img


def load_points_from_asf(file_name):
    points = np.genfromtxt(file_name)
    points.append([0.0, 0.0])
    points.append([1.0, 0.0])
    points.append([0.0, 1.0])
    points.append([1.0, 1.0])
    return np.array(points)


def caricature(img, mean_img, img_pts, mean_pts):
    assert_img_type(img)
    assert_img_type(mean_img)
    assert_points(img_pts)
    assert_points(mean_pts)

    unique_qualities = img_pts - mean_pts
    return img + unique_qualities
