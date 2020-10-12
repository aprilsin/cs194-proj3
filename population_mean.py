from my_types import *
import morph

def compute_population_mean(population_imgs: np.ndarray, population_pts: np.ndarray) ->np.ndarray:
    assert len(population_imgs) == len(population_pts)
    assert all(assert_img_type(img) for img in population_imgs)
    mean_points = np.mean(population_pts)
    mean_img = population_imgs[0]
    for img in population_imgs[1:]:
        morph.compute_middle_object()

def load_points_from_asf(file_name):
    points = np.genfromtxt(file_name)
    points.append([0.0, 0.0])
    points.append([1.0, 0.0])
    points.append([0.0, 1.0])
    points.append([1.0, 1.0])
    return np.array(points)


def caricature(img, mean_img, img_pts, mean_pts):
    unique_qualities = img_pts - mean_pts
    return img + unique_qualities
