import numpy as np
import skimage as sk
import skimage.io as io
from skimage.util import img_as_ubyte, img_as_float
from pathlib import Path
import matplotlib.pyplot as plt
import math
import numpy as np
import matplotlib.pyplot as plt
from skimage import util, transform
from typing import Tuple
import pickle
import re

#######################
#   INPUT AND OUPUT   #
#######################

data = Path("input")
data.mkdir(parents=True, exist_ok=True)


def get_points(img: np.ndarray, num_pts: int) -> np.ndarray:
    """
    Returns an array of points for one image with ginput
    """
    print(f"Please select {num_pts} points in image.")

    plt.imshow(img)
    points = plt.ginput(num_pts)
    plt.close()

    points.append((0, 0))
    points.append((0, img.shape[1]))
    points.append((img.shape[0], 0))
    points.append((img.shape[0], img.shape[1]))

    return np.array(points)


def save_points(img_name, points) -> None:
    """
    Saves points as Pickle
    """
    pickle_name = re.split("\.", img_name)[0] + ".p"
    pickle.dump(points, open(pickle_name, "wb"))


def load_points(image_name, for_alignment=False) -> np.ndarray:
    """
    Loads an array of points saved as Pickle
    """
    pickle_name = re.split("\.", image_name)[0] + ".p"
    # FIXME assert path exists
    return pickle.load(open(pickle_name, "rb"))


def read_img(img_name) -> np.ndarray:
    """
    Input Image
    """
    im_path = data / img_name
    img = io.imread(im_path)
    img = img_as_float(img)
    assert img.dtype == "float64"
    return img


def check(img) -> None:
    """ Check image data type """
    assert img.dtype == "float64"


#######################
#   TRIANGULATIONS    #
#######################


def avg_points(im1_pts, im2_pts, alpha=0.5):
    """
    Compute the (weighted) average points of correspondence
    """
    assert len(im1_pts) == len(im2_pts)
    im1_pts = np.array(im1_pts)
    im2_pts = np.array(im2_pts)
    assert type(im1_pts) == type(im2_pts) == np.ndarray
    final_points = []
    for i in range(len(im1_pts)):
        final_points.append((alpha * im1_pts[i] + (1 - alpha) * im2_pts[i]))
    return np.array(final_points)


def plot_tri_mesh(points, triangulation):
    """
    Displays the triangular mesh of an image
    """
    plt.triplot(points[:, 0], points[:, 1], triangulation.simplices)
    plt.plot(points[:, 0], points[:, 1], "o")
    plt.show()