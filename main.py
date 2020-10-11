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
import sys
import morph, utils
from scipy.spatial import Delaunay
from scipy.interpolate import interp2d

from pathlib import Path

data = Path("input")
data.mkdir(parents=True, exist_ok=True)

#######################
#   TRIANGULATIONS    #
#######################


def avg_points(im1_pts: np.ndarray, im2_pts: np.ndarray, alpha=0.5) -> np.ndarray:
    """
    Compute the (weighted) average points of correspondence
    """
    assert len(im1_pts) == len(im2_pts)
    final_points = []
    for i in range(len(im1_pts)):
        final_points.append((alpha * im1_pts[i] + (1 - alpha) * im2_pts[i]))
    return np.array(final_points)


def delaunay(points):
    return Delaunay(points)


def plot_tri_mesh(points, triangulation) -> None:
    """
    Displays the triangular mesh of an image
    """
    plt.triplot(points[:, 0], points[:, 1], triangulation.simplices)
    plt.plot(points[:, 0], points[:, 1], "o")
    plt.show()


def main(im1_name, im2_name, num_pts):
    # import image
    im1 = utils.read_img(im1_name)
    im2 = utils.read_img(im2_name)


if __name__ == "__main__":
    im1_name = sys.argv[1]
    im2_name = sys.argv[2]
    #     num_pts = sys.argv[3]
    im1 = utils.setup_img(im1_name)
    im2 = utils.setup_img(im2_name)
    im1_pts = utils.load_points(im1_name)
    im2_pts = utils.load_points(im2_name)
    mid_pts = avg_points(im1_pts, im2_pts)
    plot_tri_mesh(mid_pts, delaunay(mid_pts))