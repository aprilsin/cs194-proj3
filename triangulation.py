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

from scipy.spatial import Delaunay
from scipy.interpolate import interp2d

""" Compute the (weighted) average points of correspondence"""
def avg_points(im1_pts, im2_pts, alpha=0.5):
    assert len(im1_pts) == len(im2_pts)
    im1_pts = np.array(im1_pts)
    im2_pts = np.array(im2_pts)
    assert type(im1_pts) == type(im2_pts) == np.ndarray
    final_points = []
    for i in range(len(im1_pts)):
        final_points.append((alpha * im1_pts[i] + (1 - alpha) * im2_pts[i]))
    return np.array(final_points)

""" Displays the triangular mesh of an image"""
def plot_tri_mesh(points, triangulation):
    plt.triplot(points[:,0], points[:,1], triangulation.simplices)
    plt.plot(points[:,0], points[:,1], 'o')
    plt.show()