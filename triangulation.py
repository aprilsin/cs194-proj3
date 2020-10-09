import math
import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial import Delaunay
from scipy.interpolate import interp2d

def avg_points(im1_pts, im2_pts, alpha):
    assert len(im1_pts) == len(im2_pts)
    im1_pts = np.array(im1_pts)
    im2_pts = np.array(im2_pts)
    assert type(im1_pts) == type(im2_pts) == np.ndarray
    final_points = []
    for i in range(len(im1_pts)):
        final_points.append((alpha * im1_pts[i] + (1 - alpha) * im2_pts[i]))
    return np.array(final_points)

