import math
import numpy as np
import matplotlib.pyplot as plt

def get_points(im1, im2, num_pts):
    print('Please select 2 points in each image for alignment.')
    plt.imshow(im1)
    im1_pts = []
    im2_pts = []
    for i in range(num_pts):
        p1 = plt.ginput()
        plt.close()
        plt.imshow(im2)
        p2 = plt.ginput()
        plt.close()
        im1_pts.append(p1)
        im2_pts.append(p2)
    return im1_pts, im2_pts