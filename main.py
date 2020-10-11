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

from pathlib import Path

data = Path("input")
data.mkdir(parents=True, exist_ok=True)







def main(im1_name, im2_name, num_pts):
    # import image
    im1 = utils.read_img(data / im1_name)
    im2 = utils.read_img(data / im2_name)

    # align image
    im1_align_pts, im2_align_pts = get_points(im1, im2, 2)
    align_images(im1, im2, im1_align_pts, im2_align_pts)

    # find correspondeces
    im1_morph_pts = utils.get_point(im1, NUM_POINTS)
    im2_morph_pts = utils.get_point(im2, NUM_POINTS)


if __name__ == "__main__":
    im1_name = sys.argv[1]
    im2_name = sys.argv[2]
    num_pts = sys.argv[3]
    main(im1_name, im2_name, num_pts)