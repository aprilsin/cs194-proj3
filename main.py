import sys
from align_images_code import get_points, align_images
import triangulation, utils

from pathlib import Path
data = Path("input")
data.mkdir(parents=True, exist_ok=True)

def main(im1_name, im2_name, num_pts):
    # import image
    im1 = io.imread(data/im1_name)
    im2 = io.imread(data/im2_name)
    
    # align image
    im1_align_pts, im2_align_pts = get_points(im1, im2, 2)
    align_images(im1, im2, im1_align_pts, im2_align_pts)
    
    # find correspondeces
    im1_morph_pts, im2_morph_pts = get_point(im1, im2, 10)
    
if __name__ == "__main__":
    im1_name = sys.argv[1]
    im2_name = sys.argv[2]
    num_pts = sys.argv[3]
    main(im1_name, im2_name, num_pts)