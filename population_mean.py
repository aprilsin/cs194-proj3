from my_types import *

def read_population_shapes(array_of_imgs):
    for im in array_of_imgs:
        im = to_img_arr(im)
        shape_vector = np.genfromtxt()