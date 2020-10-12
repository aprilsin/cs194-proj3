import argparse
import math
import pickle
import re
import sys
from os import path
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import skimage as sk
import skimage.io as io
from scipy.interpolate import interp2d
from scipy.spatial import Delaunay
from skimage import transform
from skimage.util import img_as_float, img_as_ubyte

import morph
import utils

NUM_POINTS = 41
# DATA_DIR = Path("input")
DATA_DIR = "data"
# DATA_DIR.mkdir(parents=True, exist_ok=True)

# PICKLE_DIR = Path("points")
# PICKLE_DIR.mkdir(parents=True, exist_ok=True)

OUT_DIR = Path("output")
OUT_DIR.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    intro = "Project 3 for CS 194-26: Face Morphing\n"

    parser = argparse.ArgumentParser(intro)

    parser.add_argument(
        "method",
        metavar="Method",
        type=str,
        help="Method to use (Align, Middle, Video, Population).",
    )

    parser.add_argument("im1", type=str, help="image 1 for morphing")

    parser.add_argument("im2", type=str, help="image 2 for morphing")

    parser.add_argument(
        "out",
        metavar="Output",
        type=str,
        default=OUT_DIR / Path("untitled"),
        help="Path in which to save the result.",
    )

    parser.add_argument(
        "-d",
        "--depth",
        dest="depth",
        type=int,
        default=10,
        help="Sets the number of interpolation layers for the movie.",
    )

    parser.add_argument(
        "--reset",
        dest="reset",
        action="store_const",
        const=True,
        default=False,
        help="Asks for point settings again",
    )

    args = parser.parse_args()
    if not (
        args.method == "Middle" or args.method == "Video" or args.method == "Population"
    ):
        print("Invalid method!")
        exit()

    if args.reset:
        input("If you don't want to remove the current config, Interrupt process (C-c)")
        sys("rm *.p")

    if args.align:
        utils.align_img(args.im1)
        utils.align_img(args.im2)
        exit()

    if not utils.shape_vector_exist(args.im1):
        utils.define_shape_vector(args.im1)
    if not utils.shape_vector_exist(args.im2):
        utils.define_shape_vector(args.im2)

    if args.method == "Middle":
        morph.compute_middle_object(args.im1, args.im2, args.out, args.alpha)
        exit()

    if args.method == "Video":
        morph.compute_morph_video(args.im1, args.im2, args.out, args.depth)
        exit()
