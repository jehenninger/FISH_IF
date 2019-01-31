import argparse
import os
import imageio as io
from matplotlib import pyplot as plt
from scipy import ndimage as nd
import numpy as np
import pandas as pd
from skimage import img_as_float, img_as_uint, morphology, measure
from sklearn.cluster import KMeans
from datetime import datetime
import math

def parse_arguments(parser):

    # required arguments
    parser.add_argument("parent_dir")
    parser.add_argument("fish_channel")

    # optional arguments
    parser.add_argument("--o", type=str)
    parser.add_argument("--tm", type=float, default=3.0)
    parser.add_argument("--min_a", type=float, default=50)  # number of voxels
    parser.add_argument("--max_a", type=float, default=1000)  # number of voxels

    parser.add_argument("--manual", dest="autocall_flag", action="store_false", default=True)

    input_params = parser.parse_args()

    return input_params


def analyze_replicate(replicate_files, input_params, parent_dir):

    if (len(replicate_files) - 2) > 1:  # we subtract 2 here to account for the required DAPI and FISH channels
        input_params.multiple_IF_flag = True

    # get nuclear mask
    nucleus_image_file = [f for f in replicate_files if '405 DAPI' in f]
    nucleus_image_path = os.path.join(input_params.parent_dir, parent_dir, nucleus_image_file[0])

    nuclear_coords_r = []
    nuclear_coords_c = []
    nuclear_regions, nuclear_mask = find_nucleus(nucleus_image_path, input_params)
    # for idx, region in enumerate(nuclear_regions):
    #     nuclear_coords_r.append(region.coords[idx][0])  # for some reason, region.coords is a nested list
    #     nuclear_coords_c.append(region.coords[idx][1])  # JON THIS IS WRONG ANYWAY

    # get FISH spot
    fish_image_file = [s for s in replicate_files if input_params.fish_channel in s]
    fish_image_path = os.path.join(input_params.parent_dir, parent_dir, fish_image_file[0])

    fish_centers, fish_spots, fish_mask = find_fish_spot(fish_image_path, input_params)

    # filter FISH spots by nuclear localization and size
    fish_centers, fish_spots, fish_mask_new = filter_fish_spots(fish_centers, fish_spots, fish_mask,
                                                                nuclear_mask, input_params)
    # @Debug
    if True:
        fig, ax = plt.subplots(1, 2)

        ax[0].imshow(fish_mask[15,:,:], cmap='gray')
        ax[1].imshow(fish_mask_new[15,:,:], cmap='gray')

        plt.savefig(os.path.join(input_params.parent_dir, str(datetime.now()) +  "_fish_mask.png"), dpi=300)

        plt.close()


def find_nucleus(image_path, input_params):

    image = io.volread(image_path)  # image is [z, x, y] array
    image = nd.gaussian_filter(image, sigma=2.0)
    image = np.max(image, axis=0)  # max project the image because we just care about finding nuclei
    image = img_as_float(image)
    # threshold_multiplier = 0.25

    # trial method. Use K-means clustering on image to get nuclear pixels
    image_1d = image.reshape((-1, 1))
    #
    clusters = KMeans(n_clusters=2, random_state=0).fit_predict(image_1d)
    #
    cluster_mean = []
    for c in range(2):
        cluster_mean.append(np.mean(image_1d[clusters == c]))

    nuclear_cluster = np.argmax(cluster_mean)

    clusters = np.reshape(clusters, newshape=image.shape)
    nuclear_mask = np.zeros(image.shape)

    nuclear_mask[clusters == nuclear_cluster] = 1

    # # simple thresholding
    # mean_intensity = np.mean(image)
    # std_intensity = np.std(image)
    #
    # threshold = mean_intensity + (std_intensity * threshold_multiplier)
    # nuclear_mask[image > threshold] = 1

    nuclear_binary = nd.morphology.binary_fill_holes(nuclear_mask)

    nuclear_binary = nd.binary_erosion(nuclear_binary)  # to try to get rid of touching nuclei. Need to do better!
    nuclear_binary = nd.binary_erosion(nuclear_binary)
    nuclear_binary = nd.binary_erosion(nuclear_binary)
    nuclear_binary = nd.binary_erosion(nuclear_binary)
    nuclear_binary = nd.binary_erosion(nuclear_binary)
    nuclear_binary = nd.binary_erosion(nuclear_binary)

    nuclear_binary_labeled, num_of_regions = nd.label(nuclear_binary)

    nuclear_regions = measure.regionprops(nuclear_binary_labeled)
    # nuclear_regions = nd.find_objects(nuclear_binary_labeled) # @Deprecated

    # @Debug
    if False:
        fig, ax = plt.subplots(1, 2)

        ax[0].imshow(image, cmap='gray')
        ax[1].imshow(nuclear_binary, cmap='gray')

        plt.savefig(os.path.join(input_params.parent_dir, str(datetime.now()) +  "_test.png"), dpi=300)

        plt.close()

    return nuclear_regions, nuclear_mask

def find_fish_spot(image_path, input_params):

    image = io.volread(image_path)  # image is [z, x, y] array
    image = nd.gaussian_filter(image, sigma=2.0)
    # image = img_as_float(image)
    threshold_multiplier = input_params.tm

    fish_mask = np.zeros(image.shape)

    # simple thresholding
    mean_intensity = np.mean(image)/65536
    std_intensity = np.std(image)/65536

    image = img_as_float(image)
    threshold = mean_intensity + (std_intensity * threshold_multiplier)
    fish_mask[image > threshold] = 1

    fish_binary = nd.morphology.binary_fill_holes(fish_mask)
    # fish_binary = nd.binary_erosion(fish_binary)
    # fish_binary = nd.binary_erosion(fish_binary)
    # fish_binary = nd.binary_erosion(fish_binary)
    # fish_binary = nd.binary_erosion(fish_binary)
    # fish_binary = nd.binary_erosion(fish_binary)

    fish_binary_labeled, num_of_regions = nd.label(fish_binary)

    fish_regions = nd.find_objects(fish_binary_labeled)

    fish_centers = []
    for region in fish_regions:
        z = math.floor((region[0].stop - region[0].start) / 2)
        r = math.floor((region[1].stop - region[0].start) / 2)
        c = math.floor((region[2].stop - region[2].start) / 2)
        fish_centers.append([z, r, c])

    #fish_centers = nd.center_of_mass(fish_binary_labeled)

    # @Debug
    if False:
        fig, ax = plt.subplots(1, 2)

        ax[0].imshow(image[15,:,:], cmap='gray')
        ax[1].imshow(fish_binary[15,:,:], cmap='gray')

        plt.savefig(os.path.join(input_params.parent_dir, str(datetime.now()) +  "_fish_test.png"), dpi=300)

        plt.close()

    return fish_centers, fish_regions, fish_mask


def filter_fish_spots(fish_centers, fish_regions, fish_mask, nuclear_mask, input_params):
    # fish_spots_to_keep = [False] * len(fish_regions) # @Deprecated
    fish_spots_to_keep = np.zeros(len(fish_regions), dtype=bool)
    for idx, center in enumerate(fish_centers):
        fish_z = fish_regions[idx][0]
        fish_r = fish_regions[idx][1]
        fish_c = fish_regions[idx][2]

        # @ JON TO DO : Should probably calculate volume of the paralelloid here instead of the way I am doing it
        num_of_pixels_in_fish_region = (abs(fish_z.stop - fish_z.start) *
                                        abs(fish_r.stop - fish_r.start) *
                                        abs(fish_c.stop - fish_c.start))  # total number of pixels in bounding box

        if nuclear_mask[center[1], center[2]] == 1:  # makes sure that FISH center is in nucleus. Should be True if true.
            if input_params.min_a <= num_of_pixels_in_fish_region <= input_params.max_a:  # makes sure that FISH spot fits size criteria
                fish_spots_to_keep[idx] = True
        else:
            fish_mask[fish_z, fish_r, fish_c] = False

    fish_centers = fish_centers[fish_spots_to_keep]  # @ JON START HERE: Need to re-think if fish_centers should be numpy array, because this line won't work as is.
    fish_regions = fish_regions[fish_spots_to_keep]

    return fish_centers, fish_regions, fish_mask