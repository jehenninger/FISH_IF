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

    nuclear_regions, nuclear_mask = find_nucleus(nucleus_image_path, input_params)

    # get FISH spot
    fish_image_file = [s for s in replicate_files if input_params.fish_channel in s]
    fish_image_path = os.path.join(input_params.parent_dir, parent_dir, fish_image_file[0])

    fish_spots, fish_mask = find_fish_spot(fish_image_path, input_params)


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
    # fig, ax = plt.subplots(1, 2)
    #
    # ax[0].imshow(image, cmap='gray')
    # ax[1].imshow(nuclear_binary, cmap='gray')
    #
    # plt.savefig(os.path.join(input_params.parent_dir, str(datetime.now()) +  "_test.png"), dpi=300)
    #
    # plt.close()

    return nuclear_regions, nuclear_mask

def find_fish_spot(image_path, input_params):

    image = io.volread(image_path)  # image is [z, x, y] array
    image = nd.gaussian_filter(image, sigma=2.0)
    image = img_as_float(image)
    threshold_multiplier = input_params.tm

    fish_mask = np.zeros(image.shape)

    # simple thresholding
    mean_intensity = np.mean(image)
    std_intensity = np.std(image)

    threshold = mean_intensity + (std_intensity * threshold_multiplier)
    fish_mask[image > threshold] = 1

    fish_binary = nd.morphology.binary_fill_holes(fish_mask)
    fish_binary = nd.binary_erosion(fish_binary)
    fish_binary = nd.binary_erosion(fish_binary)
    fish_binary = nd.binary_erosion(fish_binary)
    fish_binary = nd.binary_erosion(fish_binary)
    fish_binary = nd.binary_erosion(fish_binary)

    fish_binary_labeled, num_of_regions = nd.label(fish_binary)


    fish_regions = nd.find_objects(fish_binary_labeled)

    # @Debug
    if False:
        fig, ax = plt.subplots(1, 2)

        ax[0].imshow(image[15,:,:], cmap='gray')
        ax[1].imshow(fish_binary[15,:,:], cmap='gray')

        plt.savefig(os.path.join(input_params.parent_dir, str(datetime.now()) +  "_fish_test.png"), dpi=300)

        plt.close()

    return fish_regions, fish_mask
