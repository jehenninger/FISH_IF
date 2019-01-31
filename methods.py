import argparse
import os
import sys
import imageio as io
from matplotlib import pyplot as plt
from scipy import ndimage as nd
import numpy as np
import pandas as pd
from skimage import img_as_float, img_as_uint, morphology, measure, color
from sklearn.cluster import KMeans
from datetime import datetime
import math
from itertools import compress

def parse_arguments(parser):

    # required arguments
    parser.add_argument("parent_dir")
    parser.add_argument("fish_channel")

    # optional arguments
    parser.add_argument("--o", type=str)
    parser.add_argument("--tm", type=float, default=3.0)
    parser.add_argument("--min_a", type=float, default=500)  # number of voxels
    parser.add_argument("--max_a", type=float, default=3000)  # number of voxels

    parser.add_argument("--manual", dest="autocall_flag", action="store_false", default=True)

    input_params = parser.parse_args()

    return input_params


def analyze_replicate(replicate_files, input_params, parent_dir):

    if (len(replicate_files) - 2) > 1:  # we subtract 2 here to account for the required DAPI and FISH channels
        input_params.multiple_IF_flag = True

    # get replicate sample name
    nd_file_name = [n for n in replicate_files if '.nd' in n]
    if len(nd_file_name) == 1:
        sample_name = get_sample_name(nd_file_name[0])
    else:
        print('Error: Found too many .nd files in sample directory')
        sys.exit(0)

    print(sample_name)

    # load images
    nucleus_image_file = [f for f in replicate_files if all(['405 DAPI' in f, get_file_extension(f) == '.TIF'])]
    if len(nucleus_image_file) < 1:
        print('Error: Could not find nucleus image file')
        sys.exit(0)

    nucleus_image_path = os.path.join(input_params.parent_dir, parent_dir, nucleus_image_file[0])
    nucleus_image = io.volread(nucleus_image_path)  # image is [z, x, y] array

    fish_image_file = [s for s in replicate_files if input_params.fish_channel in s and get_file_extension(s) == '.TIF']
    if len(fish_image_file) < 1:
        print('Error: Could not find fish image file')
        sys.exit(0)

    fish_image_path = os.path.join(input_params.parent_dir, parent_dir, fish_image_file[0])
    fish_image = io.volread(fish_image_path)

    protein_image_files = [p for p in replicate_files if
                           all(['405 DAPI' not in p,
                                input_params.fish_channel not in p,
                                get_file_extension(p) == '.TIF'])]
    if len(protein_image_files) < 1:
        print('Error: Could not find protein image files')
        sys.exit(0)

    protein_image_paths = []
    protein_images = []
    protein_channel_names = []
    for idx, p in enumerate(protein_image_files):
        protein_image_paths.append(os.path.join(input_params.parent_dir, parent_dir, p))
        protein_channel_names.append(find_image_channel_name(p))
        protein_images.append(io.volread(protein_image_paths[idx]))

    # get nuclear mask
    nuclear_regions, nuclear_mask = find_nucleus(nucleus_image, input_params)

    # get FISH spots
    fish_spots, fish_mask = find_fish_spot(fish_image, input_params)

    fish_mask_int = fish_mask*1 # because matplotlib doesn't like bools?
    # filter FISH spots by nuclear localization and size
    fish_spots, fish_mask_new, fish_spot_total_pixels = filter_fish_spots(fish_spots, fish_image,
                                                                      fish_mask, nuclear_mask, input_params)
    fish_mask_new_int = fish_mask_new*1  # because matplotlib doesn't like bools?
   # @Debug
    if True:
        # for i in range(0, fish_image.shape[0]):
        #     fig, ax = plt.subplots(1, 4)
        #     ax[0].imshow(nucleus_image[i,:,:], cmap='gray')
        #     ax[1].imshow(fish_image[i,:,:], cmap='gray')
        #     image_label_overlay = color.label2rgb(fish_binary_labeled[i,:,:], fish_image[i,:,:], bg_label=0)
        #     ax[2].imshow(image_label_overlay)
        #     ax[3].imshow(new_binary[i,:,:], cmap='gray')
        #     # ax[4].imshow(fish_mask_new[i,:,:], cmap='gray')
        #
        #     for x in ax:
        #         clear_axis_ticks(x)
        #
        #     plt.suptitle(sample_name, fontsize=4)
        #     plt.savefig(os.path.join(input_params.parent_dir, sample_name + "_" + str(i) + "_fish_mask.png"), dpi=300)
        #
        #     plt.close()

        fig, ax = plt.subplots(1, 4)
        ax[0].imshow(max_project(nucleus_image), cmap='gray')
        ax[1].imshow(max_project(fish_image), cmap='gray')
        ax[2].imshow(max_project(fish_mask_int), cmap='gray')
        ax[3].imshow(max_project(fish_mask_new_int), cmap='gray')

        for x in ax:
            clear_axis_ticks(x)

        plt.title(sample_name)
        plt.savefig(os.path.join(input_params.parent_dir, sample_name +  "_fish_mask.png"), dpi=300)

        plt.close()


def find_nucleus(image, input_params):
    image = nd.gaussian_filter(image, sigma=2.0)
    image = max_project(image)  # max project the image because we just care about finding nuclei
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
    nuclear_mask = np.full(shape=image.shape, fill_value=False, dtype=bool)
    nuclear_mask[clusters == nuclear_cluster] = True

    # nuclear_mask = np.zeros(image.shape)  # @Deprecated
    # nuclear_mask[clusters == nuclear_cluster] = 1 @Deprecated

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

def find_fish_spot(image, input_params):
    image = nd.gaussian_filter(image, sigma=2.0)
    # image = img_as_float(image)
    threshold_multiplier = input_params.tm

    fish_mask = np.full(shape=image.shape, fill_value=False, dtype=bool)

    # simple thresholding
    mean_intensity = np.mean(image)/65536
    std_intensity = np.std(image)/65536

    image = img_as_float(image)
    threshold = mean_intensity + (std_intensity * threshold_multiplier)
    # fish_mask[image > threshold] = True
    fish_mask[np.where(image > threshold)] = True

    fish_binary = nd.morphology.binary_fill_holes(fish_mask)
    # fish_binary = nd.binary_erosion(fish_binary)
    # fish_binary = nd.binary_erosion(fish_binary)
    # fish_binary = nd.binary_erosion(fish_binary)
    # fish_binary = nd.binary_erosion(fish_binary)
    # fish_binary = nd.binary_erosion(fish_binary)

    fish_binary_labeled, num_of_regions = nd.label(fish_binary)

    fish_regions = nd.find_objects(fish_binary_labeled)

    # fish_centers = []
    # for region in fish_regions:
    #     z = math.floor((region[0].stop - region[0].start) / 2)
    #     r = math.floor((region[1].stop - region[0].start) / 2)
    #     c = math.floor((region[2].stop - region[2].start) / 2)
    #     fish_centers.append([z, r, c])

    #fish_centers = nd.center_of_mass(fish_binary_labeled)

    # @Debug
    if False:
        for i in range(0, image.shape[1]):
            fig, ax = plt.subplots(1, 2)

            ax[0].imshow(image[i,:,:], cmap='gray')
            image_label_overlay = color.label2rgb(fish_binary_labeled[i, : ,:], image[i,:,:])

            ax[1].imshow(image_label_overlay)

            plt.savefig(os.path.join(input_params.parent_dir, str(datetime.now()) +str(i) +  "_fish_test.png"), dpi=300)

            plt.close()

    return fish_regions, fish_binary


def filter_fish_spots(fish_regions, fish_image, fish_mask, nuclear_mask, input_params):
    # fish_spots_to_keep = [False] * len(fish_regions) # @Deprecated
    # fish_spots_to_keep = np.zeros(len(fish_regions), dtype=bool) @Deprecated
    fish_spots_to_keep = np.full(shape=(len(fish_regions), 1), fill_value=False, dtype=bool)
    fish_spot_total_pixels = []

    for idx, region in enumerate(fish_regions):
        # spot = fish_image[region]
        spot_mask = fish_mask[region]

        # find circularity of max projection 

        # slice_of_max_z = np.argmax(spot, axis=0)
        # test_spot = spot[slice_of_max_z, :, :]

        test_spot_center_r = int(math.floor((region[1].start + region[1].stop)/2))
        test_spot_center_c = int(math.floor((region[2].start + region[2].stop)/2))

        num_of_pixels_in_fish_region = np.sum(spot_mask)

        if nuclear_mask[test_spot_center_r, test_spot_center_c]:  # makes sure that FISH center is in nucleus. Should be True if true.
            if input_params.min_a <= num_of_pixels_in_fish_region <= input_params.max_a:  # makes sure that FISH spot fits size criteria
                fish_spots_to_keep[idx] = True
                fish_spot_total_pixels.append(num_of_pixels_in_fish_region)
        else:
            fish_mask[fish_regions[idx]] = 0

    # fish_centers = list(compress(fish_centers,fish_spots_to_keep))
    fish_regions = list(compress(fish_regions, fish_spots_to_keep))

    return fish_regions, fish_mask, fish_spot_total_pixels

def max_project(image):
    projection = np.max(image, axis=0)

    return projection

def get_file_extension(file_path):
    file_ext = os.path.splitext(file_path)

    return file_ext[1]  # because splitext returns a tuple, and the extension is the second element

def find_image_channel_name(file_name):
    str_idx = file_name.find('Conf ')  # this is specific to our microscopes file name format
    channel_name = file_name[str_idx + 5 : str_idx + 8]

    return channel_name

def get_sample_name(nd_file_name):
    sample_name, ext = os.path.splitext(nd_file_name)

    return sample_name

def clear_axis_ticks(ax):
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])