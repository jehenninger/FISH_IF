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
from types import SimpleNamespace

def parse_arguments(parser):

    # required arguments
    parser.add_argument("parent_dir")
    parser.add_argument("fish_channel")

    # optional arguments
    parser.add_argument("--o", type=str)
    parser.add_argument("--tm", type=float, default=3.0)
    parser.add_argument("--min_a", type=float, default=1000)  # number of voxels
    parser.add_argument("--max_a", type=float, default=10000)  # number of voxels
    parser.add_argument("--c", type=float, default=0.7)  # circularity threshold
    parser.add_argument("--b", type=float, default=5)  # box edge length for graphing and quantification

    parser.add_argument("--manual", dest="autocall_flag", action="store_false", default=True)

    input_params = parser.parse_args()

    return input_params


def load_images(replicate_files, input_params, parent_dir):
    data = SimpleNamespace()  # this is the session data object that will be passed to functions

    if (len(replicate_files) - 2) > 1:  # we subtract 2 here to account for the required DAPI and FISH channels
        input_params.multiple_IF_flag = True

    # get replicate sample name
    nd_file_name = [n for n in replicate_files if '.nd' in n]
    if len(nd_file_name) == 1:
        sample_name = get_sample_name(nd_file_name[0])
        data.sample_name = sample_name
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
    data.nucleus_image = nucleus_image

    fish_image_file = [s for s in replicate_files if input_params.fish_channel in s and get_file_extension(s) == '.TIF']
    if len(fish_image_file) < 1:
        print('Error: Could not find fish image file')
        sys.exit(0)

    fish_image_path = os.path.join(input_params.parent_dir, parent_dir, fish_image_file[0])
    fish_image = io.volread(fish_image_path)
    data.fish_image = fish_image

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

    data.protein_images = protein_images
    data.protein_channel_names = protein_channel_names

    return data


def analyze_replicate(data, input_params):

    # get nuclear mask
    nuclear_regions, nuclear_mask, nuclear_binary_labeled = find_nucleus(data.nucleus_image, input_params)

    # get FISH spots
    fish_spots, fish_mask = find_fish_spot(data.fish_image, input_params)
    fish_mask_int = fish_mask*1 # because matplotlib doesn't like bools?

    # filter FISH spots by nuclear localization and size
    fish_spots_filt, fish_mask_filt, fish_spot_total_pixels, fish_centers, nucleus_with_fish_spot = filter_fish_spots(fish_spots, data.fish_image,
                                                                      fish_mask, nuclear_mask, nuclear_binary_labeled, input_params)
    fish_mask_filt_int = fish_mask_filt*1  # because matplotlib doesn't like bools?


    # measure IF channels
    individual_replicate_output = pd.DataFrame(columns=['sample', 'spot_id', 'IF_channel', 'mean_intensity'])

    for idx, image in enumerate(data.protein_images):
        for s, spot in enumerate(fish_spots_filt):
            mean_intensity = np.mean(image[spot])

            individual_replicate_output = individual_replicate_output.append({'sample': data.sample_name, 'spot_id': s,
                                                                              'IF_channel' : int(data.protein_channel_names[idx]),
                                                                              'mean_intensity' : mean_intensity},
                                                                             ignore_index=True)
    data.nuclear_regions = nuclear_regions
    data.nuclear_mask = nuclear_mask
    data.nuclear_binary_labeled = nuclear_binary_labeled
    data.fish_spots = fish_spots_filt
    data.fish_mask = fish_mask_filt
    data.fish_centers = fish_centers

    return individual_replicate_output, data


def generate_random_data(data, input_params):
    # maybe for every FISH spot, we select one random spot
    num_of_fish_spots = len(data.fish_spots)

    # JON START HERE. Write a function to filter nuclei based on ones that have FISH spots. Then maybe choose like 5-10 random spots.
    # pick random x, y, and z integers to make a random box within image.
    # choose low and high thresholds to make sure boxes don't go over edges of image

    mean_fish_length_z, mean_fish_length_r, mean_fish_length_c = find_average_fish_spot_parameter(data.fish_spots)

    rand_box_z = math.floor(mean_fish_length_z/2)
    rand_box_r = math.floor(mean_fish_length_r/2)
    rand_box_c = math.floor(mean_fish_length_c/2)

    r, c = np.where(data.nuclear_mask)  # limit x,y choice to nuclear pixels
    z_stack_num = data.fish_image.shape[0]  # limit z choice to z stack number
    z_range = get_middle_z_range(z_stack_num)
    z = list(range(z_range[0], z_range[1]))

    # NOTE: We take a shortcut for finding nuclear pixels where we do a max_z projection and only look at the 2D
    # image. This means that we don't know where the nuclear pixels are in the z. Therefore, for right now,
    # I am going to just find the middle 50% z-stacks of the image and call these valid. So this loop below will
    # find valid x,y pixels in nuclei, and then pick a random z within the middle 50% of the image.

    rand_spots = []
    for n in range(num_of_fish_spots):
        count = 1
        i = np.random.randint(len(r))

        valid_spot_flag = False
        while not valid_spot_flag:
            if all([0 < r[i] - rand_box_r < data.fish_image.shape[1],  # make sure we are in bounds of image
                    0 < c[i] - rand_box_c < data.fish_image.shape[2],  # make sure we are in bounds of image
                    data.nuclear_mask[r[r[i] - rand_box_r], c[c[i] - rand_box_c]],  # make sure we are still in nucleus with box
                    data.nuclear_mask[r[r[i] + rand_box_r], c[c[i] + rand_box_c]]]):

                zi = np.random.randint(len(z))

                rand_r = r[i]
                rand_c = c[i]
                rand_z = z[zi]

                rand_spots.append([slice(rand_z - rand_box_z, rand_z + rand_box_z),
                                   slice(rand_r - rand_box_r, rand_r + rand_box_r),
                                   slice(rand_c - rand_box_c, rand_c + rand_box_c)])

                valid_spot_flag = True

            else:
                i = np.random.randint(len(r))  # choose another spot if the previous one failed
                count += 1

        print("Number of random points selected before finding a valid one: ", count)
        print()

    data.rand_spots = rand_spots

    # @Debug
    test = np.full(shape=data.nuclear_mask.shape, fill_value=False, dtype=bool)
    for region in rand_spots:
        test[region[1], region[2]] = True

    test = test*1.0

    fig, ax = plt.subplots(1,2)

    ax[0].imshow(max_project(data.nucleus_image), cmap='gray')
    ax[1].imshow(test, cmap='gray')

    plt.savefig(os.path.join(input_params.parent_dir, data.sample_name + "_test_random_spot.png"), dpi=300)

    return data

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

    return nuclear_regions, nuclear_mask, nuclear_binary_labeled

def find_fish_spot(image, input_params):
    # because I run into memory errors, we will determine mean and std intensity on the middle 50% of the z-stack
    # @Improvement Could probably just choose x random spots and calculate mean/std from that. Maybe faster and more
    # memory efficient

    n = 500
    index = np.random.choice(image.shape[1], n, replace=False)
    middle_z = math.floor(image.shape[0]/2)

    # z_stacks = image.shape[0]
    # z_range = get_middle_z_range(z_stacks)
    # z_cropped_image = image[slice(z_range[0], z_range[1]),:,:]

    # z_cropped_image = nd.gaussian_filter(image, sigma=2.0)

    # image = img_as_float(image)
    threshold_multiplier = input_params.tm

    fish_mask = np.full(shape=image.shape, fill_value=False, dtype=bool)

    # simple thresholding
    mean_intensity = np.mean(image[middle_z, index, index])
    std_intensity = np.std(image[middle_z, index, index])
    print('Mean intensity: ', mean_intensity)
    print('Std intensity: ', std_intensity)

    # image = img_as_float(image)
    threshold = mean_intensity + (std_intensity * threshold_multiplier)
    # fish_mask[image > threshold] = True
    fish_mask[np.where(image > threshold)] = True

    fish_binary = nd.morphology.binary_fill_holes(fish_mask)
    fish_binary = nd.binary_erosion(fish_binary)
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

    return fish_regions, fish_binary


def filter_fish_spots(fish_regions, fish_image, fish_mask, nuclear_mask, nuclear_binary_labeled, input_params):
    # fish_spots_to_keep = [False] * len(fish_regions) # @Deprecated
    # fish_spots_to_keep = np.zeros(len(fish_regions), dtype=bool) @Deprecated
    fish_spots_to_keep = np.full(shape=(len(fish_regions), 1), fill_value=False, dtype=bool)
    fish_spot_total_pixels = []
    nucleus_with_fish_spot = []

    fish_centers = []

    for idx, region in enumerate(fish_regions):
        # spot = fish_image[region]
        spot_mask = fish_mask[region]

        # slice_of_max_z = np.argmax(spot, axis=0)
        # test_spot = spot[slice_of_max_z, :, :]
        test_spot_center_z = int(math.floor((region[0].start + region[0].stop)/2))
        test_spot_center_r = int(math.floor((region[1].start + region[1].stop)/2))
        test_spot_center_c = int(math.floor((region[2].start + region[2].stop)/2))

        fish_centers.append([test_spot_center_z, test_spot_center_r, test_spot_center_c])

        num_of_pixels_in_fish_region = np.sum(spot_mask)

        if nuclear_mask[test_spot_center_r, test_spot_center_c] and\
                input_params.min_a <= num_of_pixels_in_fish_region <= input_params.max_a:
            circularity = get_circularity_of_3D_spot(spot_mask)

            if circularity >= input_params.c:  # tests that spot is in nucleus, fits size and circularity threshold

                fish_spots_to_keep[idx] = True
                fish_spot_total_pixels.append(num_of_pixels_in_fish_region)

                nucleus_with_fish_spot.append(nuclear_binary_labeled[test_spot_center_r, test_spot_center_c])

            else:
                fish_mask[fish_regions[idx]] = False
        else:
            fish_mask[fish_regions[idx]] = False

    # fish_centers = list(compress(fish_centers,fish_spots_to_keep))
    fish_regions = list(compress(fish_regions, fish_spots_to_keep))

    print("Number of FISH spots after filtering: ", len(fish_regions))
    print()

    return fish_regions, fish_mask, fish_spot_total_pixels, fish_centers, nucleus_with_fish_spot


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

def get_circularity_of_3D_spot(spot_mask):
    # find circularity of max projection
    spot_mask_max = max_project(spot_mask)
    spot_mask_label, num_of_objects = nd.label(spot_mask_max)
    spot_mask_region = measure.regionprops(spot_mask_label)

    if num_of_objects == 1:
        circularity = circ(spot_mask_region[0])
    else:
        circularity = -1

    return circularity


def circ(region):
    circularity = (4 * math.pi * region.area) / (region.perimeter * region.perimeter)

    return circularity


def find_average_fish_spot_parameter(fish_spots):
    # this gets the average length in the z, x, and y directions for all fish spots of a given replicate
    r = []
    c = []
    z = []

    for region in fish_spots:
        z.append(region[0].stop - region[0].start)
        r.append(region[1].stop - region[1].start)
        c.append(region[2].stop - region[2].start)

    mean_z = np.mean(z)
    mean_r = np.mean(r)
    mean_c = np.mean(c)

    return mean_z, mean_r, mean_c

def make_output_directories(input_params):

    if input_params.o:
        output_parent_dir = os.path.join(os.path.dirname(input_params.parent_dir), input_params.o)
    else:
        output_parent_dir = os.path.join(os.path.dirname(input_params.parent_dir), 'output')

    output_dirs = {'parent': output_parent_dir,
                   'individual': os.path.join(output_parent_dir, 'individual'),
                   'summary': os.path.join(output_parent_dir, 'summary'),
                   'individual_images': os.path.join(output_parent_dir, 'individual', 'fish_spot_images')}

    # make folders if they don't exist
    if not os.path.isdir(output_parent_dir):
        os.mkdir(output_parent_dir)

    for key, folder in output_dirs.items():
        if key is not 'output_parent':
            if not os.path.isdir(folder):
                if not os.path.isdir(os.path.dirname(folder)):  # so I guess .items() is random order of dictionary keys. So when making subfolders, if the parent doesn't exist, then we would get an error. This accounts for that.
                    os.mkdir(os.path.dirname(folder))

                os.mkdir(folder)

    return output_dirs


def adjust_excel_column_width(writer, output):
    for key, sheet in writer.sheets.items():
        for idx, name in enumerate(output.columns):
            col_width = len(name) + 2
            sheet.set_column(idx, idx, col_width)

    return writer


def get_middle_z_range(z):
    # this will find the middle z stack indices of an image stack
    middle_z = math.floor(z/2)

    top_range = int(np.median(list(range(middle_z, z))))
    bottom_range = int(np.median(list(range(middle_z))))

    output = (bottom_range, top_range)

    return output