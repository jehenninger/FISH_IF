import methods
import grapher

import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import os
import sys
from types import SimpleNamespace
import argparse
import json
from datetime import datetime


# This is written so that all replicates for a given experiment are in a folder together (both .TIF and .nd files)

# parse input
parser = argparse.ArgumentParser()

input_params = methods.parse_arguments(parser)

input_params.multiple_IF_flag = False  # this is a flag that we switch if there are more than 1 IF channels to localize
input_params.xy_um_per_px = 0.057
input_params.z_um_per_px = 0.2

x = input_params.b / input_params.xy_um_per_px

input_params.box_edge_xy = x  # size of box around FISH spot for plotting in xy. We will get z by average depth of fish spots

if not os.path.isdir(input_params.parent_dir):
    print('Error: Could not read or find parent directory')
    sys.exit(0)

# make output directories
output_dirs = methods.make_output_directories(input_params)

# get number of experiments/sub-directories to analyze
dir_list = os.listdir(input_params.parent_dir)
dir_list.sort(reverse=False)
file_ext = ".nd"

replicate_writer = pd.ExcelWriter(os.path.join(output_dirs['individual'], 'individual_spot_output.xlsx'),
                                  engine='xlsxwriter')
random_writer = pd.ExcelWriter(os.path.join(output_dirs['individual'], 'random_spot_output.xlsx'),
                                  engine='xlsxwriter')

for folder in dir_list:  # folder is a separate experiment
    if not folder.startswith('.') and \
            os.path.isdir(os.path.join(input_params.parent_dir, folder)):  # to not include hidden files or folders

            file_list = os.listdir(os.path.join(input_params.parent_dir, folder))
            base_name_files = [f for f in file_list if file_ext in f
                               and os.path.isfile(os.path.join(input_params.parent_dir, folder,  f))]
            base_name_files.sort(reverse=False)

            individual_replicate_output = pd.DataFrame(columns=['sample', 'spot_id', 'IF_channel', 'mean_intensity'])
            random_replicate_output = pd.DataFrame(columns=['sample', 'spot_id', 'IF_channel', 'mean_intensity', 'center_r', 'center_c', 'center_z'])

            for file in base_name_files:  # file is the nd file associated with a group of images for a replicate
                sample_name = file.replace(file_ext, '')
                replicate_files = [r for r in file_list if sample_name in r
                                   and os.path.isfile(os.path.join(input_params.parent_dir, folder, r))]

                data = methods.load_images(replicate_files, input_params, folder)

                temp_individual_replicate_output, data =  methods.analyze_replicate(data, input_params)

                individual_replicate_output = individual_replicate_output.append(temp_individual_replicate_output, ignore_index=True)

                data = methods.generate_random_data(data, input_params)

                # temp_random_output = methods.analyze_random(data, input_params, folder)

                # random_output = random_output.append(temp_random_output, ignore_index=True)


            individual_replicate_output.to_excel(replicate_writer, sheet_name=folder[0:15], index=False)


replicate_writer = methods.adjust_excel_column_width(replicate_writer, individual_replicate_output)
replicate_writer.save()
