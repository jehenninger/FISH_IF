#!/Users/jon/PycharmProjects/FISH_IF/venv/bin/python

import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['text.usetex'] = False
matplotlib.rcParams['font.sans-serif'] = 'Arial'
matplotlib.rcParams['font.family'] = 'sans-serif'

import methods
import grapher


# NOTE: If python/matplotlib cannot find the correct font, then run the following in the python console:
# matplotlib.font_manager._rebuild()

import numpy as np
import pandas as pd
import os
import sys
from types import SimpleNamespace
import argparse
import json
from datetime import datetime
import math


# This is written so that all replicates for a given experiment are in a folder together (both .TIF and .nd files)

# parse input
parser = argparse.ArgumentParser()

input_params = methods.parse_arguments(parser)
input_params.parent_dir = input_params.parent_dir.replace("Volumes","lab")
input_params.output_path = input_params.output_path.replace("Volumes","lab")


input_params.xy_um_per_px = 0.057
input_params.z_um_per_px = 0.2

# x = math.floor(input_params.b / input_params.xy_um_per_px)
x = input_params.b/input_params.xy_um_per_px

input_params.box_edge_xy = x  # size of box around FISH spot for plotting in xy. We will get z by average depth of fish spots

if not os.path.isdir(input_params.parent_dir):
    print('Error: Could not read or find parent directory')
    sys.exit(0)

print('Started at: ', datetime.now())
print()

# make output directories
output_dirs = methods.make_output_directories(input_params)

# get number of experiments/sub-directories to analyze
dir_list = os.listdir(input_params.parent_dir)
dir_list.sort(reverse=False)
file_ext = ".nd"

replicate_writer = pd.ExcelWriter(os.path.join(output_dirs['individual'], 'individual_spot_output.xlsx'),
                                  engine='xlsxwriter')
fish_writer      = pd.ExcelWriter(os.path.join(output_dirs['individual'], 'individual_fish_output.xlsx'),
                                  engine='xlsxwriter')
random_writer    = pd.ExcelWriter(os.path.join(output_dirs['individual'], 'random_spot_output.xlsx'),
                                  engine='xlsxwriter')

for folder in dir_list:  # folder is a separate experiment
    if not folder.startswith('.') and \
            os.path.isdir(os.path.join(input_params.parent_dir, folder)):  # to not include hidden files or folders



            input_params.multiple_IF_flag = False  # this is a flag that we switch if there are more than 1 IF channels to localize

            mean_protein_collection = []
            mean_fish_collection = []
            random_mean_collection = []

            file_list = os.listdir(os.path.join(input_params.parent_dir, folder))
            base_name_files = [f for f in file_list if file_ext in f
                               and os.path.isfile(os.path.join(input_params.parent_dir, folder,  f))]
            base_name_files.sort(reverse=False)

            if not input_params.autocall_flag:
                manual_metadata_file = [f for f in file_list if 'manual.xlsx' in f and '$' not in f
                                        and os.path.isfile(os.path.join(input_params.parent_dir, folder, f))]
                print(manual_metadata_file)
                if len(manual_metadata_file) == 1:
                    manual_metadata_file = manual_metadata_file[0]
                    manual_metadata = methods.load_manual_metadata(os.path.join(input_params.parent_dir, folder, manual_metadata_file))
                else:
                    print('Error: Could not find single manual metadata Excel file')
                    sys.exit(0)
            else:
                manual_metadata = None

            individual_replicate_output = pd.DataFrame(columns=['sample', 'spot_id', 'IF_channel', 'mean_intensity', 'max_intensity', 'center_r', 'center_c', 'center_z'])
            random_replicate_output = pd.DataFrame(columns=['sample', 'spot_id', 'IF_channel', 'mean_intensity', 'max_intensity', 'center_r', 'center_c', 'center_z'])
            individual_fish_output = pd.DataFrame(columns=['sample', 'spot_id', 'mean_intensity', 'max_intensity', 'center_r', 'center_c', 'center_z'])

            input_params.replicate_count_idx = 1
            for file in base_name_files:  # file is the nd file associated with a group of images for a replicate
                sample_name = file.replace(file_ext, '')
                replicate_files = [r for r in file_list if sample_name in r
                                   and os.path.isfile(os.path.join(input_params.parent_dir, folder, r))]

                data = methods.load_images(replicate_files, input_params, folder)
                data.output_directories = output_dirs

                temp_individual_replicate_output, temp_individual_fish_output, mean_protein_collection, mean_fish_storage, data\
                    =  methods.analyze_replicate(data, input_params, mean_protein_collection, manual_metadata=manual_metadata)

                if not temp_individual_replicate_output.empty:
                    mean_fish_collection.append(mean_fish_storage)

                    individual_replicate_output = individual_replicate_output.append(temp_individual_replicate_output, ignore_index=True)
                    individual_fish_output = individual_fish_output.append(temp_individual_fish_output, ignore_index=True)

                    temp_random_replicate_output, random_mean_collection, data = methods.generate_random_data(data, input_params, random_mean_collection)


                    random_replicate_output = random_replicate_output.append(temp_random_replicate_output, ignore_index=True)

                    grapher.make_image_output(data, input_params)

                input_params.replicate_count_idx += 1

            individual_replicate_output = individual_replicate_output[['sample', 'spot_id', 'IF_channel', 'mean_intensity', 'max_intensity', 'center_r', 'center_c', 'center_z']]
            individual_fish_output = individual_fish_output[
                ['sample', 'spot_id', 'mean_intensity', 'max_intensity', 'center_r', 'center_c', 'center_z']]
            random_replicate_output = random_replicate_output[
                ['sample', 'spot_id', 'IF_channel', 'mean_intensity', 'max_intensity', 'center_r', 'center_c', 'center_z']]

            if len(individual_replicate_output) > 0:
                individual_replicate_output.to_excel(replicate_writer, sheet_name=folder[0:15], index=False)
                individual_fish_output.to_excel(fish_writer, sheet_name=folder[0:15], index=False)
                random_replicate_output.to_excel(random_writer, sheet_name=folder[0:15], index=False)

            methods.analyze_sample(mean_fish_collection, mean_protein_collection, random_mean_collection, data, input_params, folder)

try:
    replicate_writer = methods.adjust_excel_column_width(replicate_writer, individual_replicate_output)
    random_writer = methods.adjust_excel_column_width(random_writer, random_replicate_output)
    fish_writer = methods.adjust_excel_column_width(fish_writer, individual_fish_output)
    replicate_writer.save()
    fish_writer.save()
    random_writer.save()

    methods.write_output_params(input_params)
except:
    print('Check directory structure because no experiment folders were found')

print("Finished at: ", datetime.now())
print()
print("------------------------ Completed -----------------------")
print()
