import methods
import grapher

import numpy as np
import pandas as pd
import os
import sys

import argparse
import json
from datetime import datetime


# This is written so that all replicates for a given experiment are in a folder together (both .TIF and .nd files)

# parse input
parser = argparse.ArgumentParser()

input_params = methods.parse_arguments(parser)

input_params.multiple_IF_flag = False  # this is a flag that we switch if there are more than 1 IF channels to localize

if not os.path.isdir(input_params.parent_dir):
    print('Error: Could not read or find parent directory')
    sys.exit(0)

# get number of experiments/sub-directories to analyze
dir_list = os.listdir(input_params.parent_dir)
file_ext = ".nd"

for folder in dir_list:
    if not folder.startswith('.'):  # to not include hidden files or folders
        file_list = os.listdir(os.path.join(input_params.parent_dir, folder))
        base_name_files = [f for f in file_list if file_ext in f
                           and os.path.isfile(os.path.join(input_params.parent_dir, folder,  f))]
        base_name_files.sort(reverse=False)
        for file in base_name_files:
            sample_name = file.replace(file_ext, '')
            replicate_files = [r for r in file_list if sample_name in r
                               and os.path.isfile(os.path.join(input_params.parent_dir, folder, r))]

            methods.analyze_replicate(replicate_files, input_params, folder)


# generate output folders