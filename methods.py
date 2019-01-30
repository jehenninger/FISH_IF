import argparse
import os

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
    print(nucleus_image_path)
    # nuclear_regions, nuclear_mask = find_nucleus()
