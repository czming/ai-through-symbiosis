"""
script to read in the ground truth picklists and convert them to
the expected format for the resnet data preprocessing scripts
(prep_labeled_object_data)
"""

from utils import *

import argparse
import os
from glob import glob

parser = argparse.ArgumentParser()

parser.add_argument("--input_folder", type=str, help="input folder", required=True)

parser.add_argument("--output_folder", type=str, help="folder to store output", required=True)

parser.add_argument("--config_file", "-c", type=str, default="configs/zm.yaml",
                    help="Path to experiment config (scripts/configs)")


args = parser.parse_args()

configs = load_yaml_config(args.config_file)

output_folder = args.output_folder

input_file_name_format = "picklist_{picklist_no}_raw.txt"

output_file_name_format = "picklist_{picklist_no}.csv"

PICKLISTS = range(136, 235)

for picklist_no in PICKLISTS:

    input_file = args.input_folder + input_file_name_format.format(picklist_no=picklist_no)

    try:
        with open(input_file, "r") as infile:
            # expected to just be one line with a1a1a1, letter-number format
            infile = infile.read().strip()
    except:
        print (f"Picklist {picklist_no} not readable")
        continue

    picked_colors = ""

    for i in range(0, len(infile), 2):
        # add the new colors
        picked_colors += infile[i]

    # write to the output folder with the same file name
    with open(output_folder + f"/{output_file_name_format.format(picklist_no=picklist_no)}", "w") as outfile:
        output_string = f"{picklist_no}, {picked_colors}"
        outfile.write(output_string)
        print (output_string)