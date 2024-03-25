# import numpy as np
import logging
import argparse
import os 



import sys
sys.path.append("..")
from utils import *
sys.path.append("./calix_thesis")


"""
Convert the boundaries in an htk output file from htk ticks to seconds.
"""
def translate_htk(file_name):
    #go with sil -> empty for now
    mapping = {'a': 'pick', 'e': 'carry_item', 'i': 'place', 'm': 'carry_empty', 'sil': 'empty'}
    data = []
    htk_boundaries = defaultdict(list)
    with open(file_name, 'r') as f:
        lines = f.readlines()
        for line in lines[2:]:
            if line.strip() == ".":
                # terminating char
                break
            line = line.strip().split()
            data.append([mapping[line[2]], int(int(line[0]) / HTK_TO_FRAME_RATIO), int(int(line[1]) / HTK_TO_FRAME_RATIO)])
    return data


    return htk_boundaries
if __name__ == "__main__":
    quit()

    logging.basicConfig(level=logging.DEBUG)


    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=str, default="../configs/calix.yaml",
                        help="Path to experiment config (scripts/configs)")
    args = parser.parse_args()
    configs = load_yaml_config(args.config)
    htk_output_folder = configs["file_paths"]["htk_output_file_path"]

    # picklists that we are looking at
    # PICKLISTS = list(range(136, 224)) + list(range(225, 230)) + list(range(231, 235))
    PICKLISTS = list(range(137, 234)) # list(range(136, 235)
    # PICKLISTS = [136, 137, 138]



    # initialization (randomly assign colors)
    for picklist_no in PICKLISTS:
        logging.debug(f"Picklist number {picklist_no}")
        htk_results_file = "{}/picklist_{}.txt".format(htk_output_folder, picklist_no)
        frame_boundaries = translate_htk(htk_results_file)
        write_data = "\n".join([' '.join([str(x) for x in line]) for line in frame_boundaries])

        with open(htk_results_file, 'w+') as outfile:
            outfile.write(write_data)
        # print(frame_boundaries)
