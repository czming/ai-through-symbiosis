"""

script to take in raw aruco marker locations and output integer representing net number of pick/place markers
detected with forward filling (if marker was detected in one frame, it will be assumed to be detected in some
number of frames after that as well)

"""

## DO NOT PUSH, changed for the video id but should revert to one based on picklist number in remote (keep the use of
# config file though)

import sys
import argparse
from typing import List
import numpy as np

sys.path.append("..")

from utils import *

# from aruco binning script
# pick bins
pick_bins = [i for i in range(24 * 2)]
place_bins = [i for i in range(24 * 2, 36 * 2)]

# number of frames that the aruco marker is assumed to be detected for
# 0 indicates no persistence (i.e. only store for one frame and that's it
PERSISTANCE = 30

def perform_forward_fill(infile_data: List[str]) -> np.ndarray:
    """
    infile_data: the data in the infile (output of calling infile.readlines()).
    Returns all_lines, a list of strings
    """
    all_lines = []
    for index, line in enumerate(infile_data):
        line = line.split()
        aruco_markers = line[:72]

        # taking into account the forward fill
        aruco_pick_place_counter = 0

        for x_index in range(0, 72, 2):
            x = float(aruco_markers[x_index])
            if x != 0:
                # the x value for the aruco marker is non-zero (used 0 as the placeholder for undetected marker)
                # update the last frame where this aruco marker was detected
                last_frame_detected[x_index] = index

        # go through last_frame_detected and then count the bins again

        for x_index, last_seen in last_frame_detected.items():
            # check if the current x_index has been visited in the last PERSISTENCE number of frames
            if index - PERSISTANCE <= last_seen:
                if x_index in pick_bins:
                    # +1 for pick bin
                    aruco_pick_place_counter += 1
                if x_index in place_bins:
                    # -1 for place bin
                    aruco_pick_place_counter -= 1

        new_line = [str(aruco_pick_place_counter)] + line[72:]
        all_lines.append(list(map(float, new_line)))
    return np.array(all_lines)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", "-c", type=str, default="../configs/zm.yaml",
                        help="Path to experiment config (scripts/configs)")
    args = parser.parse_args()

    configs = load_yaml_config(args.config_file)

    htk_input_folder = configs["file_paths"]["htk_input_file_path"]

    for i in range(136, 235):
        try:
            with open(f"{htk_input_folder}/picklist_{i}.txt") as infile:

                # overwrite output file first

                # track the last frame when the aruco marker was detected (delete the marker if more than the desired number of
                # frames has passed)
                last_frame_detected = {}

                with open(f"{htk_input_folder}/picklist_{i}_forward_filled_{PERSISTANCE}.txt", "w") as outfile:

                    infile = infile.readlines()

                    all_lines = perform_forward_fill(infile)
                    for line in all_lines:
                        outfile.write(" ".join(map(str, line)))
                        outfile.write("\n")

                print (f"picklist_{i}.txt")
        except:
            print (f"no picklist {i}")