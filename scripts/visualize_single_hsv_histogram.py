"""
visualize the histograms of images (get the images and their corresponding hsv histograms

intended to verify if the HSV histogram is sufficient to differentiate between the different colored objects
"""

from utils import *
import argparse
import cv2
import math
import os

import matplotlib.pyplot as plt


def avg_hsv_bins(hsv_inputs, elan_boundaries, action_label):
    # sums up the hsv bins in the frames that are demarcated by the action label in elan_boundaries
    frame_count = 0
    hsv_bin_sum = [0 for i in range(20)]
    for i in range(0, len(elan_boundaries[action_label]), 2):
        # collect the red frames
        start_frame = math.ceil(float(elan_boundaries[action_label][i]) * 59.97)
        end_frame = math.ceil(float(elan_boundaries[action_label][i + 1]) * 59.97)
        for j in range(start_frame, end_frame):
            # see whether the hand was detected (if hand was not detected, all bins would be 0)
            hand_detected = False
            for k in range(len(hsv_bin_sum)):
                # sum up the current values
                hand_detected = hand_detected or float(hsv_inputs[j][k + 72]) != 0
                hsv_bin_sum[k] += float(hsv_inputs[j][k + 72])
            frame_count += hand_detected

    if frame_count == 0:
        # no detected frames
        return hsv_bin_sum

    for k in range(len(hsv_bin_sum)):
        hsv_bin_sum[k] = hsv_bin_sum[k] / frame_count

    return hsv_bin_sum




configs = load_yaml_config("configs/zm.yaml")

elan_label_folder = configs["file_paths"]["elan_annotated_file_path"]
htk_input_folder = configs["file_paths"]["htk_input_file_path"]
video_folder = configs["file_paths"]["video_file_path"]

def plot_hsv_histogram(htk_input_file):
    with open(htk_input_file) as infile:
        hsv_bins = [i.split() for i in infile.readlines()][0]


    print(hsv_bins)

    # fig, axs = plt.subplots(1, 1)

    hsv_bins = [float(i) for i in hsv_bins]
    # ax1 = plt.subplot(131)
    print(len(hsv_bins))
    plt.ylim([0, 1])
    plt.bar(range(20), hsv_bins)
    print(os.path.basename(htk_input_file))
    print(os.path.basename(htk_input_file).index("."))
    print(os.path.basename(htk_input_file)[0:os.path.basename(htk_input_file).index(".")])
    plt.savefig(os.path.basename(htk_input_file)[0:os.path.basename(htk_input_file).index(".")])
    
    # plt.show()


if __name__ == "__main__":

    # choose the index of the columns that we want to visualize

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--htk_input_file",
        required=True,
        help="Relative or absolute path to a single data file with 20 HSV columns",
    )
    args = parser.parse_args()
    plot_hsv_histogram(args.htk_input_file)
