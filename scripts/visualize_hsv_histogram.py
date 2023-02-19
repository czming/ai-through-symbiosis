"""
visualize the histograms of images (get the images and their corresponding hsv histograms

intended to verify if the HSV histogram is sufficient to differentiate between the different colored objects
"""

from utils import *
import cv2
import math
import matplotlib.pyplot as plt



configs = load_yaml_config("configs/zm.yaml")

elan_label_folder = configs["file_paths"]["elan_annotated_file_path"]
htk_input_folder = configs["file_paths"]["htk_input_file_path"]
video_folder = configs["file_paths"]["video_file_path"]

# picklists that we are looking at
PICKLISTS = range(1, 40)

for picklist_no in PICKLISTS:
    print (f"Picklist number {picklist_no}")
    # load elan boundaries, so we can take average of each picks elan labels
    elan_label_file = f"{elan_label_folder}/picklist_{picklist_no}.eaf"
    elan_boundaries = get_elan_boundaries(elan_label_file)

    # find the frames where carry_red, carry_blue, carry_green are "in effect"

    # get the htk_input to load the hsv bins from the relevant lines
    with open(f"{htk_input_folder}/picklist_{picklist_no}.txt") as infile:
        htk_inputs = [i.split() for i in infile.readlines()]

    red_hsv_bins = avg_hsv_bins(htk_inputs, elan_boundaries, "carry_red")
    blue_hsv_bins = avg_hsv_bins(htk_inputs, elan_boundaries, "carry_blue")
    green_hsv_bins = avg_hsv_bins(htk_inputs, elan_boundaries, "carry_green")

    print (red_hsv_bins)
    print(blue_hsv_bins)
    print(green_hsv_bins)

    fig, axs = plt.subplots(3, 1)

    axs[0].bar(range(20), red_hsv_bins)
    axs[0].set_ylim([0,1])
    axs[1].bar(range(20), blue_hsv_bins)
    axs[1].set_ylim([0, 1])
    axs[2].bar(range(20), green_hsv_bins)
    axs[2].set_ylim([0, 1])

    plt.show()


