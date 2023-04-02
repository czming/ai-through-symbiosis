"""
visualize the histograms of images (get the images and their corresponding hsv histograms

intended to verify if the HSV histogram is sufficient to differentiate between the different colored objects
"""

from utils import *
import cv2
import math
import matplotlib.pyplot as plt

def avg_hsv_bins(hsv_inputs, elan_boundaries, action_label):
    # sums up the hsv bins in the frames that are demarcated by the action label in elan_boundaries
    frame_count = 0
    hsv_bin_sum = [0 for i in range(20)]
    print(action_label)
    print(elan_boundaries)
    print(len(elan_boundaries[action_label]))
    for i in range(0, len(elan_boundaries[action_label]), 2):
        # collect the red frames
        print(elan_boundaries[action_label])
        start_frame = math.ceil(float(elan_boundaries[action_label][i]) * 59.97)
        end_frame = math.ceil(float(elan_boundaries[action_label][i + 1]) * 59.97)
        # print(start_frame)
        # print(end_frame)
        for j in range(start_frame, end_frame):
            # see whether the hand was detected (if hand was not detected, all bins would be 0)
            hand_detected = False
            for k in range(len(hsv_bin_sum)):
                # sum up the current values
                # print(j)
                # print(k)
                # print(len(hsv_inputs[j]))
                # print(hsv_inputs[j][k + 72])
                hand_detected = hand_detected or float(hsv_inputs[j][k + 72]) != 0
                hsv_bin_sum[k] += float(hsv_inputs[j][k + 72])
            frame_count += hand_detected

    if frame_count == 0:
        # no detected frames
        return hsv_bin_sum

    for k in range(len(hsv_bin_sum)):
        hsv_bin_sum[k] = hsv_bin_sum[k] / frame_count

    return hsv_bin_sum

configs = load_yaml_config("configs/jon.yaml")

elan_label_folder = configs["file_paths"]["elan_annotated_file_path"]
htk_input_folder = configs["file_paths"]["htk_input_file_path"]
htk_output_folder = configs["file_paths"]["htk_output_file_path"]
# video_folder = configs["file_paths"]["video_file_path"]

# picklists that we are looking at
PICKLISTS = range(1, 40)

for picklist_no in PICKLISTS:
    print (f"Picklist number {picklist_no}")
    # load elan boundaries, so we can take average of each picks elan labels
    elan_label_file = f"{elan_label_folder}/picklist_{picklist_no}.eaf"
    elan_boundaries = get_elan_boundaries(elan_label_file)
    print(elan_boundaries)
    htk_label_file = f"{htk_output_folder}/results-{picklist_no}"
    htk_boundaries = get_htk_boundaries(htk_label_file)
    print()
    print(htk_boundaries)

    # find the frames where carry_red, carry_blue, carry_green are "in effect"

    # get the htk_input to load the hsv bins from the relevant lines
    with open(f"{htk_input_folder}/picklist_{picklist_no}.txt") as infile:
        htk_inputs = [i.split() for i in infile.readlines()]

    red_hsv_bins = avg_hsv_bins(htk_inputs, elan_boundaries, "carry_empty")
    blue_hsv_bins = avg_hsv_bins(htk_inputs, elan_boundaries, "pick")
    green_hsv_bins = avg_hsv_bins(htk_inputs, elan_boundaries, "carry")
    hsv_bins = avg_hsv_bins(htk_inputs, elan_boundaries, "place")


    print (red_hsv_bins)
    print(blue_hsv_bins)
    print(green_hsv_bins)

    fig, axs = plt.subplots(4, 1)

    axs[0].bar(range(20), red_hsv_bins)
    axs[0].set_ylim([0,1])
    axs[1].bar(range(20), blue_hsv_bins)
    axs[1].set_ylim([0, 1])
    axs[2].bar(range(20), green_hsv_bins)
    axs[2].set_ylim([0, 1])
    axs[3].bar(range(20), hsv_bins)
    axs[3].set_ylim([0, 1])
    plt.show()


