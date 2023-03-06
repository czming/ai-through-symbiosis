"""
visualize the histograms of images (get the images and their corresponding hsv histograms

intended to verify if the HSV histogram is sufficient to differentiate between the different colored objects
"""

from utils import *
import cv2
import math
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


def avg_hsv_bins(hsv_inputs, elan_boundaries, action_label):
    # sums up the hsv bins in the frames that are demarcated by the action label in elan_boundaries
    frame_count = 0
    hsv_bin_sum = [0 for i in range(20)]
    print(action_label)
    for i in range(0, len(elan_boundaries[action_label]), 2):
        # collect the red frames
        start_frame = math.ceil(float(elan_boundaries[action_label][i]) * 30)
        end_frame = math.ceil(float(elan_boundaries[action_label][i + 1]) * 30) - 5
        print(start_frame)
        print(end_frame)
        print(len(hsv_inputs))
        for j in range(start_frame, end_frame):
            # see whether the hand was detected (if hand was not detected, all bins would be 0)
            hand_detected = False
            for k in range(len(hsv_bin_sum)):
                # sum up the current values
                hand_detected = hand_detected or float(hsv_inputs[j][k]) != 0
                hsv_bin_sum[k] += float(hsv_inputs[j][k])
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
video_folder = configs["file_paths"]["video_file_path"]

# picklists that we are looking at
PICKLISTS = range(1, 2)

for picklist_no in PICKLISTS:
    print (f"Picklist number {picklist_no}")
    # load elan boundaries, so we can take average of each picks elan labels
    elan_label_file = f"{elan_label_folder}/picklist_{picklist_no}.eaf"
    elan_label_file = '../elan_annotated/GX010044.eaf'
    elan_boundaries = get_elan_boundaries_specific(elan_label_file)

    # find the frames where carry_red, carry_blue, carry_green are "in effect"

    # get the htk_input to load the hsv bins from the relevant lines
    htk_input_file = f"{htk_input_folder}/picklist_{picklist_no}.txt"
    htk_input_file = "../symbiosis/experiments/imu-preliminary/data/picklist_4.txt" # hsv bins
    with open(htk_input_file) as infile:
        htk_inputs = [i.split() for i in infile.readlines()]
    print(htk_inputs)
    print(elan_boundaries)

    all_bins = []

    carry_empty_bins = avg_hsv_bins(htk_inputs, elan_boundaries, "carry_empty")

    red_hsv_bins = avg_hsv_bins(htk_inputs, elan_boundaries, "carry_red")
    all_bins.append(red_hsv_bins)
    blue_hsv_bins = avg_hsv_bins(htk_inputs, elan_boundaries, "carry_blue")
    all_bins.append(blue_hsv_bins)
    green_hsv_bins = avg_hsv_bins(htk_inputs, elan_boundaries, "carry_green")
    all_bins.append(green_hsv_bins)
    yellow_hsv_bins = avg_hsv_bins(htk_inputs, elan_boundaries, "carry_yellow")
    all_bins.append(yellow_hsv_bins)
    clear_hsv_bins = avg_hsv_bins(htk_inputs, elan_boundaries, "carry_clear")
    all_bins.append(clear_hsv_bins)

    darkblue_hsv_bins = avg_hsv_bins(htk_inputs, elan_boundaries, "carry_darkblue")
    all_bins.append(darkblue_hsv_bins)
    alligatorclip_hsv_bins = avg_hsv_bins(htk_inputs, elan_boundaries, "carry_alligatorclip")
    all_bins.append(alligatorclip_hsv_bins)
    darkblue_hsv_bins = avg_hsv_bins(htk_inputs, elan_boundaries, "carry_candle")
    all_bins.append(darkblue_hsv_bins)
    darkgreen_hsv_bins = avg_hsv_bins(htk_inputs, elan_boundaries, "carry_darkgreen")
    all_bins.append(darkgreen_hsv_bins)
    orange_hsv_bins = avg_hsv_bins(htk_inputs, elan_boundaries, "carry_orange")
    all_bins.append(orange_hsv_bins)


    fig, axs = plt.subplots(len(all_bins), 1)
    for index, bins in enumerate(all_bins):
        print(bins)
        print(carry_empty_bins)
        print(np.subtract(np.asarray(bins), np.asarray(carry_empty_bins)))
        axs[index].bar(range(20), np.subtract(np.asarray(bins), np.asarray(carry_empty_bins)))
        axs[index].set_ylim([-.5,.5])
        axs[index].yaxis.set_major_locator(MultipleLocator(0.2))
        axs[index].grid(True, color='black',which='both', axis='y', linestyle=':', linewidth='1')

    # axs[1].bar(range(20), blue_hsv_bins)
    # axs[1].set_ylim([0, 1])
    # axs[2].bar(range(20), green_hsv_bins)
    # axs[2].set_ylim([0, 1])

    plt.show()


