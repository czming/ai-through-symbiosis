"""
visualize the histograms of images (get the images and their corresponding hsv histograms

intended to verify if the HSV histogram is sufficient to differentiate between the different colored objects
"""

from utils import *
import cv2
import math
import matplotlib.pyplot as plt
import numpy as np

def avg_hsv_bins(hsv_inputs, htk_boundaries, action_label):
    # sums up the hsv bins in the frames that are demarcated by the action label in elan_boundaries
    frame_count = 0
    hsv_bin_sum = [0 for i in range(20)]
    for i in range(0, len(htk_boundaries[action_label]), 2):
        # collect the red frames

        start_frame = math.ceil(float(htk_boundaries[action_label][i]) * 59.97)
        end_frame = math.ceil(float(htk_boundaries[action_label][i + 1]) * 59.97) - 2 # last frame was giving index out of bounds, and lossing 1 frame wont impact histogram
        for j in range(start_frame, end_frame):
            # see whether the hand was detected (if hand was not detected, all bins would be 0)
            hand_detected = False
            a = np.asarray(hsv_inputs[j][0:20]) # 
            a = a.astype(float)
            all_zeros = not np.any(a)
            try:
                hsv_inputs[j][0] = hsv_inputs[j][0]
            except:
                print("IM trying to make this as noticeable as possible")
            for k in range(len(hsv_bin_sum)):
                # sum up the current values
                try:
                    hand_detected = hand_detected or float(hsv_inputs[j][k + 0]) != 0
                    hsv_bin_sum[k] += float(hsv_inputs[j][k + 0])
                except:
                    continue
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
# PICKLISTS = range(1, 41)
PICKLISTS = range(1, 41)

accumulated = {
    'red':[],
    'green':[],
    'blue':[]
}
out_of_place = 0
for picklist_no in PICKLISTS:
    # print (f"Picklist number {picklist_no}")
    # load elan boundaries, so we can take average of each picks elan labels
    elan_label_file = f"{elan_label_folder}/picklist_{picklist_no}.eaf"
    elan_boundaries = get_elan_boundaries(elan_label_file)

    htk_label_file = f"{htk_output_folder}/results-{picklist_no}"
    htk_boundaries, out_of_place_pick = get_htk_boundaries(htk_label_file)
    out_of_place += out_of_place_pick
    # find the frames where carry_red, carry_blue, carry_green are "in effect"

    # get the htk_input to load the hsv bins from the relevant lines
    # print(f"{htk_input_folder}/picklist_{picklist_no}.txt")
    with open(f"{htk_input_folder}/picklist_{picklist_no}.txt") as infile:
    # with open("./test.txt") as infile:
        htk_inputs = [i.split() for i in infile.readlines()]

    carry_empty_bins = avg_hsv_bins(htk_inputs, htk_boundaries, "carry_empty")
    red_hsv_bins = avg_hsv_bins(htk_inputs, htk_boundaries, "f") # red
    green_hsv_bins = avg_hsv_bins(htk_inputs, htk_boundaries, "h") # green
    blue_hsv_bins = avg_hsv_bins(htk_inputs, htk_boundaries, "g") # blue

    no_red_objects = not np.any(red_hsv_bins)
    no_green_objects = not np.any(green_hsv_bins)
    no_blue_objects = not np.any(blue_hsv_bins)

    subtracted_red = np.asarray(red_hsv_bins) - np.asarray(carry_empty_bins)
    subtracted_green = np.asarray(green_hsv_bins) - np.asarray(carry_empty_bins)
    subtracted_blue = np.asarray(blue_hsv_bins) - np.asarray(carry_empty_bins)

    # if np.sum(subtracted_blue) == 0:

    if not no_red_objects:
        accumulated['red'].append(subtracted_red[0:10])
    if not no_green_objects:
        accumulated['green'].append(subtracted_green[0:10])
    if not no_blue_objects:
        accumulated['blue'].append(subtracted_blue[0:10])

    fig, axs = plt.subplots(4, 1)
    fig.tight_layout(pad=1.0)

    axs[0].bar(range(20), np.asarray(carry_empty_bins) - np.asarray(carry_empty_bins))
    axs[0].set_ylim([0, .15])
    axs[0].set_xlabel("Empty Hand - Empty Hand")
    axs[1].bar(range(20), subtracted_red)
    axs[1].set_ylim([0, .15])
    axs[1].set_xlabel("Red - Empty Hand")
    axs[2].bar(range(20), subtracted_green)
    axs[2].set_ylim([0, .15])
    axs[2].set_xlabel("Green - Empty Hand")
    axs[3].bar(range(20), subtracted_blue)
    axs[3].set_ylim([0, .15])
    axs[3].set_xlabel("Blue - Empty Hand")
    plt.show()


blue_mean = np.mean(np.asarray(accumulated['blue']), axis=0)
red_mean = np.mean(np.asarray(accumulated['red']), axis=0)
green_mean = np.mean(np.asarray(accumulated['green']), axis=0)



red_values = np.asarray(accumulated['red'])
red_means = np.tile(red_mean, (red_values.shape[0], 1))
red_error = np.subtract(red_values, red_means)
red_rmse = np.sum(np.square(red_error))

green_values = np.asarray(accumulated['green'])
green_means = np.tile(green_mean, (green_values.shape[0], 1))
green_error = np.subtract(green_values, green_means)
green_rmse = np.sum(np.square(green_error))

blue_values = np.asarray(accumulated['blue'])
blue_means = np.tile(blue_mean, (blue_values.shape[0], 1))
blue_error = np.subtract(blue_values, blue_means)
blue_rmse = np.sum(np.square(blue_error))

print(red_rmse + blue_rmse + green_rmse, out_of_place)

# # Calculate RMSE
# for picklist_no in PICKLISTS:
#     print (f"Picklist number {picklist_no}")
#     # load elan boundaries, so we can take average of each picks elan labels
#     elan_label_file = f"{elan_label_folder}/picklist_{picklist_no}.eaf"
#     elan_boundaries = get_elan_boundaries(elan_label_file)

#     htk_label_file = f"{htk_output_folder}/results-{picklist_no}"
#     htk_boundaries = get_htk_boundaries(htk_label_file)

#     # find the frames where carry_red, carry_blue, carry_green are "in effect"

#     # get the htk_input to load the hsv bins from the relevant lines
#     print(f"{htk_input_folder}/picklist_{picklist_no}.txt")
#     with open(f"{htk_input_folder}/picklist_{picklist_no}.txt") as infile:
#     # with open("./test.txt") as infile:
#         htk_inputs = [i.split() for i in infile.readlines()]

#     carry_empty_bins = avg_hsv_bins(htk_inputs, htk_boundaries, "carry_empty")
#     red_hsv_bins = avg_hsv_bins(htk_inputs, htk_boundaries, "f") # red
#     green_hsv_bins = avg_hsv_bins(htk_inputs, htk_boundaries, "h") # green
#     blue_hsv_bins = avg_hsv_bins(htk_inputs, htk_boundaries, "g") # blue




