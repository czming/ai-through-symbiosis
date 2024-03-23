"""

iterative improvement method that swaps pairs which result in smaller intracluster distance and simulated annealing
when there is no pair which would reduce the intracluster distance

This version of the script trains the clustering obj classification model on "good" quality frames as determined by Calix's scoring metric.
"""

# from utils import *
import argparse
import math
import matplotlib.pyplot as plt
from sklearn import metrics
import copy
import logging
from sklearn.utils.multiclass import unique_labels
import pickle

from util import parse_action_label_file

import sys
sys.path.append("..")
from utils import *
sys.path.append("./calix_thesis")

np.random.seed(42)

logging.basicConfig(level=logging.DEBUG)


parser = argparse.ArgumentParser()
parser.add_argument("--config", "-c", type=str, default="../configs/calix.yaml",
                    help="Path to experiment config (scripts/configs)")
args = parser.parse_args()

configs = load_yaml_config(args.config)

elan_label_folder = configs["file_paths"]["elan_annotated_file_path"]
htk_input_folder = configs["file_paths"]["htk_input_file_path"]
video_folder = configs["file_paths"]["video_file_path"]
pick_label_folder = configs["file_paths"]["label_file_path"]

htk_output_folder = configs["file_paths"]["htk_output_file_path"]

# picklists that we are looking at
# PICKLISTS = list(range(136, 224)) + list(range(225, 230)) + list(range(231, 235))
PICKLISTS = list(range(136, 235)) # list(range(136, 235)
# PICKLISTS = [136, 137, 138]

# controls the number of bins that we are looking at (180 looks at all 180 hue bins, 280 looks at 180 hue bins +
# 100 saturation bins)
num_bins = 20

actual_picklists = {}
predicted_picklists = {}
picklists_w_symmetric_counts = set()

avg_hsv_bins_combined = {}

# stores the objects
objects_avg_hsv_bins = []
classification_objects_avg_hsv_bins = []

# {picklist_no: [index in objects_avg_hsv_bins for objects that are in this picklist]}
picklist_objects = defaultdict(lambda: set())

# {object type: [index in objects_avg_hsv_bins for objects that are predicted to be of that type]
pred_objects = defaultdict(lambda: set())

objects_pred = {}

combined_pick_labels = []

# initialization (randomly assign colors)
for picklist_no in PICKLISTS:
    logging.debug(f"Picklist number {picklist_no}")

    with open(f"{pick_label_folder}/picklist_{picklist_no}_raw.txt") as infile:
                pick_labels = [i for i in infile.read().replace("\n", "")[::2]]

    with open(f"{htk_input_folder}/picklist_{picklist_no}.txt") as infile:
        htk_inputs = [i.split() for i in infile.readlines()]
        # print(htk_inputs)

    # frame_boundaries = parse_action_label_file(os.path.join(htk_output_folder, 'picklist_{}.txt'.format(picklist_no)))
    try:
        htk_results_file = f"{htk_output_folder}/results-" + str(picklist_no)
        print(htk_results_file)
        htk_boundaries = get_htk_boundaries(htk_results_file)
        # print(htk_boundaries)
    except:
        # no labels yet
        print("Skipping picklist: No htk boundaries")
        continue


    # get the htk_input to load the hsv bins from the relevant lines
    with open(f"{htk_input_folder}/picklist_{picklist_no}") as infile:
        htk_inputs = [i.split() for i in infile.readlines()]

    # get the average hsv bins for each carry action sequence (might want to incorporate information from pick since
    # that should give some idea about what the object is as well)

    # for ELAN labels
    # pick_labels = ["carry_red",
    #                "carry_blue",
    #                "carry_green",
    #                "carry_darkblue",
    #                "carry_darkgreen",
    #                "carry_clear",
    #                "carry_alligatorclip",
    #                "carry_yellow",
    #                "carry_orange",
    #                "carry_candle",
    #                ]

    # print(elan_boundaries)
    print(htk_boundaries)
    # htk label for pick
    action_labels = ["carry_item", "carry_empty", "pick", "place"]
    # using (predicted) htk boundaries instead of elan boundaries
    all_action_frames = []
    action_averages = []
    for action_label in action_labels:
        # look through each color
        action_frames = []

        for i in range(0, len(htk_boundaries[action_label]), 2):
            # collect the red frames
            start_frame = math.ceil(float(htk_boundaries[action_label][i]) * 29.97)
            end_frame = math.ceil(float(htk_boundaries[action_label][i + 1]) * 29.97)
            action_frames.append([start_frame, end_frame])
        # sort based on start
        action_frames = sorted(action_frames, key=lambda x: x[0])
        print(action_frames)
        logging.debug(len(action_frames))
        for i in range(len(action_frames) - 1):
            if action_frames[i + 1][0] <= action_frames[i][1]:
                # start frame of subsequent pick is at or before the end of the current pick (there's an issue)
                raise Exception("pick timings are overlapping, check data")

        # avg hsv bins for each pick
        avg_hand_hsv = [get_avg_hsv_bin_frames(htk_inputs, start_frame + 10, end_frame - 10)[0] for (start_frame, end_frame) \
                            in action_frames]
        # if 0 in num_hand_detections:
        #     print("skipping bad boundaries")
        #     continue
        filtered_arr_list = [arr for arr in avg_hand_hsv if not np.isnan(arr).all()]
        avg_arr = np.mean(filtered_arr_list, axis=0)
        print(action_label)
        print(avg_arr)
        all_action_frames.append(action_frames)
        action_averages.append(avg_arr)

        # plt.bar(range(20), avg_arr)
        # plt.show()

    """
    elan_action_frame_times = []
    for boundary_list in elan_boundaries.values():
        for boundary_value in boundary_list:
            elan_action_frame_times.append(math.ceil(float(boundary_value) * 29.97))
    elan_action_frame_times = set(elan_action_frame_times)
    """
        

    # get distances (4 per frame) between each frame and each action average and plot

    print(len(htk_inputs))
    print(len(action_averages))

    distances = [[],[],[],[]]
    min_dist_indices = []
    for htk_input in htk_inputs:
        hsv = np.array(htk_input[1:21], dtype=float)
        min_dist = float('inf')
        for action_index, action_avg in enumerate(action_averages):
            action_dist = np.linalg.norm(hsv - action_avg)
            if action_dist < min_dist:
                min_dist = action_dist
                min_dist_index = action_index
        if min_dist_index == 0 or min_dist_index == 2:
            min_dist_indices.append(3)
        else:
            min_dist_indices.append(1)
        # min_dist_indices.append(min_dist_index)
        for action_index, action_avg in enumerate(action_averages):
            action_dist = np.linalg.norm(hsv - action_avg)
            if action_index == min_dist_index:
                distances[action_index].append(action_dist)
            else:
                distances[action_index].append(0)
            

    action_frame_times = sorted([v for sublist in all_action_frames for subsublist in sublist for v in subsublist])

    """
    # Create a figure and axis object
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize=(10, 8))


    # Plot the lines
    ax1.plot(distances[0], label='Pick')
    ax1.plot(distances[1], label='Carry')
    ax1.plot(distances[2], label='Place')
    ax1.plot(distances[3], label='Carry Empty')

    # for v in action_frame_times:
    #     ax.plot([v]*2, [0, 1], color='black', linewidth=2)

    for v in elan_action_frame_times:
        ax1.plot([v]*2, [0, 1], color='blue', linewidth=2)

    # Set axis labels and legend
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Distance')
    ax1.legend()


    # Plot the lines
    ax2.plot(distances[0], label='Pick')
    ax2.plot(distances[1], label='Carry')
    ax2.plot(distances[2], label='Place')
    ax2.plot(distances[3], label='Carry Empty')
    for v in action_frame_times:
        ax2.plot([v]*2, [0, 1], color='black', linewidth=2)
    # Set axis labels and legend
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Distance')
    ax2.legend()


    ax3.plot(min_dist_indices, label='action index')
    # for v in action_frame_times:
    #     ax3.plot([v]*2, [0, 3], color='black', linewidth=2)
    print("here")
    elan_action_frame_times = list(elan_action_frame_times)[1:-1]
    print(len(elan_action_frame_times))
    print(len(action_frame_times))
    print(action_frame_times)

    for v in elan_action_frame_times:
        ax3.plot([v]*2, [0, 3], color='blue', linewidth=2)
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Distance')


    # Show the plot
    plt.show()
    # print(all_action_frames[0])
    # with open("picklist_" + str(picklist_no), "w") as file:

    # # Loop over the min_dist_indices
    #     for i in min_dist_indices:
    #     # Convert the type of i to a string and write it to the file
    #         file.write(str(i) + "\n")
"""