"""

iterative improvement method that swaps pairs which result in smaller intracluster distance and simulated annealing
when there is no pair which would reduce the intracluster distance

"""
import numpy as np

from utils import *
import argparse
import cv2
import math
import matplotlib.pyplot as plt
from collections import Counter
from sklearn import metrics
import copy
import logging

import pickle

np.random.seed(42)


parser = argparse.ArgumentParser()
parser.add_argument("--config", "-c", type=str, default="configs/zm.yaml",
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

# controls the number of bins that we are looking at (180 looks at all 180 hue bins, 280 looks at 180 hue bins +
# 100 saturation bins)
num_bins = 180

actual_picklists = {}
predicted_picklists = {}
picklists_w_symmetric_counts = set()

avg_hsv_bins_combined = {}

# stores the objects
objects_avg_hsv_bins = []

# {picklist_no: [index in objects_avg_hsv_bins for objects that are in this picklist]}
picklist_objects = defaultdict(lambda: set())

# {object type: [index in objects_avg_hsv_bins for objects that are predicted to be of that type]
pred_objects = defaultdict(lambda: set())

objects_pred = {}

combined_pick_labels = []

def gaussian_kernel1D(length, sigma):
    ax = np.linspace(-(length - 1) / 2., (length - 1) / 2., length)
    kernel = np.exp(-0.5 * np.square(ax) / np.square(sigma))
    return kernel

def collapse_hue_bins(hue_bins, centers, sigma):
    # [0, 60, 120] for the centers, need to control for wraparound
    gaussian_kernel = gaussian_kernel1D(180, sigma)
    gaussian_kernel_one_side = gaussian_kernel[len(gaussian_kernel) // 2:]
    bin_sums = [0 for _ in range(len(centers))]
    for center_index, center in enumerate(centers):
        # this one needs to be treated specially otherwise would be summed twice
        bin_sums[center_index] += gaussian_kernel_one_side[0] * hue_bins[center]
        for i in range(1, 60):
            bin_sums[center_index] += gaussian_kernel_one_side[i] * hue_bins[(center + i) % len(hue_bins)]
            bin_sums[center_index] += gaussian_kernel_one_side[i] * hue_bins[center - i]
    return bin_sums

# initialization (randomly assign colors)
for picklist_no in PICKLISTS:
    logging.debug(f"Picklist number {picklist_no}")
    # try:
    #     # load elan boundaries, so we can take average of each picks elan labels
    #     elan_label_file = f"{elan_label_folder}/picklist_{picklist_no}.eaf"
    #     elan_boundaries = get_elan_boundaries(elan_label_file)
    #     total_time += elan_boundaries["carry_empty"][-1]
    # except:
    #     # no labels yet
    #     print("Skipping picklist: No elan boundaries")
    #     continue
    try:
        htk_results_file = f"{htk_output_folder}/results-" + str(picklist_no)
        htk_boundaries = get_htk_boundaries(htk_results_file)
        # print(htk_boundaries)
    except:
        # no labels yet
        print("Skipping picklist: No htk boundaries")
        continue


    # get the htk_input to load the hsv bins from the relevant lines
    with open(f"{htk_input_folder}/picklist_{picklist_no}.txt") as infile:
        htk_inputs = [i.split() for i in infile.readlines()]

    # get the average hsv bins for each carry action sequence (might want to incorporate information from pick since
    # that should give some idea about what the object is as well)

    pick_labels = ["carry_red",
                   "carry_blue",
                   "carry_green",
                   "carry_darkblue",
                   "carry_darkgreen",
                   "carry_clear",
                   "carry_alligatorclip",
                   "carry_yellow",
                   "carry_orange",
                   "carry_candle",
                   ]

    pick_frames = []

    # htk label for pick
    pick_labels = ["e"]
    elan_boundaries = htk_boundaries

    for pick_label in pick_labels:
        # look through each color
        for i in range(0, len(elan_boundaries[pick_label]), 2):
            # collect the red frames
            start_frame = math.ceil(float(elan_boundaries[pick_label][i]) * 29.97)
            end_frame = math.ceil(float(elan_boundaries[pick_label][i + 1]) * 29.97)
            pick_frames.append([start_frame, end_frame])

    # sort based on start
    pick_frames = sorted(pick_frames, key=lambda x: x[0])

    for i in range(len(pick_frames) - 1):
        if pick_frames[i + 1][0] <= pick_frames[i][1]:
            # start frame of subsequent pick is at or before the end of the current pick (there's an issue)
            raise Exception("pick timings are overlapping, check data")

    # avg hsv bins for each pick
    num_hand_detections = [get_avg_hsv_bin_frames(htk_inputs, start_frame, end_frame)[1] for (start_frame, end_frame) \
                           in pick_frames]
    if 0 in num_hand_detections:
        print("skipping bad boundaries")
        continue

    empty_hand_label = "m"

    empty_hand_frames = []

    for i in range(0, len(elan_boundaries[empty_hand_label]), 2):
        # collect the red frames
        start_frame = math.ceil(float(elan_boundaries[empty_hand_label][i]) * 29.97)
        end_frame = math.ceil(float(elan_boundaries[empty_hand_label][i + 1]) * 29.97)
        empty_hand_frames.append([start_frame, end_frame])

    # getting the sum, multiply average by counts
    sum_empty_hand_hsv = np.zeros(num_bins)
    empty_hand_frame_count = 0

    for (start_frame, end_frame) in empty_hand_frames:
        curr_avg_empty_hand_hsv, frame_count = get_avg_hsv_bin_frames(htk_inputs, start_frame, end_frame)
        sum_empty_hand_hsv += (curr_avg_empty_hand_hsv * frame_count)
        empty_hand_frame_count += frame_count

    avg_empty_hand_hsv = sum_empty_hand_hsv / empty_hand_frame_count

    # plt.bar(range(20), avg_empty_hand_hsv)
    # plt.show()

    avg_hsv_picks = [get_avg_hsv_bin_frames(htk_inputs, start_frame, end_frame)[0] - avg_empty_hand_hsv for
                     (start_frame, end_frame) \
                     in pick_frames]

    with open(f"{pick_label_folder}/picklist_{picklist_no}_raw.txt") as infile:
        pick_labels = [i for i in infile.read().replace("\n", "")[::2]]

    combined_pick_labels.extend(pick_labels)

    # randomly assign pick labels for now
    pred_labels = np.random.choice(pick_labels, replace=False, size=len(pick_labels))

    print (pick_labels, pred_labels)

    # must be a divisor of 180
    collapse_bin_size = 60

    # collapse_hue_bins = lambda x: np.array([x[180 - collapse_bin_size//2:180].sum() + x[:collapse_bin_size//2].sum()] + \
    #                                         [x[collapse_bin_size//2 + i * collapse_bin_size: collapse_bin_size//2 + (i + 1) * collapse_bin_size].sum() \
    #                                             for i in range(180 // collapse_bin_size - 1)])
    #
    # collapse_hue_bins = lambda x: np.array([x[150:180].sum() + x[:30].sum(), x[30:90].sum(), x[90:150].sum()])

    # collapse_hue_bins = lambda x: np.array([x[60 * i:60 * (i + 1)].sum() for i in range(3)])

    for i in range(len(avg_hsv_picks)):
        object_id = len(objects_avg_hsv_bins)
        picklist_objects[picklist_no].add(object_id)
        # map object to prediction and prediction to all objects with the same
        pred_objects[pred_labels[i]].add(object_id)
        objects_pred[object_id] = pred_labels[i]
        # overlapping bins so there's no sudden dropoff
        objects_avg_hsv_bins.append(collapse_hue_bins(avg_hsv_picks[i], [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165], 15))

        # plt.bar(range(180), avg_hsv_picks[i])
        #
        # plt.show()
        #
        # plt.bar(range(3), collapse_hue_bins(avg_hsv_picks[i],[0, 60, 120], 20))
        #
        # plt.show()

ax = plt.figure().add_subplot(projection='3d')

color_mapping = {
    'r': '#ff0000',
    'g': '#009900',
    'b': '#000099',
    'p': '#0000ff',
    'q': '#00ff00',
    'o': '#FFA500',
    's': '#000000',
    'a': '#FFEA00',
    't': '#777777',
    'u': '#E1C16E'
}

ax.scatter([i[0] for i in objects_avg_hsv_bins], [i[1] for i in objects_avg_hsv_bins], [i[2] for i in objects_avg_hsv_bins], \
           c = [color_mapping[i] for i in combined_pick_labels])

plt.show()

epochs = 0
num_epochs = 500

cluster_fig = plt.figure()

cluster_ax = cluster_fig.add_subplot(projection="3d")

plt.show(block=False)

while epochs < num_epochs:
    print (f"Starting epoch {epochs}...")
    # for each epoch, iterate through all the picklists and offer one swap
    for picklist_no, objects_in_picklist in picklist_objects.items():
        # check whether swapping the current element with any of the other

        # randomly pick an object
        object1_id = np.random.choice(list(picklist_objects[picklist_no]))

        # stores the reductions for the different object2s
        object2_distance_reduction = {}

        ## TODO: add controls on the number of objects looked at (if limiting the number of objects in the
        # picklist to compare to, just pick a bunch randomly to be looked at)
        for object2_id in objects_in_picklist:
            object1_pred = objects_pred[object1_id]
            object2_pred = objects_pred[object2_id]

            if objects_pred[object2_id] == objects_pred[object1_id]:
                # they objects have the same prediction currently, no point swapping
                continue

            # simple approach, look at the relative distance between the two points and the center of their
            # clusters, don't count themselves
            object1_pred_mean = np.array([objects_avg_hsv_bins[i] for i in pred_objects[object1_pred] if i != object1_id]).mean(axis=0)
            object2_pred_mean = np.array([objects_avg_hsv_bins[i] for i in pred_objects[object2_pred] if i != object2_id]).mean(axis=0)

            object1_pred_std = np.array([objects_avg_hsv_bins[i] for i in pred_objects[object1_pred] if i != object1_id]).std(axis=0, ddof=1)
            object2_pred_std = np.array([objects_avg_hsv_bins[i] for i in pred_objects[object2_pred] if i != object2_id]).std(axis=0, ddof=1)

            # should be proportional to log likelihood (assume normal, then take exponential of these / 2 to get pdf

            # current distance between the objects and their respective means, 0.00001 added for numerical stability
            curr_distance = (((object1_pred_mean - objects_avg_hsv_bins[object1_id]) ** 2) / (object1_pred_std + 0.0000001)).sum(axis=0) + \
                            (((object2_pred_mean - objects_avg_hsv_bins[object2_id]) ** 2) / (object2_pred_std + 0.0000001)).sum(axis=0)

            # distance if they were to swap
            new_distance = (((object2_pred_mean - objects_avg_hsv_bins[object1_id]) ** 2) / (object2_pred_std + 0.0000001)).sum(axis=0) + \
                            (((object1_pred_mean - objects_avg_hsv_bins[object2_id]) ** 2) / (object1_pred_std + 0.0000001)).sum(axis=0)

            distance_reduction = curr_distance - new_distance

            object2_distance_reduction[object2_id] = distance_reduction

        if len(object2_distance_reduction) == 0:
            # no object of different type
            continue

        # keep best distance reduction
        best_pos_object2 = list(object2_distance_reduction.keys())[0]

        for object2, distance_reduction in object2_distance_reduction.items():
            if distance_reduction > object2_distance_reduction[best_pos_object2]:
                # found a better one
                best_pos_object2 = object2



        if object2_distance_reduction[best_pos_object2] < 0:
            object2_distance_reduction_sum = sum(object2_distance_reduction.values())
            # all negative, need to pick one at random with decreasing probability based on the numbers,
            # take exponential of the distance reduction which should all be negative
            swap_object_id = np.random.choice(list(object2_distance_reduction.keys()), p=np.array([math.e ** (i / object2_distance_reduction_sum) for i in
                                        object2_distance_reduction.values()]) / sum([math.e ** (i / object2_distance_reduction_sum) for i in
                                        object2_distance_reduction.values()]))

            # give reducing odds of getting a random swap
            to_swap = np.random.choice([0, 1], p = [1 - math.e ** (-epochs/50), math.e ** (-epochs/50)])

            if not to_swap:
                # to_swap is False so skip the rest of the assignment
                continue


        else:
            # have the most positive distance reduction, just swap
            swap_object_id = best_pos_object2


            # remove randomness towards the end

            # only want to consider those with positive distance reduction
            object2_distance_reduction_positive = {key: value for key, value in object2_distance_reduction.items() if value > 0}

            # use to normalize the distances otherwise can get very small
            object2_distance_reduction_sum_positive = sum([i for i in object2_distance_reduction_positive.values()])

            # definitely swap, but some uncertainty about which element it is swapped with (decreasing temperature by adding epochs
            # as a factor)
            swap_object_id = np.random.choice(list(object2_distance_reduction_positive.keys()), p=np.array([math.e ** (i) for i in
                                        object2_distance_reduction_positive.values()]) / sum([math.e ** (i) for i in
                                        object2_distance_reduction_positive.values()]))


        # swap the objects, update objects_pred and pred_objects
        object1_prev_pred = objects_pred[object1_id]
        swap_object_prev_pred = objects_pred[swap_object_id]

        pred_objects[object1_prev_pred].remove(object1_id)
        pred_objects[swap_object_prev_pred].remove(swap_object_id)

        pred_objects[object1_prev_pred].add(swap_object_id)
        pred_objects[swap_object_prev_pred].add(object1_id)

        objects_pred[object1_id] = swap_object_prev_pred
        objects_pred[swap_object_id] = object1_prev_pred


    epochs += 1

    predicted_picklists = [objects_pred[i] for i in range(len(combined_pick_labels))]

    cluster_ax.clear()

    cluster_ax.scatter([i[0] for i in objects_avg_hsv_bins], [i[1] for i in objects_avg_hsv_bins], [i[2] for i in objects_avg_hsv_bins], \
               c = [color_mapping[i] for i in predicted_picklists])


    cluster_fig.canvas.draw()




plt_display_index = 0
fig, axs = plt.subplots(2, len(pred_objects) // 2)

for object, predicted_objects in pred_objects.items():

    hsv_bins = np.array([objects_avg_hsv_bins[i] for i in predicted_objects]).mean(axis=0)

    if plt_display_index < len(pred_objects) // 2:
        axs[0, plt_display_index].bar(range(len(hsv_bins)), hsv_bins)
        axs[0, plt_display_index].set_title(object)
    else:
        axs[1, plt_display_index - len(pred_objects) // 2].bar(range(len(hsv_bins)), hsv_bins)
        axs[1, plt_display_index - len(pred_objects) // 2].set_title(object)
    plt_display_index += 1


plt.show()





# print the results on a per picklist level
for picklist_no, objects_in_picklist in picklist_objects.items():
    print (f"Picklist no. {picklist_no}")
    picklist_pred = [objects_pred[i] for i in objects_in_picklist]
    print (f"Predicted labels: {picklist_pred}")
    # use the same mapping for combined_pick_labels as object ids
    picklist_gt = [combined_pick_labels[i] for i in objects_in_picklist]
    print (f"Actual labels:    {picklist_gt}")


# flatten arrays
actual_picklists = combined_pick_labels
predicted_picklists = [objects_pred[i] for i in range(len(combined_pick_labels))]

ax = plt.figure().add_subplot(projection='3d')

ax.scatter([i[0] for i in objects_avg_hsv_bins], [i[1] for i in objects_avg_hsv_bins], [i[2] for i in objects_avg_hsv_bins], \
           c = [color_mapping[i] for i in predicted_picklists])

plt.show()

confusions = defaultdict(int)
label_counts = defaultdict(int)
for pred, label in zip(predicted_picklists, actual_picklists):
    if pred != label:
        confusions[pred + label] += 1

print(confusions)


confusion_matrix = metrics.confusion_matrix(actual_picklists, predicted_picklists)
print(actual_picklists)
print(predicted_picklists)
from sklearn.utils.multiclass import unique_labels

letter_to_name = {
    'r': 'red',
    'g': 'green',
    'b': 'blue',
    'p': 'darkblue',
    'q': 'darkgreen',
    'o': 'orange',
    's': 'alligatorclip',
    'a': 'yellow',
    't': 'clear',
    'u': 'candle'
}
names = ['red', 'green', 'blue', 'darkblue', 'darkgreen', 'orange', 'alligatorclip', 'yellow', 'clear', 'candle']

actual_picklists_names = []
predicted_picklists_names = []

letter_counts = defaultdict(lambda: 0)
for letter in actual_picklists:
    name = letter_to_name[letter]
    letter_counts[name] += 1
    actual_picklists_names.append(name)

for index, letter in enumerate(predicted_picklists):
    # print(index)
    predicted_picklists_names.append(letter_to_name[letter])
confusion_matrix = metrics.confusion_matrix(actual_picklists_names, predicted_picklists_names)
conf_mat_norm = (confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis])

unique_names = unique_labels(actual_picklists_names, predicted_picklists_names)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = conf_mat_norm, display_labels = unique_names)

cm_display.plot(cmap=plt.cm.Blues)

plt.xticks(rotation=90)

plt.show()