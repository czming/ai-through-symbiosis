"""

simple method to test if we can use hsv bins to match the order of the labels (uses the inter and intra class
distance between the hsv bins given a particular permutation and see which one gets the best results), determined
using silouhette coefficient and euclidean distance between the points as the distance metric

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

import pickle


parser = argparse.ArgumentParser()
parser.add_argument("--config", "-c", type=str, default="configs/zm.yaml", help="Path to experiment config (scripts/configs)")
args = parser.parse_args()


configs = load_yaml_config(args.config)

elan_label_folder = configs["file_paths"]["elan_annotated_file_path"]
htk_input_folder = configs["file_paths"]["htk_input_file_path"]
video_folder = configs["file_paths"]["video_file_path"]
pick_label_folder = configs["file_paths"]["label_file_path"]


htk_output_folder = configs["file_paths"]["htk_output_file_path"]

# picklists that we are looking at
# PICKLISTS = list(range(136, 224)) + list(range(225, 230)) + list(range(231, 235))
PICKLISTS = list(range(136, 235))

num_bins = 180

actual_picklists = {}
predicted_picklists = {}
picklists_w_symmetric_counts = set()

avg_hsv_bins_combined = {}

# accumulates the hsv_bins for the different objects and stores the number of counts so we can get the average
# hsv bin across the different picklists
objects_hsv_bin_accumulator = defaultdict(lambda: [np.zeros(shape=(num_bins,)), 0])

total_picks = 0

total_incorrect = 0

total_order_count = 0

total_time = 0

total_picklists = 0

num_permutations_total = 0



for picklist_no in PICKLISTS:
    print (f"Picklist number {picklist_no}")
    try:
        # load elan boundaries, so we can take average of each picks elan labels
        elan_label_file = f"{elan_label_folder}/picklist_{picklist_no}.eaf"
        elan_boundaries = get_elan_boundaries(elan_label_file)
        total_time += elan_boundaries["carry_empty"][-1]
    except:
        # no labels yet
        print("Skipping picklist: No elan boundaries")
        continue
    try:
        htk_results_file = f"{htk_output_folder}/results-" + str(picklist_no)
        htk_boundaries = get_htk_boundaries(htk_results_file)
        # print(htk_boundaries)
    except:
        # no labels yet
        print("Skipping picklist: No htk boundaries")
        continue

    total_picklists += 1

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


    avg_hsv_picks = [get_avg_hsv_bin_frames(htk_inputs, start_frame, end_frame)[0] - avg_empty_hand_hsv for (start_frame, end_frame) \
                        in pick_frames]

    # print (avg_hsv_picks)

    avg_hsv_bins_combined[picklist_no] = avg_hsv_picks

    with open(f"{pick_label_folder}/picklist_{picklist_no}_raw.txt") as infile:
        pick_labels = [i for i in infile.read().replace("\n", "")[::2]]
    with open(f"{pick_label_folder}/picklist_{picklist_no}_raw.txt") as infile:
        bin_labels = [i for i in infile.read().replace("\n", "")[1::2]]

    # order is the number of unique bins filled per picklist
    order_count = len(set(bin_labels))

    total_order_count += order_count

    # check if there are two objects with the same count
    object_count_set = set()
    pick_label_count = dict(Counter(pick_labels))
    symmetric_count = False

    completely_symmetric_count = False
    partially_symmetric_count = False

    lst = pick_label_count.values()
    counts = Counter(lst)
    unique_counts = [x for x in counts if counts[x] == 1]
    if len(unique_counts) == 0:
        completely_symmetric_count = True

    elif sum(unique_counts) < len(pick_labels):
        partially_symmetric_count = True

    # print("Completely Symmetric: " + str(completely_symmetric_count))
    # print("Partially Symmetric: " + str(partially_symmetric_count))

    all_permutations = []

    generate_permutations_from_dict(pick_label_count, all_permutations)

    num_permutations_total += len(all_permutations)

    # map the colors to an index where the vectors will be appended
    color_index_mapping = {value: index for index, value in enumerate(list(set(pick_labels)))} # {'r': 0, 'g': 1, 'b': 2}


    # store the lowest beta_cv measure permutation
    best_result = (float('inf'), None)
    for permutation_index in range(len(all_permutations)):
        # iterate through the different permutations
        permutation = all_permutations[permutation_index]
        # gather the hsv bin average into clusters
        hsv_color_cluster = [[] for i in range(len(pick_labels))]

        for index, value in enumerate(permutation):
            # add the hsv bins based on their colors assigned under this permutation
            hsv_color_cluster[color_index_mapping[value]].append(avg_hsv_picks[index])
        # print(hsv_color_cluster)
        curr_result = beta_cv(hsv_color_cluster)

        if curr_result < best_result[0]:
            best_result = (curr_result, permutation)
    actual_picklists[picklist_no] = pick_labels
    predicted_picklists[picklist_no] = best_result[1]


    count_mapping = get_count_mapping(predicted_picklists[picklist_no])

    total_picks += len(avg_hsv_picks)

    # print (count_mapping)

    for count, curr_objects in count_mapping.items():
        # looking at the different objects count
        if len(curr_objects) != 1:
            # continue since there are multiple objects with the same counts so cannot determine anything
            picklists_w_symmetric_counts.add(picklist_no)
            print (count, curr_objects)
            continue
        else:
            for i in range(len(predicted_picklists[picklist_no])):
                if predicted_picklists[picklist_no][i] == curr_objects[0]:
                    # looking only at the current object type (should have only one object)
                    # add the avg hsv bin of each pick to the accumulator if there's no symmetric count
                    pred_label = best_result[1][i]
                    # add the hsv bin to the predicted label's bin and increment the count
                    objects_hsv_bin_accumulator[pred_label][0] += avg_hsv_picks[i]
                    objects_hsv_bin_accumulator[pred_label][1] += 1


    # predicted already, non-symmetric
    print("Actual:    " + str(pick_labels))
    print("Predicted: " + str(best_result[1]))

    # current assumption needs every color appears at least once in a picklist without symmetric count

# don't want to keep adding otherwise it would bias towards the symmetric picks
store_objects_hsv_bin_accumulator = copy.deepcopy(objects_hsv_bin_accumulator)

swapped = True
swap_run_count = 0

while swapped and swap_run_count <= 5:
    swap_run_count += 1
    swapped = False

    for picklist_no in picklists_w_symmetric_counts:
        # accumulate bins for the current hsv bin, likewise it's (hsv bin avg, count)

        print ("Picklist " + str(picklist_no))
        print("Prev Predicted:    " + str(predicted_picklists[picklist_no]))

        curr_picklist_hsv_bin_accumulator = defaultdict(lambda: [np.zeros(shape=(num_bins,)), 0])

        for index, object in enumerate(predicted_picklists[picklist_no]):
            # accumulator to get the average hsv bin for the current picklist (get the average for each pick from the
            # ones computed earlier)
            curr_picklist_hsv_bin_accumulator[object][0] += avg_hsv_bins_combined[picklist_no][index]
            curr_picklist_hsv_bin_accumulator[object][1] += 1

        # get the count: [objects] mapping then go through each count, using predicted picklist for the counts since
        # should have been enforced above
        count_mapping = get_count_mapping(predicted_picklists[picklist_no])

        for count, curr_objects in count_mapping.items():
            # only focus on objects that are in the current count bin

            if len(curr_objects) == 1:
                # not symmetric
                continue


            # get the avg hsv bins
            curr_picklist_avg_hsv_bins = {key: value[0] / value[1] for key, value in curr_picklist_hsv_bin_accumulator.items() \
                                            if key in curr_objects}
            object_avg_hsv_bins = {key: value[0] / value[1] for key, value in objects_hsv_bin_accumulator.items() \
                                        if key in curr_objects}

            # fix the arrangement of the objects that we are going to draw from the main hsv bins, now want to find the permutation
            # that matches this the best then can assign
            objects_in_picklist = list(curr_picklist_avg_hsv_bins.keys())


            # --------------should work on those with the same counts as a group then ignore the rest each time------------

            all_permutations = []

            # 1 count for each object type in the picklist
            generate_permutations_from_dict({i: 1 for i in objects_in_picklist}, all_permutations)


            # print (object_avg_hsv_bins)
            # print (all_permutations)

            # store the lowest beta_cv measure permutation
            best_result = (float('inf'), None)

            for permutation in all_permutations:
                # from main one, get the hsv bin based on teh objects_in_picklist order while for the curr_picklist, look
                # at the permutation
                hsv_bin_clusters = [[object_avg_hsv_bins[objects_in_picklist[i]], curr_picklist_avg_hsv_bins[permutation[i]]]
                                    for i in range(len(permutation))]

                curr_result = beta_cv(hsv_bin_clusters)

                if curr_result < best_result[0]:
                    best_result = (curr_result, permutation)

            # map from the previous label (in symmetric picklist, best result) to the corrected label (actual one from non-symmetric)
            object_mapping = {best_result[1][i]: objects_in_picklist[i] for i in range(len(objects_in_picklist))}

            for key, value in object_mapping.items():
                if key != value:
                    # there was movement in the mapping
                    swapped = True

            for i in range(len(predicted_picklists[picklist_no])):
                if predicted_picklists[picklist_no][i] not in object_mapping.keys():
                    # not mapping this element in this iteration
                    continue

                # get the corrected label
                corrected_label = object_mapping[predicted_picklists[picklist_no][i]]
                # predicted_picklists[picklist_no][i] = corrected_label



                # to learn from symmetric picks
                # # add this now assigned pick to the average bin accumulator for the relavent color
                objects_hsv_bin_accumulator[corrected_label][0] += avg_hsv_bins_combined[picklist_no][i]
                objects_hsv_bin_accumulator[corrected_label][1] += 1



        print("Updated Predicted: " + str(predicted_picklists[picklist_no]))
        print("Actual:            " + str(actual_picklists[picklist_no]))

    # fix the ground truth arrangement, e.g. rgb, then permute the labels for the current picklist e.g. rgb, bgr, brg
    # then put both the vectors from the current picklist and vector from the non-symmetric picklist and pick the
    # arrangement that gives the best match

# flatten arrays
actual_picklists = [i for j in actual_picklists.values() for i in j]
predicted_picklists = [i for j in predicted_picklists.values() for i in j]

confusions = defaultdict(int)
label_counts = defaultdict(int)
for pred, label in zip(predicted_picklists, actual_picklists):
    if pred != label:
        confusions[pred + label] += 1

print(confusions)

# print (objects_hsv_bin_accumulator)


# TODO: regather the bins and compute the average since now this would overweight the symmetric picks
with open("objects_hsv_bin_accumulator.pkl", "wb") as outfile:
    # without removing the hand
    pickle.dump(dict(objects_hsv_bin_accumulator), outfile)

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

print (f"Picklist count: {total_picklists}")
print (f"Order count: {total_order_count}")
print (f"Total time: {total_time}")
print (f"Num permutations total: {num_permutations_total}")


