"""

simple method to test if we can use hsv bins to match the order of the labels (uses the inter and intra class
distance between the hsv bins given a particular permutation and see which one gets the best results), determined
using silouhette coefficient and euclidean distance between the points as the distance metric

"""
import numpy as np

from utils import *
import cv2
import math
import matplotlib.pyplot as plt
from collections import Counter
from sklearn import metrics


def generate_permutations_from_dict(count_dict, all_permutations, curr_permutation=None):
    # generate permutations from d recursively by iterating through the keys and for non-zero keys, add that to the
    # permutation

    total_count = sum([value for value in count_dict.values()])

    if total_count == 0:
        all_permutations.append(list(curr_permutation))

    if curr_permutation == None:
        curr_permutation = []

    for key, value in count_dict.items():
        if value == 0:
            continue
        count_dict[key] -= 1
        curr_permutation.append(key)
        generate_permutations_from_dict(count_dict, all_permutations, curr_permutation)
        count_dict[key] += 1
        curr_permutation.pop()



def sum_point_distance(point, cluster):
    # calculate the sum of the distance between the point and points in the cluster

    total_distance = 0

    for point1 in cluster:
        # distance between point and point1 (point1 from cluster), can replace the distance function here if desired
        total_distance += np.linalg.norm(point - point1)

    return total_distance


def sum_cluster_distance(cluster1, cluster2):
    # return the sum of the distance in the points between the cluster and the number of edges between the two clusters
    # (assumes clusters are different so if they are the same cluster then everything should be scaled by half)

    total_distance = 0
    total_edges = 0

    for point1_index, point1 in enumerate(cluster1):
        # sum distance between all points in clsuter2 and point1
        total_distance += sum_point_distance(point1, cluster2)
        total_edges += len(cluster2)

    return total_distance, total_edges


def beta_cv(clusters):
    """
    calculates betacv measure for clusters of points
    :param clusters: array of points that are in each cluster, [cluster1's points, cluster2's points, ...]
    :return: betacv measure of clustering
    """

    intra_cluster_distance_sum = 0
    inter_cluster_distance_sum = 0

    # compute the number of edges used when summing together inter/intra_cluster_distance
    inter_cluster_edges_count = 0
    intra_cluster_edges_count = 0

    # get total sum of intra class distance
    for cluster_index, cluster in enumerate(clusters):
        distance, num_edges = sum_cluster_distance(cluster, cluster)
        intra_cluster_distance_sum += distance
        # don't double count the same combination (same combination, different permutation)
        intra_cluster_edges_count += num_edges // 2

    # get total sum of inter class distance
    for cluster1_index, cluster1 in enumerate(clusters):
        for cluster2_index in range(cluster1_index + 1, len(clusters)):
            # avoid double counting two clusters against each other
            cluster2 = clusters[cluster2_index]

            distance, num_edges = sum_cluster_distance(cluster1, cluster2)
            inter_cluster_distance_sum += distance
            inter_cluster_edges_count += num_edges

    # both cases there's not much that we can do
    if inter_cluster_edges_count == 0:
        # there's only one color
        return 0

    elif intra_cluster_edges_count == 0:
        # each color has at most one picklist
        return 0

    intra_cluster_distance_mean = intra_cluster_distance_sum / intra_cluster_edges_count

    inter_cluster_distance_mean = inter_cluster_distance_sum / inter_cluster_edges_count

    return intra_cluster_distance_mean / inter_cluster_distance_mean


# 0.5 / (2 / 4) = 1
# print (beta_cv([[1, 2], [2, 2]]))

configs = load_yaml_config("configs/zm.yaml")

elan_label_folder = configs["file_paths"]["elan_annotated_file_path"]
htk_input_folder = configs["file_paths"]["htk_input_file_path"]
video_folder = configs["file_paths"]["video_file_path"]
pick_label_folder = configs["file_paths"]["label_file_path"]


htk_output_folder = configs["file_paths"]["htk_output_file_path"]

# picklists that we are looking at
PICKLISTS = range(76, 77)

actual_picklists = []
predicted_picklists = []
picklists_w_symmetric_counts = []

avg_hsv_bins = []

# accumulates the hsv_bins for the different objects and stores the number of counts so we can get the average
# hsv bin across the different picklists
objects_hsv_bin_accumulator = defaultdict(lambda: [np.zeros(shape=(10,)), 0])

total_incorrect = 0
for picklist_no in PICKLISTS:
    print (f"Picklist number {picklist_no}")
    # load elan boundaries, so we can take average of each picks elan labels
    # elan_label_file = f"{elan_label_folder}/picklist_{picklist_no}.eaf"
    # elan_boundaries = get_elan_boundaries(elan_label_file)
    # print(elan_boundaries)
    try:
        htk_results_file = f"{htk_output_folder}/results-" + str(picklist_no)
        htk_boundaries = get_htk_boundaries(htk_results_file)
        # print(htk_boundaries)
    except:
        # no labels yet
        continue

    # get the htk_input to load the hsv bins from the relevant lines
    with open(f"{htk_input_folder}/picklist_{picklist_no}.txt") as infile:
        htk_inputs = [i.split() for i in infile.readlines()]

    # get the average hsv bins for each carry action sequence (might want to incorporate information from pick since
    # that should give some idea about what the object is as well)
    
    pick_labels = ["carry_red", "carry_blue", "carry_green"]

    pick_frames = []

    # htk label for pick
    pick_labels = ["e"]
    elan_boundaries = htk_boundaries

    for pick_label in pick_labels:
        # look through each color
        for i in range(0, len(elan_boundaries[pick_label]), 2):
            # collect the red frames
            start_frame = math.ceil(float(elan_boundaries[pick_label][i]) * 59.97)
            end_frame = math.ceil(float(elan_boundaries[pick_label][i + 1]) * 59.97)
            pick_frames.append([start_frame, end_frame])

    # sort based on start
    pick_frames = sorted(pick_frames, key=lambda x: x[0])

    for i in range(len(pick_frames) - 1):
        if pick_frames[i + 1][0] <= pick_frames[i][1]:
            # start frame of subsequent pick is at or before the end of the current pick (there's an issue)
            raise Exception("pick timings are overlapping, check data")

    # avg hsv bins for each pick
    try:
        avg_hsv_picks = [get_avg_hsv_bin_frames(htk_inputs, start_frame, end_frame)[0] for (start_frame, end_frame) \
                        in pick_frames]
    except:
        continue

    avg_hsv_bins.append(avg_hsv_picks)

    with open(f"{pick_label_folder}/picklist_{picklist_no}_raw.txt") as infile:
        pick_labels = [i for i in infile.read().replace("\n", "")[::2]]

    # check if there are two objects with the same count (will be integrated later)
    object_count_set = set()
    pick_label_count = dict(Counter(pick_labels))
    symmetric_count = False
    for pick_label, pick_count in pick_label_count.items():
        if pick_count != 0:
            symmetric_count = symmetric_count or (pick_count in object_count_set)
            object_count_set.add(pick_count)

    all_permutations = []

    generate_permutations_from_dict(pick_label_count, all_permutations)

    for i in pick_labels:
        if i not in ['r', 'g', 'b']:
            raise Exception("Unknown pick label: " + i)

    # map the colors to an index where the vectors will be appended
    color_index_mapping = {'r': 0, 'g': 1, 'b': 2}

    # store the lowest beta_cv measure permutation
    best_result = (float('inf'), None)

    for permutation_index in range(len(all_permutations)):
        # iterate through the different permutations
        permutation = all_permutations[permutation_index]
        # gather the hsv bin average into clusters
        hsv_color_cluster = [[] for i in range(3)]
        for index, value in enumerate(permutation):
            # add the hsv bins based on their colors assigned under this permutation
            hsv_color_cluster[color_index_mapping[value]].append(avg_hsv_picks[index])

        curr_result = beta_cv(hsv_color_cluster)

        if curr_result < best_result[0]:
            best_result = (curr_result, permutation)

    print("Actual:    " + str(pick_labels))
    print("Predicted: " + str(best_result[1]))

    actual_picklists.append(pick_labels)
    predicted_picklists.append(best_result[1])

    if symmetric_count:
        # if there's symmetric count then don't add to accumulator
        picklists_w_symmetric_counts.append(picklist_no)
        continue

    for i in range(len(pick_labels)):
        # add the avg hsv bin of each pick to the accumulator if there's no symmetric count
        pred_label = best_result[1][i]
        # add the hsv bin to the predicted label's bin and increment the count
        objects_hsv_bin_accumulator[pred_label][0] += avg_hsv_picks[i]
        objects_hsv_bin_accumulator[pred_label][1] += 1

    # current assumption needs every color appears at least once in a picklist without symmetric count

for picklist_no in picklists_w_symmetric_counts:
    # accumulate bins for the current hsv bin, likewise it's (hsv bin avg, count)
    # decrement by 1 since using as an index
    picklist_index = picklist_no - 1

    curr_picklist_hsv_bin_accumulator = defaultdict(lambda: [np.zeros(shape=(10,)), 0])

    for index, object in enumerate(predicted_picklists[picklist_index]):
        curr_picklist_hsv_bin_accumulator[object][0] += avg_hsv_bins[picklist_index][index]
        curr_picklist_hsv_bin_accumulator[object][1] += 1

    # get the avg hsv bins
    curr_picklist_avg_hsv_bins = {key: value[0] / value[1] for key, value in curr_picklist_hsv_bin_accumulator.items()}
    object_avg_hsv_bins = {key: value[0] / value[1] for key, value in objects_hsv_bin_accumulator.items()}

    # fix the arrangement of the objects that we are going to draw from the main hsv bins, now want to find the permutation
    # that matches this the best then can assign
    objects_in_picklist = list(curr_picklist_avg_hsv_bins.keys())

    all_permutations = []

    # 1 count for each object type in the picklist
    generate_permutations_from_dict({i: 1 for i in objects_in_picklist}, all_permutations)

    print (picklist_no)
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

    # print (object_mapping)
    print("Actual:            " + str(actual_picklists[picklist_index]))
    print("Prev Predicted:    " + str(predicted_picklists[picklist_index]))

    for i in range(len(predicted_picklists[picklist_index])):
        # get the corrected label
        predicted_picklists[picklist_index][i] = object_mapping[predicted_picklists[picklist_index][i]]

    print("Updated Predicted: " + str(predicted_picklists[picklist_index]))

    # fix the ground truth arrangement, e.g. rgb, then permute the labels for the current picklist e.g. rgb, bgr, brg
    # then put both the vectors from the current picklist and vector from the non-symmetric picklist and pick the
    # arrangement that gives the best match

# flatten arrays
actual_picklists = [i for j in actual_picklists for i in j]
predicted_picklists = [i for j in predicted_picklists for i in j]

confusions = defaultdict(int)
label_counts = defaultdict(int)
for pred, label in zip(predicted_picklists, actual_picklists):
    if pred != label:
        confusions[pred + label] += 1

print(confusions)

confusion_matrix = metrics.confusion_matrix(actual_picklists, predicted_picklists)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = ["blue", "green", "red"])
cm_display.plot()
plt.show()
