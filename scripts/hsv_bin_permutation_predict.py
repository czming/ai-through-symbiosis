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

# picklists that we are looking at
PICKLISTS = range(1, 60)

for picklist_no in PICKLISTS:
    print (f"Picklist number {picklist_no}")
    # load elan boundaries, so we can take average of each picks elan labels
    elan_label_file = f"{elan_label_folder}/picklist_{picklist_no}.eaf"
    elan_boundaries = get_elan_boundaries(elan_label_file)

    # get the htk_input to load the hsv bins from the relevant lines
    with open(f"{htk_input_folder}/picklist_{picklist_no}.txt") as infile:
        htk_inputs = [i.split() for i in infile.readlines()]

    # get the average hsv bins for each carry action sequence (might want to incorporate information from pick since
    # that should give some idea about what the object is as well)

    pick_labels = ["carry_red", "carry_blue", "carry_green"]

    # [[start1, end1], ...]
    pick_frames = []

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
    avg_hsv_picks = [get_avg_hsv_bin_frames(htk_inputs, start_frame, end_frame)[0] for (start_frame, end_frame) \
                        in pick_frames]

    with open(f"{pick_label_folder}/picklist_{picklist_no}_raw.txt") as infile:
        pick_labels = [i for i in infile.read()[::2]]

    all_permutations = []

    generate_permutations_from_dict(dict(Counter(pick_labels)), all_permutations)

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

    print (best_result[1], pick_labels)