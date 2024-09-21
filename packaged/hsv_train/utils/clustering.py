# clustering utils script
import numpy as np
from collections import defaultdict


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

        # using only hue bins, binned into 6
        collapse_hue_bins = lambda x: np.array([x[150:180].sum() + x[:30].sum(), x[30:90].sum(), x[90:150].sum()])

        # collapse_hue_bins = lambda x: np.array([x[30 * i:30 * (i + 1)].sum() for i in range(6)])

        total_distance += np.linalg.norm(collapse_hue_bins(point) - collapse_hue_bins(point1))

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

def get_count_mapping(array):
	# gets count: [object] mapping of items in array
	counts = defaultdict(lambda: 0) # get counts of each object first
	for i in array:
		counts[i] += 1
	counts_mapping = defaultdict(lambda: [])
	for key, value in counts.items():
		counts_mapping[value].append(key)
	return dict(counts_mapping)

def collapse_hue_bins(hue_bins, centers, sigma):
    # [0, 60, 120] for the centers, need to control for wraparound
    gaussian_kernel = gaussian_kernel1D(181, sigma)
    gaussian_kernel_one_side = gaussian_kernel[len(gaussian_kernel) // 2:]
    bin_sums = [0 for _ in range(len(centers))]
    for center_index, center in enumerate(centers):
        # this one needs to be treated specially otherwise would be summed twice
        bin_sums[center_index] += gaussian_kernel_one_side[0] * hue_bins[center]
        for i in range(1, 60):
            bin_sums[center_index] += gaussian_kernel_one_side[i] * hue_bins[(center + i) % len(hue_bins)]
            bin_sums[center_index] += gaussian_kernel_one_side[i] * hue_bins[center - i]
    return bin_sums

def gaussian_kernel1D(length, sigma):
    ax = np.linspace(-(length - 1) / 2., (length - 1) / 2., length)
    kernel = np.exp(-0.5 * np.square(ax) / np.square(sigma))
    return kernel