"""

    Given a list of vectors and a corresponding list of set of labels, IterativeClusteringModel first randomly assigns
    labels to the vectors based on the set of labels (set of labels creates a set of constraints that the outputs must
    fulfill. Then, in each epoch, a random object in each picklist is picked and swapped with every other object and
    labels are swapped following a simulated annealing scheme (swapped with increasing probability based on decrease in
    std normalized distance to each cluster/label mean)

"""

from .base_model import Model
import matplotlib.pyplot as plt
import math
import numpy as np
from collections import defaultdict
import itertools

class IterativeClusteringModel(Model):
    # takes in some embedding vectors and set labels for the embedding vectors and find matches between them

    def __init__(self):
        # {object class: average_hsv_vector for the object class}
        self.class_hsv_bins_mean, self.class_hsv_bins_std = None, None
        pass

    def fit(self, train_samples, train_labels, num_epochs, display_visual=False):
        """
        :param train_samples: dict of list of vectors, [[vector for (vector representing pick sequence) in picklist] \
                                                                for picklist in training_set]
        :param train_labels: dict of list of labels, [{labels in picklist} for picklist in training_set, disregard the ordering
                            of the labels in the list (figuring that out using the iterative clustering algorithm)

        :param num_epochs: num of epochs to run for
        :param display_visual: to display vector visualization
        :return: None (model is trained in-place
        """

        # take the intersection of the keys of both the train_samples and the train_labels sets
        # only work with the intersection
        picklist_nos = set(train_samples.keys()) & set(train_labels.keys())

        print (f"picklists that have samples but no labels: {set(train_samples.keys()).difference(set(train_labels.keys()))}")

        print (f"picklists that have labels but no samples: {set(train_labels.keys()).difference(set(train_samples.keys()))}")
        # preprocessing

        # {picklist_no: [index in objects_avg_hsv_bins for objects that are in this picklist]}
        picklist_objects = defaultdict(lambda: [])

        # stores the avg_hsv_bins for all objects in a 1D list (mapped by the object id)
        objects_avg_hsv_bins = []

        # pick labels in a 1D list matching the object's hsv vectors in objects_avg_hsv_bins
        combined_pick_labels = []

        # {index in objects_avg_hsv_bins: curr_predicted_object_type}
        objects_pred = {}

        # {object type: [index in objects_avg_hsv_bins for objects that are predicted to be of that type]
        pred_objects = defaultdict(lambda: set())

        for picklist_no in picklist_nos:
            picklist_vectors = train_samples[picklist_no]
            # picklist_vectors contains the list of vectors representing each pick in the pioklist
            # picklist no is just the order of their occurrence in train_samples)

            # get the list of labels that are present (the list is unordered, the only thing that is important is the
            # count of each element in the labels)
            pick_labels = train_labels[picklist_no]

            combined_pick_labels.extend(pick_labels)

            # randomly assign pick labels for now
            pred_labels = np.random.choice(pick_labels, replace=False, size=len(pick_labels))

            # processing the picklists to have a global list of objects and mapping picklists to the list of objects that
            # it contains
            for i in range(len(picklist_vectors)):
                # assign object_id based on the index of object in objects_avg_hsv_bins

                object_vector = picklist_vectors[i]

                object_id = len(objects_avg_hsv_bins)
                picklist_objects[picklist_no].append(object_id)
                # map object to prediction and prediction to all objects with the same
                pred_objects[pred_labels[i]].add(object_id)
                objects_pred[object_id] = pred_labels[i]
                # overlapping bins so there's no sudden dropoff
                objects_avg_hsv_bins.append(object_vector)

        # training

        epochs = 0

        if display_visual:
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

            cluster_fig = plt.figure()

            cluster_ax = cluster_fig.add_subplot(projection="3d")

            cluster_ax.scatter([i[0] for i in objects_avg_hsv_bins],
                       [i[len(objects_avg_hsv_bins[0]) // 3] for i in objects_avg_hsv_bins], \
                       [i[2 * len(objects_avg_hsv_bins[0]) // 3] for i in objects_avg_hsv_bins], \
                       c=[color_mapping[i] for i in combined_pick_labels])

            # show one time with blocking before starting to run the iterative clustering algorithm
            # plt.show()

            plt.show(block=False)

        while epochs < num_epochs:
            print(f"Starting epoch {epochs}...")
            # for each epoch, iterate through all the picklists and offer one swap
            for picklist_no, objects_in_picklist in picklist_objects.items():
                # check whether swapping the current element with any of the other
                # randomly pick an object
                object1_id = np.random.choice(picklist_objects[picklist_no])

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
                    object1_pred_mean = np.array(
                        [objects_avg_hsv_bins[i] for i in pred_objects[object1_pred] if i != object1_id]).mean(axis=0)
                    object2_pred_mean = np.array(
                        [objects_avg_hsv_bins[i] for i in pred_objects[object2_pred] if i != object2_id]).mean(axis=0)

                    object1_pred_std = np.array(
                        [objects_avg_hsv_bins[i] for i in pred_objects[object1_pred] if i != object1_id]).std(axis=0,
                                                                                                              ddof=1)
                    object2_pred_std = np.array(
                        [objects_avg_hsv_bins[i] for i in pred_objects[object2_pred] if i != object2_id]).std(axis=0,
                                                                                                              ddof=1)

                    object1_pred_std = np.nan_to_num(object1_pred_std)
                    object2_pred_std = np.nan_to_num(object2_pred_std)

                    # should be proportional to log likelihood (assume normal, then take exponential of these / 2 to get pdf

                    # current distance between the objects and their respective means, 0.00001 added for numerical stability
                    curr_distance = (((object1_pred_mean - objects_avg_hsv_bins[object1_id]) ** 2) / (
                                object1_pred_std ** 2 + 0.000001)).sum(axis=0) + \
                                    (((object2_pred_mean - objects_avg_hsv_bins[object2_id]) ** 2) / (
                                                object2_pred_std ** 2 + 0.000001)).sum(axis=0)

                    # distance if they were to swap
                    new_distance = (((object2_pred_mean - objects_avg_hsv_bins[object1_id]) ** 2) / (
                                object2_pred_std ** 2 + 0.000001)).sum(axis=0) + \
                                   (((object1_pred_mean - objects_avg_hsv_bins[object2_id]) ** 2) / (
                                               object1_pred_std ** 2 + 0.000001)).sum(axis=0)

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

                if object2_distance_reduction[best_pos_object2] <= 0:
                    object2_distance_reduction_sum = sum(object2_distance_reduction.values())
                    # all negative, need to pick one at random with decreasing probability based on the numbers,
                    # take exponential of the distance reduction which should all be negative
                    swap_object_id = np.random.choice(list(object2_distance_reduction.keys()), p=np.array(
                        [math.e ** (i / object2_distance_reduction_sum) for i in
                         object2_distance_reduction.values()]) / sum(
                        [math.e ** (i / object2_distance_reduction_sum) for i in
                         object2_distance_reduction.values()]))

                    # give reducing odds of getting a random swap
                    to_swap = np.random.choice([0, 1], p=[1 - math.e ** (-epochs / 50), math.e ** (-epochs / 50)])

                    if not to_swap:
                        # to_swap is False so skip the rest of the assignment
                        continue


                else:
                    # reduce randomness towards the end (simulated annealing)

                    # only want to consider those with positive distance reduction
                    object2_distance_reduction_positive = {key: value for key, value in
                                                           object2_distance_reduction.items() if value > 0}

                    # use to normalize the distances otherwise can get very small
                    object2_distance_reduction_sum_positive = sum(
                        [i for i in object2_distance_reduction_positive.values()])

                    # definitely swap, but some uncertainty about which element it is swapped with (decreasing temperature by adding epochs
                    # as a factor)
                    swap_object_id = np.random.choice(list(object2_distance_reduction_positive.keys()), p=np.array(
                        [math.e ** (i / object2_distance_reduction_sum_positive) for i in
                         object2_distance_reduction_positive.values()]) / sum(
                        [math.e ** (i / object2_distance_reduction_sum_positive) for i in
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

            # flatten out the predicted labels for each picklist so all the predictions across picklists
            # are in a 1D array (for visualization)
            predicted_picklists = [objects_pred[i] for i in range(len(combined_pick_labels))]

            if display_visual:

                cluster_ax.clear()

                cluster_ax.scatter([i[0] for i in objects_avg_hsv_bins],
                                   [i[len(objects_avg_hsv_bins[0]) // 3] for i in objects_avg_hsv_bins], \
                                   [i[2 * len(objects_avg_hsv_bins[0]) // 3] for i in objects_avg_hsv_bins], \
                                   c=[color_mapping[i] for i in predicted_picklists])

                cluster_fig.canvas.draw()

        # reinitialize self.class_hsv_bins
        self.class_hsv_bins_mean = {}
        self.class_hsv_bins_std = {}

        for object_class, object_indices in pred_objects.items():
            self.class_hsv_bins_mean[object_class] = np.array([objects_avg_hsv_bins[i] for i in object_indices]).mean(axis=0)
            self.class_hsv_bins_std[object_class] = np.array([objects_avg_hsv_bins[i] for i in object_indices]).std(axis=0, ddof=1)
        # print (self.class_hsv_bins)

        # predictions for the objects grouped by the picklist (and in order of the objects as they occurred in the picklist
        objects_pred_grouped_picklist = {}

        for picklist_index in picklist_objects.keys():
            objects_pred_grouped_picklist[picklist_index] = [objects_pred[i] for i in picklist_objects[picklist_index]]



        # use the predict function to predict labels
        pred_labels = []

        for input_vector in objects_avg_hsv_bins:
            pred_labels.append(self.predict(input_vector)[0])

        print(combined_pick_labels)
        print (pred_labels)


        # need to change objects_pred to be object predictions grouped by picklist
        return self.class_hsv_bins_mean, self.class_hsv_bins_std, objects_pred_grouped_picklist


    # need a method to fit just a single new example (get the ones closest and add that to the mean)
    def fit_iterative(self, avg_hsv_bins, pick_labels, beta):
        """
        fits just a single example by checking different permutation of pick labels and finding
        the permutation with the smallest weighted (weighted by std) distance betweem avg_hsv_bins
        and matched clusters

        :param avg_hsv_bins (list(list(float)): 2D list where each element in axis 0 contains the hsv bins
        of a particular pick
        :param pick_labels (list): multiset of pick labels (the order doesn't matter, so can be in
        the incorrect order
        :param beta: beta in the exponential weighted moving average, which is used as the weight to put on
        the previous value (conversely, 1 - beta weight is put on the new reading)
        :return: self.class_hsv_bins_mean, self.class_hsv_bins_std, pred_objects (pred_objects is the
        predicted object order that was used in the training)
        """

        # store the vector distances across all of the picks to compute the total distance later on
        vector_distances_all_picks = []

        classes = set(pick_labels)

        for hsv_bin_vector in avg_hsv_bins:
            vector_distances = {
                key: np.linalg.norm((self.class_hsv_bins_mean[key] - hsv_bin_vector) / self.class_hsv_bins_std[key]) \
                for key in classes
                }

            vector_distances_all_picks.append(vector_distances)

        # get all of the distinct permutations of the pick_labels for comparison
        permutations = set(itertools.permutations(pick_labels))

        # find the permutation with the smallest distance and use that as the labels for updating the bins later on
        # (permutation, sum of weighted distance for permutation)
        smallest_distance = ([], float('inf'))

        for permutation in permutations:
            total_distance = 0.

            for i in range(len(permutation)):
                # get the distance given the current label
                total_distance += vector_distances_all_picks[i][permutation[i]]

            if total_distance < smallest_distance[1]:
                smallest_distance = (permutation, total_distance)


        # we just update the mean since we don't store all of the previous points so can't compute std
        labels = smallest_distance[0]

        for i in range(len(labels)):
            # update the mean bins for the one with the current label
            label = labels[i]

            # update using EWMA
            self.class_hsv_bins_mean[label] = beta * self.class_hsv_bins_mean[label] + \
                                              (1 - beta) * np.array(avg_hsv_bins[i])

        # return the bin means, the std, and the predicted labels that were derived for this picklist
        return self.class_hsv_bins_mean, self.class_hsv_bins_std, labels


    def predict(self, input_vector, constrained_classes=None):

        if constrained_classes is not None:
            classes = constrained_classes
        else:
            classes = self.class_hsv_bins_mean.keys()
        #
        # for key in classes:
        #     print (key, self.class_hsv_bins_mean[key], self.class_hsv_bins_std[key])

        vector_distances = {key: np.linalg.norm((self.class_hsv_bins_mean[key] - input_vector) / self.class_hsv_bins_std[key]) \
                            for key in classes
                            }

        # return the best class and the distances between that class and the final output
        return min(vector_distances.keys(), key=lambda x: vector_distances[x]), vector_distances