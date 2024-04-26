import sys
from collections import defaultdict

sys.path.append("..")

from .base_model import Model
import logging
from utils import *
from . import IterativeClusteringModel
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.utils.multiclass import unique_labels
import pickle
import math
import numpy as np


class CarryHSVHistogramModel(Model):
    # takes in some embedding vector and

    def __init__(self):
        self.iterative_clustering_model = None

    def load_hsv_vectors(self, picklist_nos, htk_input_folder, htk_output_folder, fps=29.97, train=False):
        """
        load hsv bin data from the htk_inputs and preprocess to get the avg hsv vectors from the video and
        subtract away hand, then ready to train/fit on the vector obtained

        :param htk_inputs: htk inputs for the whole video (including non-carry sequences)
        :param action_boundaries: dict containing labels for the boundaries of actions (aeim action labels for
        pick, carry, place, carry empty)
        :return: average hsv vectors (preprocessed to get the vector that we want for iterative clustering model)
        for the picks in the sequence
        """

        # controls the number of bins that we are looking at (180 looks at all 180 hue bins, 280 looks at 180 hue bins +
        # 100 saturation bins)
        num_bins = 180

        # stores the objects
        # stores the avg_hsv_bins for all objects in a 1D list (mapped by the object id)
        objects_avg_hsv_bins = []
        # object avg hsv bin vectors grouped by picklist
        objects_avg_hsv_bins_grouped_picklist = {}

        # use as an index for picklists (since we don't necessarily have all picklist_nos
        picklist_index = 0

        for picklist_no in picklist_nos:
            logging.debug(f"Get hsv vectors, picklist number {picklist_no}, index {picklist_index}")
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
                print(htk_results_file)
                htk_boundaries = get_htk_boundaries(htk_results_file, fps=fps)
                # print(htk_boundaries)
            except Exception as e:
                # no labels yet
                print("Skipping picklist: No htk output boundaries")
                continue

            try:
                # get the htk_input to load the hsv bins from the relevant lines
                htk_inputs_file = f"{htk_input_folder}/picklist_{picklist_no}.txt"
                print(htk_inputs_file)
                with open(htk_inputs_file) as infile:
                    htk_inputs = [i.split() for i in infile.readlines()]
            except Exception as e:
                print("Skipping picklist: No htk input boundaries")
                continue

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

            pick_frames = []

            # htk label for pick
            pick_labels_char = ["e"]
            # using (predicted) htk boundaries instead of elan boundaries
            # elan_boundaries = htk_boundaries

            # print (htk_boundaries)

            if len(htk_boundaries) == 0:
                # no htk boundaries, skip
                logging.debug(f"no htk boundaries, skipping picklist {picklist_no}")
                continue

            for pick_label in pick_labels_char:
                # look through each color
                for i in range(0, len(htk_boundaries[pick_label]), 2):
                    # collect the red frames
                    start_frame = math.ceil(float(htk_boundaries[pick_label][i]) * fps)
                    end_frame = math.ceil(float(htk_boundaries[pick_label][i + 1]) * fps)
                    pick_frames.append([start_frame, end_frame])

            # sort based on start
            pick_frames = sorted(pick_frames, key=lambda x: x[0])

            logging.debug(len(pick_frames))

            for i in range(len(pick_frames) - 1):
                if pick_frames[i + 1][0] <= pick_frames[i][1]:
                    # start frame of subsequent pick is at or before the end of the current pick (there's an issue)
                    raise Exception("pick timings are overlapping, check data")

            # avg hsv bins for each pick
            num_hand_detections = [get_avg_hsv_bin_frames(htk_inputs, start_frame + 10, end_frame - 10)[1] for
                                   (start_frame, end_frame) \
                                   in pick_frames]

            print(f"pick_frames: {pick_frames}")

            print(len(htk_inputs))

            if 0 in num_hand_detections:
                print("skipping bad boundaries")
                continue

            # gather empty hand hsv bins to compare with the hsv bins of the picks
            empty_hand_label = "m"

            empty_hand_frames = []

            for i in range(0, len(htk_boundaries[empty_hand_label]), 2):
                # collect the red frames
                start_frame = math.ceil(float(htk_boundaries[empty_hand_label][i]) * fps)
                end_frame = math.ceil(float(htk_boundaries[empty_hand_label][i + 1]) * fps)
                empty_hand_frames.append([start_frame, end_frame])

            # avg hsv bins for each pick
            num_hand_detections = [get_avg_hsv_bin_frames(htk_inputs, start_frame + 10, end_frame - 10)[1] for
                                   (start_frame, end_frame) \
                                   in empty_hand_frames]

            print(f"empty_hand_frames: {empty_hand_frames}")

            if 0 in num_hand_detections and train:
                print("no empty hand, skipping bad boundaries")

                continue
                # for empty hand, we can skip those where the results were bad

            # getting the sum, multiply average by counts
            sum_empty_hand_hsv = np.zeros(num_bins)
            empty_hand_frame_count = 0

            for (start_frame, end_frame) in empty_hand_frames:
                if end_frame - start_frame < 10:
                    continue

                curr_avg_empty_hand_hsv, frame_count = get_avg_hsv_bin_frames(htk_inputs, start_frame, end_frame)

                # if frame_count == 0:
                #     # no frames where the hand was present
                #     continue
                sum_empty_hand_hsv += (curr_avg_empty_hand_hsv * frame_count)
                empty_hand_frame_count += frame_count

            # if empty_hand_frame_count == 0:
            #     # no empty hand at all
            #     continue

            # normalize so it's 1
            avg_empty_hand_hsv = sum_empty_hand_hsv / np.sum(sum_empty_hand_hsv)

            # plt.bar(range(20), avg_empty_hand_hsv)
            # plt.show()

            avg_hsv_picks = [get_avg_hsv_bin_frames(htk_inputs, start_frame + 5, end_frame - 5)[0] - avg_empty_hand_hsv
                             for
                             (start_frame, end_frame) \
                             in pick_frames]

            # collapse the avg_hsv_picks into more discrete bins for training
            avg_hsv_picks = [collapse_hue_bins(i, [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165], 15) \
                             for i in avg_hsv_picks]

            objects_avg_hsv_bins.extend(avg_hsv_picks)
            objects_avg_hsv_bins_grouped_picklist[picklist_no] = avg_hsv_picks

            picklist_index += 1

        print (f"objects_avg_hsv_bins: {objects_avg_hsv_bins}")

        return objects_avg_hsv_bins, objects_avg_hsv_bins_grouped_picklist

    def fit(self, picklist_nos, htk_input_folder, htk_output_folder, pick_label_folder, fps=29.97, visualize=False,
            write_predicted_labels=False):
        """

        :param picklist_nos: picklist numbers that we want to train on
        :param htk_input_folder: folder containing the htk_inputs (used to read hsv bin data from)
        :param htk_output_folder: folder containing the htk boundary predictions
        :param pick_label_folder: folder containing the pick labels for the different picklists
        :param visualize: flag to visualize outputs from the result of the model
        :return:
        """

        # actual_picklists = {}
        # predicted_picklists = {}
        # picklists_w_symmetric_counts = set()
        #
        # avg_hsv_bins_combined = {}

        # list of labels (corresponding to each
        combined_pick_labels = []

        # set of pick_labels
        pick_labels_grouped_picklist = {}

        # iterating through all of the different picklists
        for picklist_no in picklist_nos:
            logging.debug(f"Picklist number {picklist_no}")

            try:
                with open(f"{pick_label_folder}/picklist_{picklist_no}_raw.txt") as infile:

                    pick_labels = [i for i in infile.read().replace("\n", "")[::2]]

            except:
                logging.debug(f"no labels for picklist number {picklist_no}")
                continue

            combined_pick_labels.extend(pick_labels)
            pick_labels_grouped_picklist[picklist_no] = pick_labels

            # # randomly assign pick labels for now
            # pred_labels = np.random.choice(pick_labels, replace=False, size=len(pick_labels))

            # logging.info(f"ground_truth: {pick_labels}, pred_labels: {pred_labels}")

        objects_avg_hsv_bins, objects_avg_hsv_bins_grouped_picklist = \
            self.load_hsv_vectors(picklist_nos, htk_input_folder, htk_output_folder, fps=fps, train=True)

        return self.fit_to_data(pick_labels_grouped_picklist, objects_avg_hsv_bins, objects_avg_hsv_bins_grouped_picklist,
                                visualize, write_predicted_labels)

    def fit_to_data(self, pick_labels_grouped_picklist, objects_avg_hsv_bins, objects_avg_hsv_bins_grouped_picklist,
                    visualize, write_predicted_labels):
        """

        :param picklist_nos: picklist numbers that we want to train on
        :param htk_input_folder: folder containing the htk_inputs (used to read hsv bin data from)
        :param htk_output_folder: folder containing the htk boundary predictions
        :param pick_label_folder: folder containing the pick labels for the different picklists
        :param visualize: flag to visualize outputs from the result of the model
        :return:
        """

        # actual_picklists = {}
        # predicted_picklists = {}
        # picklists_w_symmetric_counts = set()
        #
        # avg_hsv_bins_combined = {}
        #
        # # list of labels (corresponding to each
        # combined_pick_labels = []
        #
        # # set of pick_labels
        # pick_labels_grouped_picklist = {}

        # iterating through all of the different picklists
        # for picklist_no in picklist_nos:
        #     logging.debug(f"Picklist number {picklist_no}")

            # try:
            #     with open(f"{pick_label_folder}/picklist_{picklist_no}_raw.txt") as infile:
            #
            #         pick_labels = [i for i in infile.read().replace("\n", "")[::2]]
            #
            # except:
            #     logging.debug(f"no labels for picklist number {picklist_no}")
            #     continue



            # # randomly assign pick labels for now
            # pred_labels = np.random.choice(pick_labels, replace=False, size=len(pick_labels))

            # logging.info(f"ground_truth: {pick_labels}, pred_labels: {pred_labels}")

        # objects_avg_hsv_bins, objects_avg_hsv_bins_grouped_picklist = \
        #     self.load_hsv_vectors(picklist_nos, htk_input_folder, htk_output_folder, fps=fps, train=True)

        self.iterative_clustering_model = IterativeClusteringModel()

        object_class_hsv_bins, object_class_hsv_bins_std, objects_pred_grouped_picklist = \
            self.iterative_clustering_model.fit(train_samples=objects_avg_hsv_bins_grouped_picklist,
                                                train_labels=pick_labels_grouped_picklist, \
                                                num_epochs=500, display_visual=visualize)

        if visualize:
            # the rest is just basically visualization of the code
            self.visualize(object_class_hsv_bins, objects_pred_grouped_picklist, write_predicted_labels,
                           pick_labels_grouped_picklist, objects_avg_hsv_bins)

        # convert the vectors to regular list so can be serialized and sent
        object_class_hsv_bins = {key: value.tolist() for key, value in object_class_hsv_bins.items()}
        object_class_hsv_bins_std = {key: value.tolist() for key, value in object_class_hsv_bins_std.items()}

        # return the bin means and the bin std
        return object_class_hsv_bins, object_class_hsv_bins_std
        

    def fit_iterative(self, picklist_no:int, htk_input_folder, htk_output_folder, pick_label_folder, beta, fps=29.97):
        # takes in a single picklist_no and does the processing for that
        try:
            with open(f"{pick_label_folder}/picklist_{picklist_no}_raw.txt") as infile:

                pick_labels = [i for i in infile.read().replace("\n", "")[::2]]

        except:
            logging.debug(f"no labels for picklist number {picklist_no}")
            raise Exception(f"no labels for picklist number {picklist_no}")

        objects_avg_hsv_bins, objects_avg_hsv_bins_grouped_picklist = \
            self.load_hsv_vectors([picklist_no], htk_input_folder, htk_output_folder, fps=fps, train=True)

        return self.fit_iterative_to_data(objects_avg_hsv_bins[0], pick_labels, beta)

    def fit_iterative_to_data(self, avg_hsv_bins, pick_labels, beta):
        """
        fits just a single example by checking different permutation of pick labels and finding
        the permutation with the smallest weighted (weighted by std) distance betweem avg_hsv_bins
        and matched clusters

        calls fit_iterative in the iterative clustering model

        :param avg_hsv_bins (list(list(float)): 2D list where each element in axis 0 contains the hsv bins
        of a particular pick
        :param pick_labels (list): multiset of pick labels (the order doesn't matter, so can be in
        the incorrect order
        :param beta: beta in the exponential weighted moving average
        :return: self.class_hsv_bins_mean, self.class_hsv_bins_std, pred_objects (pred_objects is the
        predicted object order that was used in the training)
        """

        return self.iterative_clustering_model.fit_iterative(avg_hsv_bins, pick_labels, beta)

    def predict(self, picklist_nos, htk_input_folder, htk_output_folder, fps=29.97, constrained_classes=None):
        # hsv_inputs: list[int[280]], 180 bins for hue, 100 bins for saturation
        # action_boundaries: dict[str, int] --> contains the timestamps of the different action start and end times
        # e.g. {'a': [start1, end1]}
        if not self.iterative_clustering_model:
            raise Exception("Model not trained")

        _, objects_avg_hsv_bins_grouped_picklist = self.load_hsv_vectors(picklist_nos, htk_input_folder,
                                                                         htk_output_folder, fps)

        return self.predict_from_hsv_bins(objects_avg_hsv_bins_grouped_picklist, constrained_classes)

    def predict_from_hsv_bins(self, objects_avg_hsv_bins_grouped_picklist, constrained_classes):
        # hsv_inputs: list[int[280]], 180 bins for hue, 100 bins for saturation
        # action_boundaries: dict[str, int] --> contains the timestamps of the different action start and end times
        # e.g. {'a': [start1, end1]}
        if not self.iterative_clustering_model:
            raise Exception("Model not trained")

        # _, objects_avg_hsv_bins_grouped_picklist = self.load_hsv_vectors(picklist_nos, htk_input_folder,
        #                                            htk_output_folder, fps)

        print(f"num picklists: {len(objects_avg_hsv_bins_grouped_picklist)}")
        output = {}

        for picklist_no in objects_avg_hsv_bins_grouped_picklist.keys():

            print(f"Picklist no. {picklist_no}")
            curr_hsv_bins = objects_avg_hsv_bins_grouped_picklist[picklist_no]
            curr = []
            # For each vector per action boundary (pick)
            for hsv_vector in curr_hsv_bins:
                curr.append(self.iterative_clustering_model.predict(hsv_vector, \
                                                                    constrained_classes=constrained_classes)[0])
            output[picklist_no] = curr

        return output

    def visualize(self, object_class_hsv_bins, objects_pred_grouped_picklist, write_predicted_labels,
                  pick_labels_grouped_picklist, objects_avg_hsv_bins):
        plt_display_index = 0
        fig, axs = plt.subplots(2, len(object_class_hsv_bins) // 2)

        for object_class, hsv_bins in object_class_hsv_bins.items():

            if plt_display_index < len(object_class_hsv_bins) // 2:
                axs[0, plt_display_index].bar(range(len(hsv_bins)), hsv_bins)
                axs[0, plt_display_index].set_title(object)
            else:
                axs[1, plt_display_index - len(object_class_hsv_bins) // 2].bar(range(len(hsv_bins)), hsv_bins)
                axs[1, plt_display_index - len(object_class_hsv_bins) // 2].set_title(object)
            plt_display_index += 1

        plt.show()

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

        colors = {
            'g': 'green',
            'a': 'yellow',
            'u': 'tan',
            'b': 'blue',
            'q': 'darkgreen',
            'r': 'red',
            'o': 'orange',
            'p': 'darkblue',
            't': 'grey',
            's': 'black'
        }

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

        plt_display_index = 0
        fig, axs = plt.subplots(2, len(object_class_hsv_bins) // 2)

        for object_class, hsv_bins in object_class_hsv_bins.items():

            if plt_display_index < len(object_class_hsv_bins) // 2:
                axs[0, plt_display_index].bar(range(len(hsv_bins)), hsv_bins, color=colors[object_class])
                axs[0, plt_display_index].set_title(letter_to_name[object_class])
                axs[0, plt_display_index].set_ylim([-0.15, 0.15])
                axs[0, plt_display_index].set_yticks([])
                axs[0, plt_display_index].set_xticks([])

            else:
                axs[1, plt_display_index - len(object_class_hsv_bins) // 2].bar(range(len(hsv_bins)), hsv_bins,
                                                                                color=colors[object_class])
                axs[1, plt_display_index - len(object_class_hsv_bins) // 2].set_title(letter_to_name[object_class])
                axs[1, plt_display_index - len(object_class_hsv_bins) // 2].set_ylim([-0.15, 0.15])
                axs[1, plt_display_index - len(object_class_hsv_bins) // 2].set_yticks([])
                axs[1, plt_display_index - len(object_class_hsv_bins) // 2].set_xticks([])
            plt_display_index += 1

            axs[0, 0].set_yticks([-0.15, -0.1, -0.05, 0, 0.05, 0.1, 0.15])
            axs[1, 0].set_yticks([-0.15, -0.1, -0.05, 0, 0.05, 0.1, 0.15])

        plt.show()

        # print the results on a per picklist level
        for picklist_no, picklist_pred in objects_pred_grouped_picklist.items():
            print(f"Picklist no. {picklist_no}")
            print(f"Predicted labels: {picklist_pred}")
            # use the same mapping for combined_pick_labels as object ids
            picklist_gt = pick_labels_grouped_picklist[picklist_no]
            if write_predicted_labels:
                # want to write the results
                with open(f"pick_labels/picklist_{picklist_no}.csv", "w") as outfile:
                    outfile.write(f"{picklist_no}, {''.join(picklist_pred)}, {''.join(picklist_gt)}")

            print(f"Actual labels:    {picklist_gt}")

        # flatten arrays
        actual_picklists = [i for j in sorted(objects_pred_grouped_picklist.keys()) for i in
                            pick_labels_grouped_picklist[j]]
        # actual_picklists = combined_pick_labels
        predicted_picklists = [i for j in sorted(objects_pred_grouped_picklist.keys()) for i in
                               objects_pred_grouped_picklist[j]]

        ax = plt.figure().add_subplot(projection='3d')

        ax.scatter([i[0] for i in objects_avg_hsv_bins],
                   [i[len(objects_avg_hsv_bins[0]) // 3] for i in objects_avg_hsv_bins], \
                   [i[2 * len(objects_avg_hsv_bins[0]) // 3] for i in objects_avg_hsv_bins], \
                   c=[color_mapping[i] for i in predicted_picklists])

        plt.show()

        confusions = defaultdict(int)
        label_counts = defaultdict(int)
        for pred, label in zip(predicted_picklists, actual_picklists):
            if pred != label:
                confusions[pred + label] += 1

        logging.debug(confusions)

        confusion_matrix = metrics.confusion_matrix(actual_picklists, predicted_picklists)

        logging.debug(actual_picklists)
        logging.debug(predicted_picklists)

        names = ['red', 'green', 'blue', 'darkblue', 'darkgreen', 'orange', 'alligatorclip', 'yellow', 'clear',
                 'candle']

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
        cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=conf_mat_norm, display_labels=unique_names)

        cm_display.plot(cmap=plt.cm.Blues)

        plt.xticks(rotation=90)

        plt.tight_layout()

        plt.show()
