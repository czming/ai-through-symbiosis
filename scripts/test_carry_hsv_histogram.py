"""

iterative improvement method that swaps pairs which result in smaller intracluster distance and simulated annealing
when there is no pair which would reduce the intracluster distance

"""

import csv
from utils import *
import argparse
import logging
from models import CarryHSVHistogramModel
import pickle
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.utils.multiclass import unique_labels

np.random.seed(42)

logging.basicConfig(level=logging.DEBUG)


parser = argparse.ArgumentParser()
parser.add_argument("--config_file", "-c", type=str, default="configs/zm.yaml",
                    help="Path to experiment config (scripts/configs)")
args = parser.parse_args()

configs = load_yaml_config(args.config_file)

elan_label_folder = configs["file_paths"]["elan_annotated_file_path"]
htk_input_folder = configs["file_paths"]["htk_input_file_path"]
video_folder = configs["file_paths"]["video_file_path"]
pick_label_folder = configs["file_paths"]["label_file_path"]

htk_output_folder = configs["file_paths"]["htk_output_file_path"]

# picklists that we are looking at
# PICKLISTS = list(range(136, 224)) + list(range(225, 230)) + list(range(231, 235))
PICKLISTS = list(range(136, 235)) # list(range(136, 235)

with open("saved_models/carry_histogram_hsv_model.pkl", "rb") as f:
    carry_histogram_hsv_model = pickle.load(f)

    predictions = carry_histogram_hsv_model.predict(PICKLISTS, htk_input_folder, htk_output_folder, fps=29.97)

    total_missed = 0
    total_objects = 0
    total_incorrect_objects = 0
    pred_picklists = []
    actual_picklists = []

    for picklist_no in predictions:
        pred_picklist = predictions[picklist_no]

        with open(f"pick_labels/picklist_{picklist_no}.csv", "r") as plf:
            pl_reader = csv.reader(plf)
            for row in pl_reader:
                actual_picklist = [ch for ch in row[1].strip()]

            pred_pickset = {}
            for item in pred_picklist:
                if item not in pred_pickset:
                    pred_pickset[item] = 0
                pred_pickset[item] += 1

            actual_pickset = {}
            for item in actual_picklist:
                if item not in actual_pickset:
                    actual_pickset[item] = 0
                actual_pickset[item] += 1

            if pred_pickset != actual_pickset:
                total_missed += 1
                max_len = max(len(pred_picklist), len(actual_picklist))
                pred_picklist += [''] * (max_len - len(pred_picklist))
                actual_picklist += [''] * (max_len - len(actual_picklist))
                for pred, actual in zip(pred_picklist, actual_picklist):
                    if pred != actual:
                        total_incorrect_objects += 1
                    total_objects += 1
            else:
                total_objects += len(pred_picklist)

            pred_picklists.extend(pred_picklist)
            actual_picklists.extend(actual_picklist)

    print("Total picklists:", len(predictions))
    print("Total incorrect picklists:", total_missed)
    print("Total items:", total_objects)
    print("Total incorrect items:", total_incorrect_objects)

    # Confusion matrix:

    # confusion_matrix = metrics.confusion_matrix(actual_picklists, pred_picklists)
    # conf_mat_norm = (confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis])

    # letter_to_name = {
    #     'r': 'red',
    #     'g': 'green',
    #     'b': 'blue',
    #     'p': 'darkblue',
    #     'q': 'darkgreen',
    #     'o': 'orange',
    #     's': 'alligatorclip',
    #     'a': 'yellow',
    #     't': 'clear',
    #     'u': 'candle'
    # }
    # names = ['red', 'green', 'blue', 'darkblue', 'darkgreen', 'orange', 'alligatorclip', 'yellow', 'clear', 'candle']
    #
    # actual_picklists_names = []
    # predicted_picklists_names = []
    #
    # letter_counts = defaultdict(lambda: 0)
    # for letter in actual_picklists:
    #     name = letter_to_name[letter]
    #     letter_counts[name] += 1
    #     actual_picklists_names.append(name)
    #
    # for index, letter in enumerate(pred_picklists):
    #     # print(index)
    #     predicted_picklists_names.append(letter_to_name[letter])
    #
    # unique_names = unique_labels(actual_picklists_names, predicted_picklists_names)
    # cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=conf_mat_norm, display_labels = unique_names)
    # cm_display.plot(cmap=plt.cm.Blues)
    # plt.xticks(rotation=90)
    # plt.tight_layout()
    # plt.show()