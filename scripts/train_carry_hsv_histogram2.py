"""

iterative improvement method that swaps pairs which result in smaller intracluster distance and simulated annealing
when there is no pair which would reduce the intracluster distance

"""

from utils import *

import argparse

import logging

from models import CarryHSVHistogramModel

import pickle

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

carry_histogram_hsv_model = CarryHSVHistogramModel()

carry_histogram_hsv_model.fit(PICKLISTS, htk_input_folder, htk_output_folder, pick_label_folder, \
                              fps=29.97, visualize=True, write_predicted_labels=True)
with open("saved_models/carry_histogram_hsv_model.pkl", "wb") as outfile:
    pickle.dump(carry_histogram_hsv_model, outfile)


# saving the model

# objects_pred_hsv_bins = defaultdict(lambda: [])
#
# # gather the bins for the predicted
# for object, pred in objects_pred.items():
#     # use the one for classification
#     objects_pred_hsv_bins[pred].append(objects_avg_hsv_bins[object])
#
# objects_pred_avg_hsv_bins = {}
#
# for object_type, object_hsv_bins in objects_pred_hsv_bins.items():
#     # store the standard deviation as well of the different axes
#     objects_pred_avg_hsv_bins[object_type] = [np.array(object_hsv_bins).mean(axis=0), np.array(object_hsv_bins).std(axis=0, ddof=1)]
#
# with open("object_type_hsv_bins_copy.pkl", "wb") as outfile:
#     # without removing the hand
#     pickle.dump(objects_pred_avg_hsv_bins, outfile)
#
# objects_pred_hsv_bins = defaultdict(lambda: [])