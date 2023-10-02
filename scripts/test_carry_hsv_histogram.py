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

    for picklist_no in predictions:
        pred_picklist = predictions[picklist_no]

        with open(f"pick_labels/picklist_{picklist_no}.csv", "r") as plf:
            pl_reader = csv.reader(plf)
            for row in pl_reader:
                actual_picklist = row[1].strip()

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
                print(picklist_no, "differs")