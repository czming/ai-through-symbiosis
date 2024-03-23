import numpy as np
import logging
import argparse



import sys
sys.path.append("..")
from utils import *
sys.path.append("./calix_thesis")


np.random.seed(42)

logging.basicConfig(level=logging.DEBUG)


parser = argparse.ArgumentParser()
parser.add_argument("--config", "-c", type=str, default="../configs/calix.yaml",
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
PICKLISTS = list(range(136, 137)) # list(range(136, 235)
# PICKLISTS = [136, 137, 138]

# controls the number of bins that we are looking at (180 looks at all 180 hue bins, 280 looks at 180 hue bins +
# 100 saturation bins)
num_bins = 20

actual_picklists = {}
predicted_picklists = {}
picklists_w_symmetric_counts = set()

avg_hsv_bins_combined = {}

# stores the objects
objects_avg_hsv_bins = []
classification_objects_avg_hsv_bins = []

# {picklist_no: [index in objects_avg_hsv_bins for objects that are in this picklist]}
picklist_objects = defaultdict(lambda: set())

# {object type: [index in objects_avg_hsv_bins for objects that are predicted to be of that type]
pred_objects = defaultdict(lambda: set())

objects_pred = {}

combined_pick_labels = []

# initialization (randomly assign colors)
for picklist_no in PICKLISTS:
    logging.debug(f"Picklist number {picklist_no}")

    # with open(f"{pick_label_folder}/picklist_{picklist_no}_raw.txt") as infile:
                # pick_labels = [i for i in infile.read().replace("\n", "")[::2]]

    try:
        # load elan boundaries, so we can take average of each picks elan labels
        elan_label_file = f"{elan_label_folder}/picklist_{picklist_no}.eaf"
        elan_boundaries = get_elan_boundaries_general(elan_label_file)
        # total_time += elan_boundaries["carry_empty"][-1]
    except:
        # no labels yet
        print("Skipping picklist: No elan boundaries")
        continue

    # # frame_boundaries = parse_action_label_file(os.path.join(htk_output_folder, 'picklist_{}.txt'.format(picklist_no)))
    # try:
    #     htk_results_file = f"{htk_output_folder}/results-" + str(picklist_no)
    #     print(htk_results_file)
    #     htk_boundaries = get_htk_boundaries(htk_results_file)
    #     # print(htk_boundaries)
    # except:
    #     # no labels yet
    #     print("Skipping picklist: No htk boundaries")
    #     continue


    # # print(elan_boundaries)
    # print(htk_boundaries)
    # # htk label for pick
    # action_labels = ["a", "e", "i", "m"]
    # # using (predicted) htk boundaries instead of elan boundaries
    # all_action_frames = []
    # action_averages = []
    # for action_label in action_labels:
    #     # look through each color
    #     action_frames = []

    #     for i in range(0, len(htk_boundaries[action_label]), 2):
    #         # collect the red frames
    #         start_frame = math.ceil(float(htk_boundaries[action_label][i]) * 29.97)
    #         end_frame = math.ceil(float(htk_boundaries[action_label][i + 1]) * 29.97)
    #         action_frames.append([start_frame, end_frame])
    #     # sort based on start
    #     action_frames = sorted(action_frames, key=lambda x: x[0])
    #     print(action_frames)
    
    elan_action_frame_times = []
    for boundary_list in elan_boundaries.values():
        for boundary_value in boundary_list:
            elan_action_frame_times.append(math.ceil(float(boundary_value) * 29.97))
    elan_action_frame_times = set(elan_action_frame_times)
    print(elan_action_frame_times)