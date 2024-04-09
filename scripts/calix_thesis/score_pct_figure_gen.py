import os
import cv2
import argparse

import sys
from pathlib import Path # if you haven't already done so
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))
# from utils.hand_tracker import HandTracker
import numpy as np
np.set_printoptions(suppress=True)
# import mediapipe as mp
import matplotlib.pyplot as plt

import sys
sys.path.append("..")
from utils import *
sys.path.append("./calix_thesis")

from util import parse_action_label_file


if __name__ == '__main__':
    
    #controlled random
    np.random.seed(42)

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", "-c", type=str, default="../configs/calix.yaml",
                    help="Path to experiment config (scripts/configs)")
    parser.add_argument("--percent_frames", type=float, default = 1)
    parser.add_argument("--display", action = "store_true")
    args = parser.parse_args()

    configs = load_yaml_config(args.config_file)

    elan_label_folder = configs["file_paths"]["elan_annotated_file_path"]
    htk_input_folder = configs["file_paths"]["htk_input_file_path"]
    video_folder = configs["file_paths"]["video_file_path"]
    pick_label_folder = configs["file_paths"]["label_file_path"]
    htk_output_folder = configs["file_paths"]["htk_output_file_path"]
    score_folder = configs["file_paths"]["calix_score_file_path"]

    picklist_nos = np.random.choice(np.arange(275, 355), 5)
    print("Chose picklist {}".format(picklist_nos))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    pcts = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99])

    for picklist_no in picklist_nos:
        try:
            # Since I annotated with raw frame count info, I define my own parsing method
            # htk_boundaries = get_htk_boundaries(f"{htk_output_folder}/results-{picklist_no}")
            frame_boundaries = parse_action_label_file(os.path.join(htk_output_folder, 'picklist_{}.txt'.format(picklist_no)))
            
        except Exception as e:
            print (f"Couldn't find files for {picklist_no}")
            quit()
    
        carry_item_bounds, num_carry_item_seq = frame_boundaries["carry_item"], len(frame_boundaries["carry_item"]) 
        carry_empty_bounds, num_carry_empty_seq = frame_boundaries["carry_empty"], len(frame_boundaries["carry_empty"])

        carry_item_seq_no, carry_empty_seq_no = np.random.randint(num_carry_item_seq), np.random.randint(num_carry_empty_seq)
        # print(carry_item_seq_no, carry_empty_seq_no)

        item_seq_bounds, empty_seq_bounds = carry_item_bounds[carry_item_seq_no], carry_empty_bounds[carry_empty_seq_no]
        item_seq_scores, empty_seq_scores = np.load(f"{score_folder}/picklist_{picklist_no}/picklist_{picklist_no}_{carry_item_seq_no}_carry_item.npy"), np.load(f"{score_folder}/picklist_{picklist_no}/picklist_{picklist_no}_{carry_item_seq_no}_carry_empty.npy")
        
        sorted_scores = np.argsort(item_seq_scores[:, -1], axis = 0)
        n = sorted_scores.shape[0]
        pct_frames = (n * pcts).astype(int)
        #get worst 10% of frames
        ordered_frames = (item_seq_scores[sorted_scores[pct_frames], 0].astype(int))

        cap = cv2.VideoCapture(f"{video_folder}/picklist_{picklist_no}.mp4")
        frame_width, frame_height = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fig, axs = plt.subplots(nrows = 3, ncols = 4)
        fig.set_size_inches(16, 9, forward=True)

        for i in range(ordered_frames.shape[0]):
            frame_no = ordered_frames[i]
            cap.set(1, frame_no)
            hasFrame, frame = cap.read()
            axs[i // 4, i % 4].imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            axs[i // 4, i % 4].set_title(f"{int(pcts[i] * 100)}th %ile")
        fig.suptitle(f"Picklist {picklist_no} carry item frames")
        plt.savefig(f"score_pct_figures/picklist_{picklist_no}_{carry_item_seq_no}_item.jpg")
        # plt.show()
        plt.clf()
            

        sorted_scores = np.argsort(empty_seq_scores[:, -1], axis = 0)
        n = sorted_scores.shape[0]
        #get worst 10% of frames
        pct_frames = (n * pcts).astype(int)
        ordered_frames = (empty_seq_scores[sorted_scores[pct_frames], 0].astype(int))
        # print(f"picklist_{picklist_no} empty", empty_seq_scores[sorted_scores[ : (int)(n * 0.1)]][:, (0, -1)])
        # print(worst_frames)
        fig, axs = plt.subplots(nrows = 3, ncols = 4)    
        fig.set_size_inches(16, 9, forward=True)

        for i in range(ordered_frames.shape[0]):
            frame_no = ordered_frames[i]
            cap.set(1, frame_no)
            hasFrame, frame = cap.read()
            axs[i // 4, i % 4].imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            axs[i // 4, i % 4].set_title(f"{int(pcts[i] * 100)}th %ile")
        fig.suptitle(f"Picklist {picklist_no} carry empty frames")
        plt.savefig(f"score_pct_figures/picklist_{picklist_no}_{carry_empty_seq_no}_empty.jpg")
        # plt.show()
        plt.clf()

    