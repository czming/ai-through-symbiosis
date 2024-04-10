"""
Provides functions for drawing mock place bin setup
"""
import cv2
import numpy as np
import math 
import os
import pickle

import sys
sys.path.append("..")
from utils import *
sys.path.append("./calix_thesis")

from util import parse_action_label_file, get_hand_crop_from_frame
from object_classification_calix import find_closest_in_set

#640 x 360???
#1280 x 720

def create_background(width = 1280, height = 720):
    bg = np.zeros([height, width, 3], dtype=np.uint8)

    return bg


def process_picklist(pick_label_folder, picklist_no):
    

    try:
        #get the picklist raw file and process
        with open(f"{pick_label_folder}/picklist_{picklist_no}_raw.txt") as infile:
            lines = [i for i in infile.read().strip().replace("\n", "")]

            pick_labels = lines[::2]
            pick_bins = [int(_) for _ in lines[1::2]]
            print(pick_labels, pick_bins)

    except Exception as e:
        print (f"Couldn't find files for {picklist_no}")
        quit()

    num_picks = len(pick_labels)

    for i in range(num_picks):
        pass
        #draw gt in left bins
        #draw hand picks
    

    raw_item_pic_folder = "./item_imgs"

def generate_ground_truth_images(color_model_path, picklist_range, htk_input_folder, htk_output_folder, video_folder, pct_frames = 0.9):
    #HSV color model - formatted as {color key, [HSV histogram bin featvec]}
    with open(color_model_path, "rb") as infile:
        objects_hsv_model = pickle.load(infile)
    

    objects = list(objects_hsv_model.keys())
    best_obj_map = {key: (float('inf'), None) for key in objects}


    for picklist_no in range(picklist_range[0], picklist_range[1]):
        print (f"picklist_no: {picklist_no}")
        try:
            # with open(f"{pick_label_folder}/picklist_{picklist_no}_raw.txt") as infile:
                # pick_labels = [i for i in infile.read().replace("\n", "")[::2]]

            with open(f"{htk_input_folder}/picklist_{picklist_no}.txt") as infile:
                htk_inputs = [i.split() for i in infile.readlines()]

            frame_boundaries = parse_action_label_file(os.path.join(htk_output_folder, 'picklist_{}.txt'.format(picklist_no)))
        except Exception as e:
            print (f"Skipping picklist {picklist_no}")
            continue

     
        #Since I haven't used HTK to generate boundaries as of now, I don't need this
        
        #number of carry item sequenecs
        picks = len(frame_boundaries['carry_item'])
        empty_picks = len(frame_boundaries['carry_empty'])

        #singular best frame number per pick
        top_scoring_frame_set = []

        #list of best n% frames per pick
        best_frames = []
        best_empty_hand_frames = []

        for pick_no in range(picks):
            score_matrix = np.load(os.path.join(score_folder, 'picklist_{}'.format(picklist_no), 'picklist_{}_{}_carry_item.npy'.format(picklist_no, pick_no)))
            sorted_scores = np.argsort(score_matrix[:, -1], axis = 0)
            n = sorted_scores.shape[0]

            best_frames.append(score_matrix[sorted_scores[-(int)(n * pct_frames) :]][:, 0].astype(int))
            top_scoring_frame_set.append(score_matrix[sorted_scores[-1], 0])

        for pick_no in range(empty_picks):
            score_matrix = np.load(os.path.join(score_folder, 'picklist_{}'.format(picklist_no), 'picklist_{}_{}_carry_empty.npy'.format(picklist_no, pick_no)))
            sorted_scores = np.argsort(score_matrix[:, -1], axis = 0)
            n = sorted_scores.shape[0]

            best_empty_hand_frames.extend(score_matrix[sorted_scores[-(int)(n * pct_frames) :]][:, 0].astype(int))
                
        #no need for summation used in previous versions since this should weight each frame equally
        avg_empty_hand_hsv = get_avg_hsv_bin_frames_frame_list(htk_inputs, best_empty_hand_frames)[0]
        avg_hsv_picks = [get_avg_hsv_bin_frames_frame_list(htk_inputs, frame_list)[0] - avg_empty_hand_hsv for frame_list in best_frames]

        for pick_no, i in enumerate(avg_hsv_picks):
            # print(pick_labels[index])#ground truth
            i = collapse_hue_bins(i, [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165], 15)
            pred_obj, distances = find_closest_in_set(i, objects_hsv_model) #unconditional rpediction

            dist_to_centroid = distances[pred_obj]
            best_dist, best_obj_frame = best_obj_map[pred_obj]
            if dist_to_centroid < best_dist:
                #search for best score
                frame_no = top_scoring_frame_set[pick_no]
                obj_frame = get_hand_crop_from_frame(os.path.join(video_folder, f"picklist_{picklist_no}.mp4"), frame_no)
                if obj_frame is not None:
                    best_obj_map[pred_obj] = (dist_to_centroid, obj_frame)
                else:
                    print('found new closest distance but no hand detection in frame!')
              

    return best_obj_map



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--picklist", type = int)
    parser.add_argument("--config_file", "-c", type=str, default="../configs/calix.yaml",
                    help="Path to experiment config (scripts/configs)")
    parser.add_argument("--display", action = "store_true")
    parser.add_argument("--color_model_path", default = "./hsv_models/hsv_bin_classification_frame_filtered_90_percent_frames_unconditional.pkl")
    args = parser.parse_args()


    configs = load_yaml_config(args.config_file)

    elan_label_folder = configs["file_paths"]["elan_annotated_file_path"]
    htk_input_folder = configs["file_paths"]["htk_input_file_path"]
    video_folder = configs["file_paths"]["video_file_path"]
    pick_label_folder = configs["file_paths"]["label_file_path"]
    htk_output_folder = configs["file_paths"]["htk_output_file_path"]
    score_folder = configs["file_paths"]["calix_score_file_path"]
    
    #(136, 235)
    gt_images = generate_ground_truth_images(args.color_model_path, (275, 355), htk_input_folder, htk_output_folder, video_folder)

    for k, v in gt_images.items():
        print(k)
        best_img = v[1]
        if best_img is not None:
            # print(best_img)
            # cv2.imshow('tmp', best_img)
            # if cv2.waitKey(-1) & 0xFF == ord('q'):
            #     continue
            cv2.imwrite(os.path.join('./item_imgs', f"{k}.jpg"), best_img)

    # process_picklist(pick_label_folder, args.picklist)
    

    # os.path.join(raw_item_pic_folder, )