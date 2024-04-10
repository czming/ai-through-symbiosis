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

import glob

from util import parse_action_label_file, get_hand_crop_from_frame
from object_classification_calix import find_closest_in_set

#640 x 360???
#1280 x 720

def create_background(width = 1280, height = 720):
    bg = np.zeros([height, width, 3], dtype=np.uint8)
    
    bg[:, (639, 640, 641)] = 255.
    bg[(239, 240, 241, 479, 480, 481), : 640] = 255.

    for i in range(3):
        bg = cv2.putText(bg, "Bin " + str(i + 1), (0, 240 * (i + 1) - 5), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 0), 3)

    bg = cv2.putText(bg, "Your picks", (640, 720 - 5), cv2.FONT_HERSHEY_PLAIN, 3, (0,255, 255), 3)
    return bg


def process_picklist(pick_label_folder, picklist_no, gt_img_folder, score_folder):
    bg = create_background()
    bg_w, bg_h = bg.shape[1], bg.shape[0]
    gt_w, gt_h = 128 if bg_w > 1000 else 64, 128 if bg_w > 1000 else 64

    picklist_score_folder = os.path.join(score_folder, f"picklist_{picklist_no}")

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
    # gt_imgs = [os.path.join(gt_img_folder, x) for x in os.listdir(gt_img_folder)]
    bin_cts = [0, 0, 0]

    for i in range(num_picks):
        """gt frame"""
        gt_pick = pick_labels[i]
        gt_bin = pick_bins[i]
        
        gt_img = cv2.resize(cv2.imread(os.path.join(gt_img_folder, f"{gt_pick}.jpg")), (gt_w, gt_h))
        
        tlx, tly = (gt_w * (bin_cts[gt_bin - 1] % 8)) + 3, ((gt_bin - 1) * (bg_h // 3)) + (gt_h * (i // 8)) + 3
        bg[tly : tly + gt_h, tlx : tlx + gt_w] = gt_img

        bin_cts[gt_bin - 1] += 1


        """best actual pick frame"""
        pick_tlx, pick_tly = bg_w // 2 + (i % 4) * gt_w + 5, (i // 4) * gt_h

        best_frame_path = glob.glob(os.path.join(picklist_score_folder, f"picklist_{picklist_no}_{i}_carry_item_best_frame_*.jpg"))[0]

        best_frame = cv2.resize(cv2.imread(best_frame_path), (gt_w, gt_h))
        bg[pick_tly : pick_tly + gt_h, pick_tlx : pick_tlx + gt_w] = best_frame



    return bg

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

def generate_ground_truth_images_score_based(picklist_range, htk_output_folder, pick_label_folder, score_folder):
    #HSV color model - formatted as {color key, [HSV histogram bin featvec]}
    
    #score metric, frame
    best_obj_map = dict()


    for picklist_no in range(picklist_range[0], picklist_range[1]):
        print (f"picklist_no: {picklist_no}")
        try:
            frame_boundaries = parse_action_label_file(os.path.join(htk_output_folder, 'picklist_{}.txt'.format(picklist_no)))
        
            with open(f"{pick_label_folder}/picklist_{picklist_no}_raw.txt") as infile:
                lines = [i for i in infile.read().strip().replace("\n", "")]

                pick_labels = lines[::2]
                # pick_bins = [int(_) for _ in lines[1::2]]
        except Exception as e:
            print (f"Skipping picklist {picklist_no}")
            continue

     
        #Since I haven't used HTK to generate boundaries as of now, I don't need this
        
        #number of carry item sequenecs
        picks = len(frame_boundaries['carry_item'])

        for pick_no in range(picks):
            score_matrix = np.load(os.path.join(score_folder, 'picklist_{}'.format(picklist_no), 'picklist_{}_{}_carry_item.npy'.format(picklist_no, pick_no)))
            sorted_scores = np.argsort(score_matrix[:, -1], axis = 0)
            # print(sorted_scores)
            best_score_frame_no, best_score = score_matrix[sorted_scores[-1], 0], score_matrix[sorted_scores[-1], -1]
            # print(best_score_frame_no, best_score)

            gt_object = pick_labels[pick_no]
            # best_frames.append(score_matrix[sorted_scores[-(int)(n * pct_frames) :]][:, 0].astype(int))
            
            # dist_to_centroid = distances[pred_obj]
            res = best_obj_map.get(gt_object, None)
            if res is None or (res and best_score > res[0]):
                print(f"updating {gt_object} from score {res[0] if res is not None else '-inf'} to {best_score}")
                obj_frame = cv2.imread(os.path.join(score_folder, f"picklist_{picklist_no}/picklist_{picklist_no}_{pick_no}_carry_item_best_frame_{int(best_score_frame_no)}.jpg"))
                best_obj_map[gt_object] = (best_score, obj_frame)
                
    return best_obj_map

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--picklist", type = int, required = True)
    parser.add_argument("--config_file", "-c", type=str, default="../configs/calix.yaml",
                    help="Path to experiment config (scripts/configs)")
    parser.add_argument("--display", action = "store_true")
    parser.add_argument("--color_model_path", default = "./hsv_models/hsv_bin_classification_frame_filtered_90_percent_frames_unconditional.pkl")
    parser.add_argument("--gt_img_path", default = "./item_imgs")
    parser.add_argument("--out_folder", default = "./error_checking_vis")
    args = parser.parse_args()


    configs = load_yaml_config(args.config_file)

    elan_label_folder = configs["file_paths"]["elan_annotated_file_path"]
    htk_input_folder = configs["file_paths"]["htk_input_file_path"]
    video_folder = configs["file_paths"]["video_file_path"]
    pick_label_folder = configs["file_paths"]["label_file_path"]
    htk_output_folder = configs["file_paths"]["htk_output_file_path"]
    score_folder = configs["file_paths"]["calix_score_file_path"]

    gt_img_folder = args.gt_img_path
    
    #color model trained on 136-235, generate gt color imgs on 41-90
    # gt_images = generate_ground_truth_images_score_based((41, 91), htk_output_folder, pick_label_folder, score_folder)

    
    # gt_images = generate_ground_truth_images(args.color_model_path, (41, 91), htk_input_folder, htk_output_folder, video_folder)
    
    # for k, v in gt_images.items():
    #     print(k)
    #     best_img = v[1]
    #     if best_img is not None:
    #         # print(best_img)
    #         # cv2.imshow('tmp', best_img)
    #         # if cv2.waitKey(-1) & 0xFF == ord('q'):
    #         #     continue
    #         cv2.imwrite(os.path.join(gt_img_folder, f"{k}.jpg"), best_img)

    error_check_vis = process_picklist(pick_label_folder, args.picklist, gt_img_folder, score_folder)
    
    cv2.imwrite(os.path.join(args.out_folder, f"picklist_{args.picklist}_error_checking.jpg"), error_check_vis)

    # os.path.join(raw_item_pic_folder, )