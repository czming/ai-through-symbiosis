# using the object representation learnt to make predictions on picks that we have not seen before

import pickle
import argparse
import matplotlib.pyplot as plt
from sklearn import metrics
import logging
from sklearn.utils.multiclass import unique_labels
from util import parse_action_label_file

import sys
sys.path.append("..")
from utils import *
sys.path.append("./calix_thesis")

import cv2


def find_closest_in_set(vector, vector_dict, pick_labels=None):
    # find the vector in vector_dict that has the smallest distance to vector and return the key for that vector

    if pick_labels is None:
        # pick labels were not passed in, no restrictions on picks that can be used
        pick_labels = list(vector_dict.keys())
    else:
        pick_labels = list(pick_labels)


    # get distances of each vector in vector_dict to vector
    vector_distances = {key: np.linalg.norm((vector_dict[key][0] - vector)) for key in vector_dict.keys()}

    return min(pick_labels, key=lambda x: vector_distances[x]), vector_distances


    

def object_classification_no_filtering(config_file, picklist_start, picklist_end):
    # change depending on configs
    PICKLISTS = range(picklist_start, picklist_end) #range(41, 91)

    configs = load_yaml_config(config_file)

    # elan_label_folder = configs["file_paths"]["elan_annotated_file_path"]
    htk_input_folder = configs["file_paths"]["htk_input_file_path"]
    # video_folder = configs["file_paths"]["video_file_path"]
    pick_label_folder = configs["file_paths"]["label_file_path"]
    htk_output_folder = configs["file_paths"]["htk_output_file_path"]
    # score_folder = configs["file_paths"]["calix_score_file_path"]

    with open("hsv_bin_classification_normal.pkl", "rb") as infile:
        objects_type_hsv_bins = pickle.load(infile)


    object_hsv_representation = objects_type_hsv_bins

    rmse_errors = []

    predicted_picklists = []
    actual_picklists = []

    # sum_squared_error = 0
    action_count = 0

    for picklist_no in PICKLISTS:
        print (f"picklist_no: {picklist_no}")
        try:
            with open(f"{pick_label_folder}/picklist_{picklist_no}_raw.txt") as infile:
                pick_labels = [i for i in infile.read().replace("\n", "")[::2]]

            with open(f"{htk_input_folder}/picklist_{picklist_no}.txt") as infile:
                htk_inputs = [i.split() for i in infile.readlines()]
                # print(htk_inputs)

            # Since I annotated with raw frame count info, I define my own parsing method
            # htk_boundaries = get_htk_boundaries(f"{htk_output_folder}/results-{picklist_no}")
            frame_boundaries = parse_action_label_file(os.path.join(htk_output_folder, 'picklist_{}.txt'.format(picklist_no)))
        except Exception as e:
            print (f"Skipping picklist {picklist_no}")
            continue

     
        #Since I haven't used HTK to generate boundaries as of now, I don't need this
        
        # # check with rmse to see if reasonable
        # general_elan_boundaries = get_elan_boundaries_general(f"{elan_label_folder}/picklist_{picklist_no}.eaf")
        # se, count = get_squared_error(general_elan_boundaries, htk_boundaries)

        # logging.debug(f"squared_error, count: {se}, {count}")

        # rmse_errors.append((se / count) ** 0.5)

        # sum_squared_error += se
        # action_count += count
        
        #pick frames are the carry_item frames
        pick_frames = frame_boundaries['carry_item']

        # # looking through the various picks
        # for i in range(0, len(frame_boundaries["carry_item"])):
        #     # collect the red frames
        #     start_frame = math.ceil(float(htk_boundaries["e"][i]) * 59.94)
        #     end_frame = math.ceil(float(htk_boundaries["e"][i + 1]) * 59.94)
        #     pick_frames.append([start_frame, end_frame])


        empty_hand_frames = frame_boundaries['carry_empty']

        # for i in range(0, len(htk_boundaries[empty_hand_label]), 2):
        #     # collect the red frames
        #     start_frame = math.ceil(float(htk_boundaries[empty_hand_label][i]) * 59.94)
        #     end_frame = math.ceil(float(htk_boundaries[empty_hand_label][i + 1]) * 59.94)
        #     empty_hand_frames.append([start_frame, end_frame])

        # getting the sum, multiply average by counts
        sum_empty_hand_hsv = np.zeros(180)
        empty_hand_frame_count = 0

        # # avg hsv bins for each pick
        # num_hand_detections = [get_avg_hsv_bin_frames(htk_inputs, start_frame + 5, end_frame - 5)[1] if (end_frame - start_frame) > 10 else for (start_frame, end_frame) \
        #                        in empty_hand_frames]
        # if 0 in num_hand_detections:
        #     print("skipping bad boundaries")
        #     continue

        for (start_frame, end_frame) in empty_hand_frames:
            if end_frame - start_frame < 10:
                continue
            # print (start_frame, end_frame) #in my data there are none
            # print (len(htk_inputs))
            # cut out 5 frames if 30fps and 10 frames if 60fps
            curr_avg_empty_hand_hsv, frame_count = get_avg_hsv_bin_frames(htk_inputs, start_frame, end_frame)
            sum_empty_hand_hsv += (curr_avg_empty_hand_hsv * frame_count)
            empty_hand_frame_count += frame_count

        avg_empty_hand_hsv = sum_empty_hand_hsv / np.sum(sum_empty_hand_hsv)

        avg_hsv_picks = [get_avg_hsv_bin_frames(htk_inputs, start_frame + 5, end_frame - 5)[0] - avg_empty_hand_hsv for (start_frame, end_frame) \
                        in pick_frames]

        pred_labels = []

        for index, i in enumerate(avg_hsv_picks):
            print(pick_labels[index])
            i = collapse_hue_bins(i, [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165], 15)


            pred, distances = find_closest_in_set(i, object_hsv_representation, set(pick_labels))

            pred_labels.append(pred)

        logging.debug(pred_labels, pick_labels)

        predicted_picklists.extend(pred_labels)
        actual_picklists.extend(pick_labels)

    confusion_matrix = metrics.confusion_matrix(actual_picklists, predicted_picklists)
    appeared_objects = set()

    for i in actual_picklists:
        for j in i:
            appeared_objects.add(j)

    for i in predicted_picklists:
        for j in i:
            appeared_objects.add(j)


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
    names = ['red', 'green', 'blue', 'darkblue', 'darkgreen', 'orange', 'alligatorclip', 'yellow', 'clear', 'candle']

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
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = conf_mat_norm, display_labels = unique_names)

    cm_display.plot(cmap=plt.cm.Blues)

    plt.xticks(rotation=90)

    plt.tight_layout()

    fig1 = plt.gcf()

    plt.show()

    fig1.savefig("object_classification_no_filtering.jpg")

    # print (sum_squared_error, action_count)

    print (sum([1 if predicted_picklists[i] == actual_picklists[i] else 0 for i in range(len(predicted_picklists))]) / len(predicted_picklists))

    print (rmse_errors) 

def object_classification_carry_item_filtering(config_file, picklist_start, picklist_end):
    # change depending on configs
    PICKLISTS = range(picklist_start, picklist_end) #range(41, 91)

    configs = load_yaml_config(config_file)

    # elan_label_folder = configs["file_paths"]["elan_annotated_file_path"]
    htk_input_folder = configs["file_paths"]["htk_input_file_path"]
    video_folder = configs["file_paths"]["video_file_path"]
    pick_label_folder = configs["file_paths"]["label_file_path"]
    htk_output_folder = configs["file_paths"]["htk_output_file_path"]
    score_folder = configs["file_paths"]["calix_score_file_path"]

    with open("object_type_hsv_bins_copy.pkl", "rb") as infile:
        objects_type_hsv_bins = pickle.load(infile)


    object_hsv_representation = objects_type_hsv_bins

    rmse_errors = []

    predicted_picklists = []
    actual_picklists = []

    # sum_squared_error = 0
    action_count = 0

    for picklist_no in PICKLISTS:
        print (f"picklist_no: {picklist_no}")
        try:
            with open(f"{pick_label_folder}/picklist_{picklist_no}_raw.txt") as infile:
                pick_labels = [i for i in infile.read().replace("\n", "")[::2]]

            with open(f"{htk_input_folder}/picklist_{picklist_no}.txt") as infile:
                htk_inputs = [i.split() for i in infile.readlines()]
                # print(htk_inputs)

            # Since I annotated with raw frame count info, I define my own parsing method
            # htk_boundaries = get_htk_boundaries(f"{htk_output_folder}/results-{picklist_no}")
            frame_boundaries = parse_action_label_file(os.path.join(htk_output_folder, 'picklist_{}.txt'.format(picklist_no)))
        except Exception as e:
            print (f"Skipping picklist {picklist_no}")
            continue

     
        #Since I haven't used HTK to generate boundaries as of now, I don't need this
        
        # # check with rmse to see if reasonable
        # general_elan_boundaries = get_elan_boundaries_general(f"{elan_label_folder}/picklist_{picklist_no}.eaf")
        # se, count = get_squared_error(general_elan_boundaries, htk_boundaries)

        # logging.debug(f"squared_error, count: {se}, {count}")

        # rmse_errors.append((se / count) ** 0.5)

        # sum_squared_error += se
        # action_count += count
        
        #number of carry item sequenecs
        picks = len(frame_boundaries['carry_item'])

        empty_hand_frames = frame_boundaries['carry_empty']
        # getting the sum, multiply average by counts
        sum_empty_hand_hsv = np.zeros(180)
        empty_hand_frame_count = 0

        # # avg hsv bins for each pick
        # num_hand_detections = [get_avg_hsv_bin_frames(htk_inputs, start_frame + 5, end_frame - 5)[1] if (end_frame - start_frame) > 10 else for (start_frame, end_frame) \
        #                        in empty_hand_frames]
        # if 0 in num_hand_detections:
        #     print("skipping bad boundaries")
        #     continue

        for (start_frame, end_frame) in empty_hand_frames:
            if end_frame - start_frame < 10:
                continue
            # print (start_frame, end_frame) #in my data there are none
            # print (len(htk_inputs))
            # cut out 5 frames if 30fps and 10 frames if 60fps
            curr_avg_empty_hand_hsv, frame_count = get_avg_hsv_bin_frames(htk_inputs, start_frame, end_frame)
            sum_empty_hand_hsv += (curr_avg_empty_hand_hsv * frame_count)
            empty_hand_frame_count += frame_count

        avg_empty_hand_hsv = sum_empty_hand_hsv / np.sum(sum_empty_hand_hsv)


        best_frames = []
        for pick_no in range(picks):
            score_matrix = np.load(os.path.join(score_folder, 'picklist_{}'.format(picklist_no), 'picklist_{}_{}.npy'.format(picklist_no, pick_no)))
            sorted_scores = np.argsort(score_matrix[:, -1], axis = 0)
            #best 20 frames
            best_frames.append(score_matrix[sorted_scores[-20:]][:, 0].astype(int))
        #list of 20 best frames for each pick
        avg_hsv_picks = [get_avg_hsv_bin_frames_frame_list(htk_inputs, frame_list)[0] - avg_empty_hand_hsv for frame_list in best_frames]

        # cap = cv2.VideoCapture(os.path.join(video_folder, 'picklist_{}.mp4'.format(picklist_no)))
        # for frame_list in best_frames:
        #     for frame in frame_list:
        #         cap.set(1, frame)
        #         success, image = cap.read()
        #         image = cv2.resize(image, (image.shape[1] // 2, image.shape[0] // 2))
        #         cv2.imshow('e', image)
        #         if cv2.waitKey(-1) & 0xFF == ord('q'):
        #             pass

        pred_labels = []

        for index, i in enumerate(avg_hsv_picks):
            # print(pick_labels[index])#ground truth
            i = collapse_hue_bins(i, [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165], 15)

            pred, distances = find_closest_in_set(i, object_hsv_representation, set(pick_labels))

            pred_labels.append(pred)

        logging.debug(pred_labels, pick_labels)
        print(pick_labels, pred_labels)

        predicted_picklists.extend(pred_labels)
        actual_picklists.extend(pick_labels)

    confusion_matrix = metrics.confusion_matrix(actual_picklists, predicted_picklists)
    appeared_objects = set()

    for i in actual_picklists:
        for j in i:
            appeared_objects.add(j)

    for i in predicted_picklists:
        for j in i:
            appeared_objects.add(j)

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
    names = ['red', 'green', 'blue', 'darkblue', 'darkgreen', 'orange', 'alligatorclip', 'yellow', 'clear', 'candle']

    actual_picklists_names = []
    predicted_picklists_names = []

    letter_counts = defaultdict(lambda: 0)
    for letter in actual_picklists:
        name = letter_to_name[letter]
        letter_counts[name] += 1
        actual_picklists_names.append(name)

    for index, letter in enumerate(predicted_picklists):
        # print(letter)
        predicted_picklists_names.append(letter_to_name[letter])
    confusion_matrix = metrics.confusion_matrix(actual_picklists_names, predicted_picklists_names)
    conf_mat_norm = (confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis])

    unique_names = unique_labels(actual_picklists_names, predicted_picklists_names)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = conf_mat_norm, display_labels = unique_names)

    cm_display.plot(cmap=plt.cm.Blues)

    plt.xticks(rotation=90)

    plt.tight_layout()
    fig1 = plt.gcf()

    plt.show()

    fig1.savefig("object_classification_carry_item_filtering.jpg")

    # print (sum_squared_error, action_count)

    print (sum([1 if predicted_picklists[i] == actual_picklists[i] else 0 for i in range(len(predicted_picklists))]) / len(predicted_picklists))

    # print (rmse_errors) 
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=str, help="Path to experiment config (scripts/configs)", default = '../configs/calix.yaml')
    args = parser.parse_args()

    # object_classification_no_filtering(args.config, 275, 355)

    object_classification_carry_item_filtering(args.config, 275, 355)
    # object_classification_carry_empty_filtering(args.config)
    # object_classification_full_filtering(args.config)