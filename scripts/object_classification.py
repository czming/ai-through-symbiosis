# using the object representation learnt to make predictions on picks that we have not seen before

from utils import *
import pickle
import argparse
import matplotlib.pyplot as plt
from collections import Counter
from sklearn import metrics

def find_closest_in_set(vector, vector_dict):
    # find the vector in vector_dict that has the smallest distance to vector and return the key for that vector

    # get the distances of each vector in vector_dict to vector
    vector_distances = {key: np.linalg.norm(vector_dict[key] - vector) for key in vector_dict.keys()}

    return min(vector_distances.keys(), key=lambda x: vector_distances[x]), vector_distances



parser = argparse.ArgumentParser()
parser.add_argument("--config", "-c", type=str, default="configs/zm.yaml", help="Path to experiment config (scripts/configs)")
args = parser.parse_args()


configs = load_yaml_config(args.config)

elan_label_folder = configs["file_paths"]["elan_annotated_file_path"]
htk_input_folder = configs["file_paths"]["htk_input_file_path"]
video_folder = configs["file_paths"]["video_file_path"]
pick_label_folder = configs["file_paths"]["label_file_path"]


htk_output_folder = configs["file_paths"]["htk_output_file_path"]

# read the object representation in

with open("objects_hsv_bin_accumulator.pkl", "rb") as infile:
    objects_hsv_bin_accumulator = pickle.load(infile)

object_hsv_representation = {key: value[0] / value[1] for key, value in objects_hsv_bin_accumulator.items()}

# one off thing, don't change configs
ood_htk_outputs_folder = "C:/Users/chngz/OneDrive/Georgia Tech/AI Through Symbiosis/pick_list_dataset/" + \
                         "htk_outputs/ood-1-90-results/ood-1-90-results/"

PICKLISTS = list(range(1, 41))

predicted_picklists = []
actual_picklists = []

for picklist_no in PICKLISTS:

    with open(f"{pick_label_folder}/picklist_{picklist_no}_raw.txt") as infile:
        pick_labels = [i for i in infile.read().replace("\n", "")[::2]]

    with open(f"{htk_input_folder}/picklist_{picklist_no}.txt") as infile:
        htk_inputs = [i.split() for i in infile.readlines()]

    htk_boundaries = get_htk_boundaries(f"{ood_htk_outputs_folder}/results-{picklist_no}")

    # check with rmse to see if reasonable
    # general_elan_boundaries = get_elan_boundaries_general(f"{elan_label_folder}/picklist_{picklist_no}.eaf")
    # print (get_squared_error(general_elan_boundaries, htk_boundaries))

    pick_frames = []

    # looking through the various picks
    for i in range(0, len(htk_boundaries["e"]), 2):
        # collect the red frames
        start_frame = math.ceil(float(htk_boundaries["e"][i]) * 29.97)
        end_frame = math.ceil(float(htk_boundaries["e"][i + 1]) * 29.97)
        pick_frames.append([start_frame, end_frame])

    empty_hand_label = "m"

    empty_hand_frames = []

    for i in range(0, len(htk_boundaries[empty_hand_label]), 2):
        # collect the red frames
        start_frame = math.ceil(float(htk_boundaries[empty_hand_label][i]) * 29.97)
        end_frame = math.ceil(float(htk_boundaries[empty_hand_label][i + 1]) * 29.97)
        empty_hand_frames.append([start_frame, end_frame])

    # getting the sum, multiply average by counts
    sum_empty_hand_hsv = np.zeros(20)
    empty_hand_frame_count = 0

    for (start_frame, end_frame) in empty_hand_frames:
        curr_avg_empty_hand_hsv, frame_count = get_avg_hsv_bin_frames(htk_inputs, start_frame, end_frame)
        sum_empty_hand_hsv += (curr_avg_empty_hand_hsv * frame_count)
        empty_hand_frame_count += frame_count

    avg_empty_hand_hsv = sum_empty_hand_hsv / empty_hand_frame_count

    avg_hsv_picks = [get_avg_hsv_bin_frames(htk_inputs, start_frame, end_frame)[0] - avg_empty_hand_hsv for (start_frame, end_frame) \
                     in pick_frames]

    all_permutations = []

    pick_label_count = dict(Counter(pick_labels))

    generate_permutations_from_dict(pick_label_count, all_permutations)

    best_result = (float('inf'), None)
    for permutation_index in range(len(all_permutations)):
        # iterate through the different permutations
        permutation = all_permutations[permutation_index]

        sum_distance = 0

        for index, value in enumerate(permutation):
            # looking at the distance between the hsv vector at that and the assigned label which the permutation
            # provides
            sum_distance += np.linalg.norm(avg_hsv_picks[index] - object_hsv_representation[value])

        if sum_distance < best_result[0]:
            best_result = (sum_distance, permutation)

    print (best_result[1], pick_labels)

    predicted_picklists.extend(best_result[1])
    actual_picklists.extend(pick_labels)

    # pred_labels = []

    # for i in avg_hsv_picks:
    #     pred, distances = find_closest_in_set(i, object_hsv_representation)
    #     pred_labels.append(pred)
    #
    # print (pred_labels, pick_labels)

confusion_matrix = metrics.confusion_matrix(actual_picklists, predicted_picklists)
print(actual_picklists)
print(predicted_picklists)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = ["b", "g", "r"])
cm_display.plot()
plt.show()

