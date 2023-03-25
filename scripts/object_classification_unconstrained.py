# using the object representation learnt to make predictions on picks that we have not seen before

from utils import *
import pickle
import argparse
import matplotlib.pyplot as plt
from collections import Counter
from sklearn import metrics

from sklearn.utils.multiclass import unique_labels


def find_closest_in_set(vector, vector_dict):
    # find the vector in vector_dict that has the smallest distance to vector and return the key for that vector

    # to restrict the objects that we want to consider
    # vector_dict = {key: vector_dict[key] for key in ["r", "g", "b"]}

    # collapsed_vector_dict = {}

    # for key in vector_dict.keys():
    #     curr_vector = vector_dict[key]
    #     print (key, curr_vector)
    #     # for some reason wraparound doesn't seem to work in numpy array
    #     blue = sum(curr_vector[0:2]) + curr_vector[2] / 2 + curr_vector[8] / 2 + curr_vector[9]
    #     green = sum(curr_vector[2:4]) + curr_vector[4] / 2 + curr_vector[1] / 2
    #     red = sum(curr_vector[4:8]) + curr_vector[8] / 2 + curr_vector[4] / 2
    #     collapsed_vector_dict[key] = np.array([blue, green, red])
    #
    # blue = sum(vector[0:2]) + vector[2] / 2 + vector[8] / 2 + vector[9]
    # green = sum(vector[2:4]) + vector[4] / 2 + vector[1] / 2
    # red = sum(vector[5:8]) + vector[8] / 2 + vector[5] / 2
    #
    # collapsed_vector = np.array([blue, green, red])
    # print (collapsed_vector_dict)
    #


    # plt.bar([i + 0.3 for i in range(10)], vector[:10], width=0.2, color='black')
    # plt.bar([i + 0.1 for i in range(10)], vector_dict["r"][:10], width=0.2, color='r')
    # plt.bar([i - 0.1 for i in range(10)], vector_dict["g"][:10], width=0.2, color='g')
    # plt.bar([i - 0.3 for i in range(10)], vector_dict["b"][:10], width=0.2, color='b')
    #
    # plt.show()
    #
    # plt.bar([i + 0.3 for i in range(3)], collapsed_vector, width=0.2, color='black')
    # plt.bar([i + 0.1 for i in range(3)], collapsed_vector_dict["r"], width=0.2, color='r')
    # plt.bar([i - 0.1 for i in range(3)], collapsed_vector_dict["g"], width=0.2, color='g')
    # plt.bar([i - 0.3 for i in range(3)], collapsed_vector_dict["b"], width=0.2, color='b')
    #
    # plt.show()

    # get the distances of each vector in vector_dict to vector
    vector_distances = {key: np.linalg.norm((vector_dict[key][0] - vector) / vector_dict[key][1]) for key in vector_dict.keys()}

    print (vector_distances)

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

# with open("objects_hsv_bin_accumulator.pkl", "rb") as infile:
#     objects_hsv_bin_accumulator = pickle.load(infile)

with open("object_type_hsv_bins.pkl", "rb") as infile:
    objects_type_hsv_bins = pickle.load(infile)

# object_hsv_representation = {key: value[0] / value[1] for key, value in objects_hsv_bin_accumulator.items()}

object_hsv_representation = objects_type_hsv_bins

print (len(object_hsv_representation['r']))

# one off thing, don't change configs
ood_htk_outputs_folder = "C:/Users/chngz/OneDrive/Georgia Tech/AI Through Symbiosis/pick_list_dataset/htk_outputs/ood-1-90-results/ood-1-90-results"

PICKLISTS = range(1, 41)

predicted_picklists = []
actual_picklists = []

sum_squared_error = 0
action_count = 0

for picklist_no in PICKLISTS:

    with open(f"{pick_label_folder}/picklist_{picklist_no}_raw.txt") as infile:
        pick_labels = [i for i in infile.read().replace("\n", "")[::2]]

    with open(f"{htk_input_folder}/picklist_{picklist_no}.txt") as infile:
        htk_inputs = [i.split() for i in infile.readlines()]

    htk_boundaries = get_htk_boundaries(f"{ood_htk_outputs_folder}/results-{picklist_no}")

    # check with rmse to see if reasonable
    general_elan_boundaries = get_elan_boundaries_general(f"{elan_label_folder}/picklist_{picklist_no}.eaf")
    se, count = get_squared_error(general_elan_boundaries, htk_boundaries)

    print (se, count)

    sum_squared_error += se
    action_count += count

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
    sum_empty_hand_hsv = np.zeros(180)
    empty_hand_frame_count = 0

    for (start_frame, end_frame) in empty_hand_frames:
        curr_avg_empty_hand_hsv, frame_count = get_avg_hsv_bin_frames(htk_inputs, start_frame, end_frame)
        sum_empty_hand_hsv += (curr_avg_empty_hand_hsv * frame_count)
        empty_hand_frame_count += frame_count

    avg_empty_hand_hsv = sum_empty_hand_hsv / np.sum(sum_empty_hand_hsv)

    # plt.bar(range(10), avg_empty_hand_hsv[:10])
    # plt.show()
    avg_hsv_picks = [get_avg_hsv_bin_frames(htk_inputs, start_frame, end_frame)[0] / np.sum(get_avg_hsv_bin_frames(htk_inputs, start_frame, end_frame)[0]) - avg_empty_hand_hsv for (start_frame, end_frame) \
                     in pick_frames]

    ## if given picklist
    # all_permutations = []
    #
    # pick_label_count = dict(Counter(pick_labels))
    #
    # generate_permutations_from_dict(pick_label_count, all_permutations)
    #
    # best_result = (float('inf'), None)
    # for permutation_index in range(len(all_permutations)):
    #     # iterate through the different permutations
    #     permutation = all_permutations[permutation_index]
    #
    #     sum_distance = 0
    #
    #     for index, value in enumerate(permutation):
    #         # looking at the distance between the hsv vector at that and the assigned label which the permutation
    #         # provides
    #         sum_distance += np.linalg.norm(avg_hsv_picks[index][:10] - object_hsv_representation[value][:10])
    #
    #
    #     if sum_distance < best_result[0]:
    #         best_result = (sum_distance, permutation)



    pred_labels = []

    for index, i in enumerate(avg_hsv_picks):
        print(pick_labels[index])
        i = collapse_hue_bins(i, [0, 30, 60, 90, 120, 150], 10)


        pred, distances = find_closest_in_set(i, object_hsv_representation)


        print (pred)

        pred_labels.append(pred)

    print (pred_labels, pick_labels)

    predicted_picklists.extend(pred_labels)
    actual_picklists.extend(pick_labels)

confusion_matrix = metrics.confusion_matrix(actual_picklists, predicted_picklists)
print(actual_picklists)
print(predicted_picklists)
appeared_objects = set()

for i in actual_picklists:
    for j in i:
        appeared_objects.add(j)

for i in predicted_picklists:
    for j in i:
        appeared_objects.add(j)

from sklearn.utils.multiclass import unique_labels

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

plt.show()

plt.tight_layout()

plt.savefig("object_classification.png")


print ("test")
print (sum_squared_error, action_count)

print (sum([1 if predicted_picklists[i] == actual_picklists[i] else 0 for i in range(len(predicted_picklists))]) / len(predicted_picklists))
