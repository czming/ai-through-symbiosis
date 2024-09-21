"""

visualizes data with the x axis as the row and the y-axis as the selected
column values

"""

import numpy as np
import matplotlib.pyplot as plt

import xml.etree.ElementTree as ET

PICKLIST_NO = 1

ELAN_ANNOTATION_FILE = f'../elan_annotated/picklist_{PICKLIST_NO}.eaf'

HTK_INPUT_FILE = f"""../htk_inputs/picklist_{PICKLIST_NO}_forward_filled_30_gaussian_filter_9_3.txt"""

PICK_LABEL_FILE = f"../Labels/picklist_{PICKLIST_NO}_raw.txt"

# choose the index of the columns that we want to visualize
VISUALIZED_COLUMNS = [0, -4,-3, -2, -1]


# frames per second of the camera to correspond between the event timings and the frames
FPS = 60

# ----------------------------- retrieving ELAN annotations in frame units--------------------------------

elan_tree = ET.parse(ELAN_ANNOTATION_FILE)
root = elan_tree.getroot()
elan_annotations = []
it = root[1][:]
event_frames = []

for index in it:
    event_frames.append(int(int(index.attrib['TIME_VALUE']) * 60/1000))

# getting the ending time based on the htk_inputs


# ----------------------- processing htk_input data and the columns to be visualized --------------------

with open(PICK_LABEL_FILE, "r") as pick_file:
    pick_labels = pick_file.read()[::2]


# matplotlib seems to use rgb instead of opencv bgr
pick_labels_color = {"b": (0, 0, 1), "g": (0, 1, 0), "r": (1, 0, 0)}

# black color starting point
# points_color = [(0, 0, 0)]
points_color = []

# added to account for the carry_empty at the start and left as grey since no color object associated
points_color.append((0.5, 0.5, 0.5))
points_color.append((0.5, 0.5, 0.5))

for pick_label in pick_labels:
    for i in range(8):
        # 8 labels, since 4 actions and includes (start, end)
        points_color.append(pick_labels_color[pick_label])


data = np.genfromtxt(HTK_INPUT_FILE, delimiter=" ")

fig, axs = plt.subplots(len(VISUALIZED_COLUMNS), 1)

# add point at the last frame to indicate that this is the ending 'sil', to get the last ending point
# event_frames.append(len(data))
# points_color.append((0, 0, 0))

print (event_frames)

print (len(points_color))

if len(VISUALIZED_COLUMNS) == 1:
    # axs will be a single element if only one column visualized
    axs.plot(range(len(data)), data[:, VISUALIZED_COLUMNS[0]])
    axs.scatter(x = event_frames, y = [0] * len(event_frames), c = points_color)
else:
    for i, column in enumerate(VISUALIZED_COLUMNS):
        curr_data = data[:, column]

        curr_data_min = curr_data.min()
        curr_data_max = curr_data.max()

        # axs is 1D if only one dimension is used (even if the second dimension exists but is just 1,
        # which is the case here)
        axs[i].plot(range(len(data)), curr_data)
        axs[i].scatter(x = event_frames, y = [(curr_data_max + curr_data_min) / 2] * len(event_frames), c = points_color)


plt.show()
