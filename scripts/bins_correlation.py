"""

(DEPRACATED, beyond here there be dragons)

creates an indicator variable sequence indicating whether a particular action (pick red, carry blue)
is taking place and the value for each of the inputs

"""

import numpy as np

import xml.etree.ElementTree as ET

import logging

logging.getLogger().setLevel(logging.DEBUG)

ELAN_ANNOTATION_FILE_FORMAT = '../elan_annotated/picklist_{}.eaf'

HTK_INPUT_FILE_FORMAT = "../htk_inputs/picklist_{}_forward_filled_30_gaussian_filter_9_3.txt"

PICK_LABEL_FILE_FORMAT = "../Labels/picklist_{}_raw.txt"

# store the correlations coefficients in an output CSV so easier to look at and Excel if needed
OUTPUT_FILE = "outputs/bins_correlation_output.csv"

# indices of the pick lists that we want to process
PICKLISTS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 17, 18, 19, 20, 21, 22, 26]

def add_curr_data(data_bins, data):
    # gather all of the data in the current run into the data bins
    for i, data_bin in enumerate(data_bins):
        data_bins[i] = np.concatenate((data_bin, data[:, i]))
    

# stores the arrays for the indicator variables
indicator_arrays = {}

# gather all the data across the different picklists and store them here for correlation
# calculation together with the indicator arrays
data_bins = []

# populate the indicator_arrays dictionary, 0 - pick, 1 - carry, 2 - place, 3 - carry empty
for color in ('r', 'g', 'b'):
    for action_index in range(4):
        indicator_arrays[(color, action_index)] = np.array([])

# looking at indicator for action associated to that color
for color in ('r', 'g', 'b'):
    indicator_arrays[(color, '')] = np.array([])

for picklist in PICKLISTS:
    elan_annotation_file = ELAN_ANNOTATION_FILE_FORMAT.format(picklist)
    htk_input_file = HTK_INPUT_FILE_FORMAT.format(picklist)
    pick_label_file = PICK_LABEL_FILE_FORMAT.format(picklist)

    data = np.genfromtxt(htk_input_file, delimiter=" ")

    # if this is the first picklist being processed, then we initialize the bins arrays
    # based on the number of bins in the data (assume all data has same number of bins)
    if len(data_bins) == 0:
        data_bins = [np.array([]) for i in range(data.shape[1])]

    add_curr_data(data_bins, data)

    
    # get ELAN labels for the events
    elan_tree = ET.parse(elan_annotation_file)
    root = elan_tree.getroot()
    elan_annotations = []
    it = root[1][:]
    event_frames = []

    for index in it:
        event_frames.append(int(int(index.attrib['TIME_VALUE']) * 60/1000))

    event_frames = event_frames[2:]

    # preparing the indicator arrays for the current pick list (creating them so we can
    # flip those within the range of the picks)
    curr_indicator_arrays = {}

    for key in indicator_arrays.keys():
        curr_indicator_arrays[key] = np.zeros(data.shape[0])

        
    # get the labelled colors for the picks
    with open(pick_label_file, "r") as pick_file:
        pick_labels = pick_file.read()[::2]

    if len(pick_labels) * 8 != len(event_frames):
        raise Exception("Number of pick labels doesn't match event frames, should have 8 event frames for each pick label")

    for pick_index, pick_label in enumerate(pick_labels):
        # go through each pick and draw the timestamp from the event frames to
        # alter the indicator_arrays
        for action_index in range(4):
            # can directly get the values from the events that were recorded
            # 8 events for one pick and 2 events for each action
            start_frame = event_frames[pick_index * 8 + action_index * 2]
            end_frame = event_frames[pick_index * 8 + action_index * 2 + 1]
            # might have some bounds issues, not sure if end frame is intended to be inclusive or
            # exclusive
            curr_indicator_arrays[(pick_label, action_index)][start_frame: end_frame] = 1
            curr_indicator_arrays[(pick_label, '')][start_frame: end_frame] = 1

    # add debug logs just to double check things
    logging.debug("curr_indicator_array:")
    for key, value in curr_indicator_arrays.items():
        logging.debug("%s %d", str(key), value.sum())
    logging.debug("Picklist: %s", pick_labels)
    logging.debug("Events: %s", event_frames)

    logging.debug("indicator_array:")

    # concatenate indicator variables to existing arrays
    for key in indicator_arrays.keys():
        indicator_arrays[key] = np.concatenate((indicator_arrays[key], curr_indicator_arrays[key]))
        # use the new value of the value
        logging.debug("%s %d", str(key), indicator_arrays[key].sum())

# need to compare correlation with the different bins and the different indicator variables
with open(OUTPUT_FILE, "w") as outfile:
    # first column blank for the row headers
    outfile.write("," + ",".join([str(i) for i in range(len(data_bins))]) + "\n")
    for key, value in indicator_arrays.items():
        # write out the key that we are doing this for
        outfile.write("".join([str(i) for i in key]) + ",")
        max_bin_index = 0
        max_corr = float("-inf")
        curr_corr = []
        for data_bin_index, data_bin in enumerate(data_bins):
            # compute the correlation coefficient of the indicator variable for the curr key
            # and the curr data_bin

            # take the correlation since we don't need the diagonal and it's a symmetric matrix
            # corr = np.corrcoef(np.log(data_bin, where=(data_bin != 0)), value)[0][1]  # using log of the values
            corr = np.corrcoef(data_bin, value)[0][1]
            outfile.write(str(corr) + ",")
            # want to look only at hsv bins which are 1 to 20
            if data_bin_index in range(1, 21) and abs(corr) >= max_corr:
                max_corr = abs(corr)
                max_bin_index = data_bin_index
            curr_corr.append(corr)
        outfile.write(str(max_bin_index) + ",")
        outfile.write(str(curr_corr[max_bin_index]) + ",")
        outfile.write("\n")
    
