import xml.etree.ElementTree as ET
from collections import defaultdict

def get_elan_boundaries(file_name):
    # gets the generic elan boundaries (specifies the action but not the color nor the bin location that is being used)

    # Passing the path of the xml document to enable the parsing process
    tree = ET.parse(file_name)

    # getting the parent tag of the xml document
    root = tree.getroot()
    elan_boundaries = defaultdict(list)
    elan_annotations = []
    for annotation in root[2][:]:
        all_descendants = list(annotation.iter())
        for desc in all_descendants:
            if desc.tag == "ANNOTATION_VALUE":
                elan_annotations.append(desc.text)

    prev_time_value = int(root[1][1].attrib['TIME_VALUE'])/1000
    it = root[1][:]
    for index in range(0, len(it), 2):
        letter = elan_annotations[int(index/2)]
        letter_start = int(it[index].attrib['TIME_VALUE'])/1000
        letter_end = int(it[index + 1].attrib['TIME_VALUE'])/1000
        elan_boundaries[letter].append(letter_start)
        elan_boundaries[letter].append(letter_end)

    # remove the first two points from carry_empty because seems to have extra 2 elements for the period before the first pick starts (if first element is 0 then this
    # seems to be the case)
    elan_boundaries["carry_empty"] = elan_boundaries["carry_empty"][2:] if elan_boundaries["carry_empty"][0] == 0 else elan_boundaries["carry_empty"]

    return elan_boundaries

def get_generic_elan_boundaries(file_name):
    # gets the labels for the actions of the elan boundaries (i.e. ignores the color and the place bin number and only
    # takes the action of pick, carry, place, carry_empty into consideration)
    elan_boundaries = get_elan_boundaries(file_name)

    generic_elan_boundaries = {}

    for action in ["pick", "carry", "place"]:
        # get timestamps associated with action
        # need to check for empty since carry_empty is a special case
        timestamps = [i for key in elan_boundaries.keys() for i in elan_boundaries[key] if "empty" not in key and key.split("_")[0] == action]
        # sort the timestamps
        generic_elan_boundaries[action] = sorted(timestamps)

    # handling carry_empty special case
    timestamps = [i for key in elan_boundaries.keys() for i in elan_boundaries[key] if key == "carry_empty"]
    # sort the timestamps
    generic_elan_boundaries["carry_empty"] = sorted(timestamps)

    return generic_elan_boundaries



def get_htk_boundaries(file_name):

    htk_boundaries = defaultdict(list)
    with open(file_name, 'r') as f:
        lines = f.readlines()
        for line in lines[2:]:
            if line.strip() == ".":
                # terminating char
                break
            boundaries = line.split()[0:2]
            letter = line.split()[2]
            letter_start = [int(boundary)/117000 for boundary in boundaries][0]
            letter_end = [int(boundary)/117000 for boundary in boundaries][1]
            htk_boundaries[letter].append(letter_start)
            htk_boundaries[letter].append(letter_end)


    return htk_boundaries

def get_squared_error(elan_boundaries, htk_boundaries):
    """
    returns the squared error between the ELAN and HTK boundaries, along with number of points (for rmse computation)
    """
    squared_error = 0
    num_points = 0

    # elan to htk mapping
    key_swap = {'pick':'a',
                'carry':'e',
                'place':'i',
                'carry_empty':'m'} 

    for elan_key in elan_boundaries.keys():
        if elan_key == "sil":
            # ignore the sil
            continue
        # get the elan values and compare them to the htk ones
        htk_key = key_swap[elan_key]
        for i in range(0, len(elan_boundaries[elan_key]), 2):
            # we only want to compare one of them since the end point is the start point for another letter (i.e. double counting)
            start_elan, end_elan = elan_boundaries[elan_key][i], elan_boundaries[elan_key][i + 1]
            start_htk, end_htk = htk_boundaries[htk_key][i], htk_boundaries[htk_key][i + 1]
                
            num_points += 1
            squared_error += (start_elan - start_htk) ** 2

    return squared_error, num_points
