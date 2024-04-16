import xml.etree.ElementTree as ET
from collections import defaultdict
import random
import itertools


def get_elan_boundaries(file_name):
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
                if 'empty' not in desc.text:
                    elan_annotations.append(desc.text[0:desc.text.index("_")])
                else:
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

htk_labels_to_elan = {'a':'pick',
            'e':'carry',
            'i':'place',
            'm':'carry_empty'} 

def get_ground_truth_carry_items(picklist_number):
    carry_items = []
    with open("./full-labels/picklist_" + str(picklist_number) + ".lab", 'r') as f:
        lines = f.readlines()
        for i in lines[2:][0::4]:
            carry_items.append(i.replace("\n",""))
    return carry_items

def get_picklist_permutations(letters):
    permutations = itertools.permutations(letters)
    permutations = set(list(permutations)) # convert to set of tuples
    permutations = [list(ele) for ele in permutations] #convert into list of lists
    return permutations

def get_combinations_of_permutations(picklists_permutations):
    product = itertools.product(picklists_permutations[1], picklists_permutations[0])
    print(len(picklists_permutations[2:]))
    for index, pick_permutation in enumerate(picklists_permutations[2:]):
        print(index)
        product = itertools.product(pick_permutation, product)
    return product

def unreverse_combination_picklists(combinations):
    all_combinations = []
    for element in combinations:
        curr_tuple = element
        picklist_array = []
        while isinstance(curr_tuple, tuple):
            picklist_array.append(curr_tuple[0])
            curr_tuple = curr_tuple[1]
        picklist_array.append(curr_tuple)
        all_combinations.append(picklist_array[::-1])
    return all_combinations

def write_permutations():
    ground_truth_carry_picklists = []
    # for i in range(1, 41):
    for i in range(1, 41):
        ground_truth_carry_picklists.append(get_ground_truth_carry_items(i))
    picklists_permutations = []
    for index, ground_truth_carry_picklist in enumerate(ground_truth_carry_picklists):
        picklists_permutations.append(get_picklist_permutations(ground_truth_carry_picklist))
    multiples = 1
    for picklist_permutation in picklists_permutations:
        multiples = multiples * len(picklist_permutation)
    print(multiples)
    # combinations = get_combinations_of_permutations(picklists_permutations)
    # all_combinations = unreverse_combination_picklists(combinations)
    # for combination in all_combinations:
    #     print(combination)
    # print(len(all_combinations))
# write_permutations()

def get_permuted_carry_items(permutation_number, picklist_number):
    permuted_carry_items = []
    with open("../full-labels-permutations/" + "/" + str(picklist_number) + ".lab", 'r') as f:
        lines = f.readlines()
        for i in lines:
            permuted_carry_items.append(i.replace("\n",""))
    return permuted_carry_items
    

def get_htk_boundaries(file_name):

    carry_items = get_ground_truth_carry_items(file_name[file_name.rindex("-")+1:])
    out_of_place = 0

    # if file_name[file_name.rindex("-")+1:] == "3":
    
    # shuffled_carry_items = get_ground_truth_carry_items(file_name[file_name.rindex("-")+1:])
    # random.shuffle(shuffled_carry_items)
    # for index, item in enumerate(carry_items):
    #     same_item = carry_items[index] == shuffled_carry_items[index]
    #     if not same_item:
    #         out_of_place += 1
    # carry_items = shuffled_carry_items
        

    htk_boundaries = defaultdict(list)
    with open(file_name, 'r') as f:
        lines = f.readlines()
        index = 0
        for line in lines[2:]:
            if line.strip() == ".":
                # terminating char
                break
            boundaries = line.split()[0:2]
            letter = line.split()[2]
            ######## Added for visualize_hsv_histogram.py with htk general boundaries
            if letter == "sil":
                letter = "m" # carry empty
            letter = htk_labels_to_elan[letter]
            ########

            ######## hard coded for picklist 1, carry cause most signal
            # print(letter)
            if letter == "carry":
                # print(picklist_1_carry_items[index])
                letter = carry_items[index]
                index += 1
            ########
            
            letter_start = [int(boundary)/120000 for boundary in boundaries][0]
            letter_end = [int(boundary)/120000 for boundary in boundaries][1]
            htk_boundaries[letter].append(letter_start)
            htk_boundaries[letter].append(letter_end)
    return htk_boundaries, out_of_place

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
            print(elan_key)
            print(htk_key)
            print(type(i))
            print(htk_boundaries[htk_key])
            start_elan, end_elan = elan_boundaries[elan_key][i], elan_boundaries[elan_key][i + 1]
            start_htk, end_htk = htk_boundaries[htk_key][i], htk_boundaries[htk_key][i + 1]
                
            num_points += 1
            squared_error += (start_elan - start_htk) ** 2

    return squared_error, num_points
