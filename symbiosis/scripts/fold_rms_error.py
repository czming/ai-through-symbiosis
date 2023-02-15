from rms_error_utils import *
import glob
import os

PICKLISTS = list(range(1, 90))

ROOT_FOLD_DIR = "./experiments/aruco1dim-101avgfilter-picklists41-90/results/"

folds_dir = glob.glob(ROOT_FOLD_DIR + "/fold*/")

total_rms = 0

num_picklists = 0

print("Number of Folds: " + str(len(folds_dir)))
for fold_dir in folds_dir:
    print(fold_dir)
    for picklist_num in PICKLISTS:
        print(picklist_num)
        # iterate through each of the picklists and see if results file is in there
        if os.path.exists(fold_dir + f"results-{picklist_num}"):
            # such a file exists, look at the rms error            
            elan_annotated_file = '../elan_annotated/picklist_'+str(picklist_num)+'.eaf'

            elan_boundaries = get_elan_boundaries(elan_annotated_file)
            print(elan_boundaries)

            htk_results_file = fold_dir + f"results-{picklist_num}"

            num_picklists += 1

            htk_boundaries = get_htk_boundaries(htk_results_file)
            print(htk_boundaries)
            try:
                squared_error, num_points = get_squared_error(elan_boundaries, htk_boundaries)

            except IndexError as e:
                # skip for now
                print (f"Index Error: {e}, file {htk_results_file}")
                continue
            rms_error = (squared_error / num_points) ** 0.5

            print (f"{htk_results_file}: {rms_error}")

            total_rms += rms_error

print (f"Average rms error per picklist: {total_rms / num_picklists}")
