from scripts.utils.rms_error_utils import *
import glob
import os

PICKLISTS = list(range(1, 41))

ROOT_FOLD_DIR = "../results/pca16dims"

folds_dir = glob.glob(ROOT_FOLD_DIR + "/*/")

total_rms = 0

num_picklists = 0

for fold_dir in folds_dir:
    for picklist_num in PICKLISTS:
        # iterate through each of the picklists and see if results file is in there
        if os.path.exists(fold_dir + f"/results-{picklist_num}"):
            # such a file exists, look at the rms error            
            elan_annotated_file = '../elan_annotated/picklist_'+str(picklist_num)+'.eaf'

            elan_boundaries = get_elan_boundaries(elan_annotated_file)

            htk_results_file = fold_dir + f"/results-{picklist_num}"

            num_picklists += 1

            htk_boundaries = get_htk_boundaries(htk_results_file)

            try:

                squared_error, num_points = get_squared_error(elan_boundaries, htk_boundaries)

            except IndexError as e:
                # skip for now
                print (f"Index Error: {e}, file {htk_results_file}")
                continue
            rms_error = (squared_error / num_points) ** 0.5

            print (f"{htk_results_file}: {rms_error}")

            total_rms += rms_error

print (f"Average rms error for fold: {total_rms / num_picklists}")
