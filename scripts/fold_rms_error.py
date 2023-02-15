import glob
import os
from utils import *

configs = load_yaml_config("configs/jon.yaml")

PICKLISTS = list(range(1, 90))

htk_output_folder = configs["file_paths"]["htk_output_file_path"]
elan_annotated_folder = configs["file_paths"]["elan_annotated_file_path"]

root_fold_dir = f"{htk_output_folder}/results/"

folds_dir = glob.glob(root_fold_dir + "/*/")

total_rms = 0

num_picklists = 0

for fold_dir in folds_dir:
    for picklist_num in PICKLISTS:
        # iterate through each of the picklists and see if results file is in there
        if os.path.exists(fold_dir + f"/results-{picklist_num}"):
            # such a file exists, look at the rms error            
            elan_annotated_file = f'{elan_annotated_folder}/picklist_'+str(picklist_num)+'.eaf'

            elan_boundaries = get_elan_boundaries(elan_annotated_file)

            htk_results_file = fold_dir + f"/results-{picklist_num}"

            num_picklists += 1

            htk_boundaries = get_htk_boundaries(htk_results_file)

            try:

                squared_error, num_points = get_squared_error(elan_boundaries, htk_boundaries)

            except IndexError as e:
                # skip for now
                print (f"Index Error: {e}, file {htk_results_file}")
                print()
                continue
            rms_error = (squared_error / num_points) ** 0.5

            print (f"{os.path.basename(htk_results_file)}: {rms_error}")

            total_rms += rms_error

print (f"Average rms error per picklist: {total_rms / num_picklists}")

