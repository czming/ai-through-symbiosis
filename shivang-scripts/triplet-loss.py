from scripts.utils.rms_error_utils import *
from utils import *

# use path from the current working directory (not the one from the utils module)
configs = load_yaml_config("scripts/configs/zm.yaml")

elan_label_folder = configs["file_paths"]["elan_annotated_file_path"]
htk_output_folder = configs["file_paths"]["htk_output_file_path"]

PICKLISTS =  list(range(1, 10)) + list(range(17, 23)) + [26]

total_rms = 0


for picklist_num in PICKLISTS:

    elan_annotated_file = f'{elan_label_folder}/picklist_'+str(picklist_num)+'.eaf'

    elan_boundaries = get_elan_boundaries(elan_annotated_file)

    htk_results_file = f"{htk_output_folder}/forward_filled_30_gaussian_filter_9_3_aruco_hs_bins/results-" + str(picklist_num)

    htk_boundaries = get_htk_boundaries(htk_results_file)

    squared_error, num_points = get_squared_error(elan_boundaries, htk_boundaries)

    rms_error = (squared_error / num_points) ** 0.5

    print (f"{rms_error}")

    total_rms += rms_error

average_rms = total_rms / len(PICKLISTS)

print (f"Average root mean squared error (across pick lists): {average_rms}")
    
