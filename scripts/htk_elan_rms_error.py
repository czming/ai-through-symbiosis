from rms_error_utils import *

PICKLISTS =  list(range(1, 10)) + list(range(17, 23)) + [26]

total_rms = 0


for picklist_num in PICKLISTS:

    elan_annotated_file = '../elan_annotated/picklist_'+str(picklist_num)+'.eaf'

    elan_boundaries = get_elan_boundaries(elan_annotated_file)

    htk_results_file = "../results/forward_filled_30_gaussian_filter_9_3_aruco_hs_bins/results-" + str(picklist_num)

    htk_boundaries = get_htk_boundaries(htk_results_file)

    squared_error, num_points = get_squared_error(elan_boundaries, htk_boundaries)

    rms_error = (squared_error / num_points) ** 0.5

    print (f"{rms_error}")

    total_rms += rms_error

average_rms = total_rms / len(PICKLISTS)

print (f"Average root mean squared error (across pick lists): {average_rms}")
    
