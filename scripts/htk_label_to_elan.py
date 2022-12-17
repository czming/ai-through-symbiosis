import glob as glob
import os
import xml.etree.ElementTree as ET
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from rms_error_utils import get_elan_boundaries, get_htk_boundaries

PICKLIST_NUM = '2'
# Passing the path of the
# xml document to enable the
# parsing process
elan_file_name = '../elan_annotated/picklist_' + PICKLIST_NUM + '.eaf' 

elan_boundaries = get_elan_boundaries(elan_file_name)


# labels = glob.glob("./picklist-specific-results/*")
htk_file_name = "../results/forward_filled_30_gaussian_filter_9_3_aruco_only/results-" + PICKLIST_NUM

htk_boundaries = get_htk_boundaries(htk_file_name)

print("ELAN")
print(elan_boundaries)
print("HTK")
print(htk_boundaries)

legends = {}
colors = {'sil':'black',
          'a':'c',
          'e':'y',
          'i':'m',
          'm':'r'}
# colors2 = {'pick':'c',
#            'carry':'y',
#            'place':'m',
#            'empty':'r'}
key_swap = {'pick':'a',
            'carry':'e',
            'place':'i',
            'carry_empty':'m'} # elan to htk

for key, value in elan_boundaries.items():
    # elan boundaries are at the bottom
    y_key = [2] * len(value)
    new_key = key_swap[key]
    legends[new_key] = plt.scatter(value, y_key, color=colors[new_key])

for key, value in htk_boundaries.items():
    y_key = [1] * len(value)
    legends[key] = plt.scatter(value, y_key, color=colors[key])

plt.legend(legends.values(),
           legends.keys(),
           scatterpoints=1,
           loc='best',
           ncol=1,
           fontsize=8)
plt.title("HTK (1) vs. ELAN (2)")
plt.xlabel("Time (seconds)")
plt.show()

# y_htk = [1] * len(htk_boundaries)
# y_elan = [2] * len(elan_boundaries)
# print(elan_boundaries)
# print(htk_boundaries)
# print(elan_boundaries)
# plt.scatter(htk_boundaries, y_htk, marker='x')
# plt.scatter(elan_boundaries, y_elan, marker='o')
#
# plt.show()
