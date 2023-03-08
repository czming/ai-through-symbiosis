"""

Verifies that the length of the htk_input data matches the number of video frames
in the corresponding video

"""

import cv2
import numpy as np
import sys

sys.path.append("..")

from utils import *

configs = load_yaml_config("configs/zm.yaml")

htk_input_folder = configs["file_paths"]["htk_input_file_path"]
video_folder = configs["file_paths"]["video_file_path"]


for index in range(53, 77):

    try:
        open(f"""{htk_input_folder}/GX010{str(index).zfill(3)}_forward_filled_30.txt""")

    except:
        continue

    cap = cv2.VideoCapture(f"{video_folder}/GX010{str(index).zfill(3)}.MP4")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    picklist = np.genfromtxt(f"{htk_input_folder}/GX010{str(index).zfill(3)}_forward_filled_30_gaussian_filter_9_3.txt")

    print (frame_count, picklist.shape[0], frame_count == picklist.shape[0])
    
