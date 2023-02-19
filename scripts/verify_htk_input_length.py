"""

Verifies that the length of the htk_input data matches the number of video frames
in the corresponding video

"""

import cv2
import numpy as np
from utils import *

# use path from the current working directory (not the one from the utils module)
configs = load_yaml_config("configs/zm.yaml")

htk_input_folder = configs["file_paths"]["htk_input_file_path"]
video_folder = configs["file_paths"]["video_file_path"]


VIDEO_FILE_FORMAT = f"{video_folder}/picklist_%(index)d.MP4"
PICKLIST_FILE_FORMAT = f"{htk_input_folder}/picklist_%(index)d_forward_filled_30_gaussian_filter_9_3.txt"


for index in range(1, 91):

    cap = cv2.VideoCapture(VIDEO_FILE_FORMAT % {"index": index})
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    picklist = np.genfromtxt(PICKLIST_FILE_FORMAT % {"index": index})

    print (frame_count, picklist.shape[0], frame_count == picklist.shape[0])

    if frame_count == picklist.shape[0] is False:
        raise Exception(f"Lengths not the same for picklist {index}")
    
