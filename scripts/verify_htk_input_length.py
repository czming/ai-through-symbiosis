"""

Verifies that the length of the htk_input data matches the number of video frames
in the corresponding video

"""

import cv2
import numpy as np


VIDEO_FILE_FORMAT = "../Videos/picklist_%(index)d.MP4"
PICKLIST_FILE_FORMAT = "../htk_inputs/picklist_%(index)d_forward_filled_30_gaussian_filter_9_3.txt"


for index in range(41, 71):

    cap = cv2.VideoCapture(VIDEO_FILE_FORMAT % {"index": index})
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    picklist = np.genfromtxt(PICKLIST_FILE_FORMAT % {"index": index})

    print (frame_count, picklist.shape[0], frame_count == picklist.shape[0])
    
