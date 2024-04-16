from dotenv import load_dotenv
load_dotenv()

import os
import time
import logging
import traceback
import ast

# import cv2
# import numpy as np
# import skimage.color
from google.protobuf.json_format import MessageToDict
# import mediapipe as mp
from models import CarryHSVHistogramModel

from service import Service

model = CarryHSVHistogramModel()

htk_input_folder = "/shared/htk_inputs/"
htk_output_folder = "/shared/htk_outputs/"
pick_label_folder = "/shared/raw_labels/"

def main(id, picklist_nos):

	picklist_nos = ast.literal_eval(picklist_nos)
	
	return str((id, *model.fit(picklist_nos, htk_input_folder, htk_output_folder, pick_label_folder)))


service = Service(
		"hsv_train",
		lambda id, picklist_nos: main(id, picklist_nos),
		{
      		'form': ['id', 'picklist_nos'], # form is a dictionary, then passing in a list of keys
		}
	).create_service(init_cors=True)
if __name__ == '__main__':
	service.run(
		host=os.environ.get("HOST", "0.0.0.0"),
		port=os.environ.get("PORT", 5000)
	)
