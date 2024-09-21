from dotenv import load_dotenv
load_dotenv()

import os
import time
import logging
import traceback

# import cv2
# import numpy as np
# import skimage.color
from google.protobuf.json_format import MessageToDict
# import mediapipe as mp
from models import CarryHSVHistogramModel, IterativeClusteringModel

from service import Service

import json
import numpy as np
import ast

model = CarryHSVHistogramModel()

htk_input_folder = "/shared/htk_inputs/"
htk_output_folder = "/shared/htk_outputs/"
pick_label_folder = "/shared/raw_labels/"


def main(picklist_nos, hsv_avg_mean, hsv_avg_std, htk_input_folder, htk_output_folder):
	# hsv_avg_mean is expected to be a {object type: hsv_mean_vector as list}, then convert the value back to numpy arrays
	hsv_avg_mean = json.loads(hsv_avg_mean)
	hsv_avg_std = json.loads(hsv_avg_std)

	picklist_nos = ast.literal_eval(picklist_nos)

	# instantiate iterative clustering model and do fit iterative on HSVCarry model with that
	iterative_clustering_model = IterativeClusteringModel()

	iterative_clustering_model.class_hsv_bins_mean = {key: np.array(value) for key, value in hsv_avg_mean.items()}
	iterative_clustering_model.class_hsv_bins_std = {key: np.array(value) for key, value in hsv_avg_std.items()}

	model.iterative_clustering_model = iterative_clustering_model
	# returns {picklist: predicted_labels}
	return model.predict(picklist_nos, htk_input_folder, htk_output_folder)


service = Service(
		"hsv_test",
		lambda id, picklist_nos, hsv_avg_mean, hsv_avg_std: str((id, main(picklist_nos, hsv_avg_mean, hsv_avg_std, htk_input_folder, htk_output_folder))),
		{
      		'form': ['id', 'picklist_nos', 'hsv_avg_mean', 'hsv_avg_std'], # form is a dictionary, then passing in a list of keys
		}
	).create_service(init_cors=True)

if __name__ == '__main__':
	service.run(
		host=os.environ.get("HOST", "0.0.0.0"),
		port=os.environ.get("PORT", 5000)
	)
