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
from models import CarryHSVHistogramModel,IterativeClusteringModel

from service import Service

import json
import numpy as np

model = CarryHSVHistogramModel()

htk_input_folder = "/shared/htk_inputs/"
htk_output_folder = "/shared/htk_outputs/"
pick_label_folder = "/shared/raw_labels/"

def main(id, picklist_no, hsv_avg_mean, hsv_avg_std):
	# load from json 
	hsv_avg_mean = json.loads(hsv_avg_mean)
	hsv_avg_std = json.loads(hsv_avg_std)

	# instantiate iterative clustering model and do fit iterative on HSVCarry model with that
	iterative_clustering_model = IterativeClusteringModel()

	iterative_clustering_model.class_hsv_bins_mean = {key: np.array(value) for key, value in hsv_avg_mean.items()}
	iterative_clustering_model.class_hsv_bins_std = {key: np.array(value) for key, value in hsv_avg_std.items()}

	model.iterative_clustering_model = iterative_clustering_model

	hsv_avg_mean, hsv_avg_std = model.fit_iterative(int(picklist_no), htk_input_folder, htk_output_folder, pick_label_folder, beta=0.9)[:2]

	# convert from np array to lists
	hsv_avg_mean = {key:list(value) for key, value in hsv_avg_mean.items()}
	hsv_avg_std = {key:list(value) for key, value in hsv_avg_std.items()}

	output = str((id, hsv_avg_mean, hsv_avg_std))
	
	print (output)

	# only want hsv avg mean and std, don't want the labels from the fit iterative output
	return output

service = Service(
		"hsv_train_iterative",
		lambda id, picklist_no, hsv_avg_mean, hsv_avg_std: main(id, picklist_no, hsv_avg_mean, hsv_avg_std),
		{	# take in a specific picklist_no
      		'form': ['id', 'picklist_no', 'hsv_avg_mean', 'hsv_avg_std'], # form is a dictionary, then passing in a list of keys
		}
	).create_service(init_cors=True)
	
if __name__ == '__main__':
	service.run(
		host=os.environ.get("HOST", "0.0.0.0"),
		port=os.environ.get("PORT", 5000)
	)
