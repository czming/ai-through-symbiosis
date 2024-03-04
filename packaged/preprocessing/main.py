# did you push yet?

from dotenv import load_dotenv
load_dotenv()

import os
import time
import logging
import traceback

import cv2
import numpy as np
import skimage.color
from google.protobuf.json_format import MessageToDict
import mediapipe as mp

from service import Service
from utils.forward_fill import perform_forward_fill
from utils.gaussian_convolution import perform_gaussian_convolution

def preprocess_features(data):  
    __start = 0
    if os.environ.get("DEBUG_TIME", True):
        __start = time.time()
  
    data = perform_forward_fill(data)
    data = perform_gaussian_convolution(data)
    
    __end = 0
    if os.environ.get("DEBUG_TIME", True):
        __end = time.time()
  
    return data.tostring(), data.dtype.str, data.shape, __end - __start

service = Service(
		"preprocessing",
		lambda id, shape, data: str((id, *preprocess_features(np.fromstring(data.read()).reshape([int(i) for i in shape.split(",")])))),
		{
      		'form': ['id', 'shape'],
			'files': ['data'],
		}
	).create_service(init_cors=True)
if __name__ == '__main__':
	service.run(
		host=os.environ.get("HOST", "0.0.0.0"),
		port=os.environ.get("PORT", 5000)
	)
