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

class ArUcoDetector:
	def __init__(self, intrinsic: np.ndarray, distortion: np.ndarray, aruco_dict, square_length=1):
		assert intrinsic.shape == (3,3)
		self.intrinsic     = intrinsic
		self.distortion    = distortion
		self.aruco_dict    = aruco_dict
		self.square_length = square_length
		
	def getCorners(self, image: np.ndarray):
		corners, ids, _ = cv2.aruco.detectMarkers(
			image, 
			self.aruco_dict, 
			# TODO: check if this is still required: cameraMatrix=self.intrinsic, 
			# TODO: check if this is still required: distCoeff=self.distortion
		)

		if corners:
			rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, self.square_length, self.intrinsic, self.distortion)
			ids = ids.flatten()

			assert len(rvecs) == len(corners)
			return [ArUcoMarker(corners[i][0], int(ids[i]), rvecs[i].reshape((3, 1)), tvecs[i].reshape((3,1))) for i in range(len(corners))]
		return []

class ArUcoMarker(object):
	def __init__(self, corners: np.ndarray, _id: int, rvec: np.ndarray, tvec: np.ndarray):
		assert rvec.shape == tvec.shape == (3, 1)
		self.corners = corners
		self._id = _id
		self.rvec = rvec
		self.tvec = tvec

	def get_id(self):
		return self._id

def hand_pos(landmarks, image):
	"""
	outputs the average (x, y) coordinate of the landmarks to indicate the hand position (which is denoted by center
	of palm
	:param landmarks: (x,y) of each landmark in an array as a ratio of image's shape
	:return: (x, y) coordinates of hand
	"""
	if len(landmarks) < 21:
		# need at least 21 for a hand
		return ()

	# center of palm appears to be a better indicator for the location of the hand (more mass concentrated at that point)
	fingers = [[np.array(landmarks[i]) for i in range(j * 4 + 1, j * 4 + 5)] for j in range(5)]
	# base of the palm
	total_x = landmarks[0][0] * image.shape[1]
	total_y = landmarks[0][1] * image.shape[0]
	for finger in fingers:
		# look at base of the finger
		curr = finger[0]
		total_x += (curr[0] * image.shape[1])
		total_y += (curr[1] * image.shape[0])
	return (total_x / 6, total_y / 6)

def get_hs_bins(cropped_hand_image):
	"""
	returns the hue and saturation bins for cropped_hand_image (bins are proportion of pixels within the image)
	:param cropped_hand_image: image to calculate the hue and saturation over
	:return: [hue_bins[:10], saturation_bins[10:20]], [] if cropped_hand_image is empty
	"""
	try: 
		number_of_pixels = cropped_hand_image.shape[0] * cropped_hand_image.shape[1]

		# hand might not be detected so number of pixels might be 0
		if number_of_pixels == 0:
			return []
		cropped_hand_image = cv2.cvtColor(cropped_hand_image, cv2.COLOR_BGR2RGB)
		hsv_image = skimage.color.rgb2hsv(cropped_hand_image)
		hues = []
		saturations = []
		
		hsv_image = np.array(hsv_image)

		hues = hsv_image[:, :, 0].flatten()
		saturations = hsv_image[:, :, 1].flatten()

		if np.array(hues).max() > 1 or np.array(hues).min() < 0 or np.array(saturations).max() > 1 or np.array(
				saturations).min() < 0:
			raise Exception("Hue or saturation not in range [0, 1]")

		# calculate the histograms
		hist_n, _ = np.histogram(hues, bins=180, range=(0, 1))
		sat_n, _ = np.histogram(saturations, bins=100, range=(0, 1))

		histsat = np.concatenate((hist_n, sat_n)) / number_of_pixels
		mystring = str(histsat).replace("[", "").replace("]", "").replace("\n", "")
		newstring = ' '.join(mystring.split())
		logging.debug(f"color vector: {str(newstring)}")
	except:
		logging.error(traceback.format_exc())

	return histsat

__detector = ArUcoDetector(
	np.array(os.environ.get("INTRINSIC", [[900., 0, 640], [0, 900, 360], [0, 0, 1]])), 
	np.array(os.environ.get("DISTORTION", [[0., 0., 0., 0., 0.]])), 
	cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_1000), 
	square_length=os.environ.get("SQUARE_LENGTH", 0.0254)
)
__hands = mp.solutions.hands.Hands(
	min_detection_confidence=os.environ.get("HANDS_MIN_DETECTION_COONFIDENCE", 0.7), 
	min_tracking_confidence=os.environ.get("HANDS_TRACKING_CONFIDENCE", 0.3), 
	max_num_hands=os.environ.get("HANDS_MAX_NUM_HANDS", 1)
)

def extract_features(image):
	"""

	:param image: array containing the image to be processed
	:param output_dict: dict to store the output
	:param counter: index to store the output at (index for the output dict)
	:return:
	"""
	__start = 0
	if os.environ.get("DEBUG_TIME", True):
		__start = time.time()
	htk_output_vector = [0] * 356

	image = cv2.resize(image, (os.environ.get("ORIGINAL_FRAME_WIDTH", 1920), os.environ.get("ORIGINAL_FRAME_HEIGHT", 1080)))
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	if os.environ.get("TO_FLIP", False):
		image = cv2.flip(image, 1)

	aruco_vectors = __detector.getCorners(image)

	logging.debug("\rnumber of markers: " + str(len(aruco_vectors)))

	# id to tvec mapping for aruco markers
	aruco_tvecs = {}

	# gathering the tvecs for plotting using matplotlib
	aruco_tvecs_plot = []

	logging.debug("Actual: ")

	# stores indices of aruco markers in aruco_vectors to be removed if the backprojection gives invalid values
	remove_aruco_vectors = []

	# plot aruco markers in image and scatter plot
	for index in range(len(aruco_vectors)):
		i = aruco_vectors[index]
		if i.get_id() in os.environ.get("VALID_SHELF_MARKERS", 
				set(os.environ.get("VALID_BIN_MARKERS", {110, 120, 130, 210, 220, 230, 310, 320, 330, 410, 420, 430, 510, 520, 530, 610, 620, 630})) 
			  | set(os.environ.get("VALID_LOCALIZATION_MARKERS", {111, 121, 131, 211, 221, 231, 311, 321, 331, 411, 421, 431, 511, 521, 531, 611, 621, 631}))
			):
			# make sure the aruco marker is valid shelf marker
			logging.debug(f"{i.get_id()}: {i.tvec[:, 0]}")

			aruco_2d = \
				cv2.projectPoints(np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]), i.tvec[:, 0], __detector.intrinsic,
								  __detector.distortion)[0][0][0]
			# objectPoints might be the actual 3D points, tvecs and rvecs are the estimated ones from the camera coord system
			# print (str(i.get_id()) + " " + str(aruco_2d))
			# if 0 <= int(aruco_2d[0]) and int(aruco_2d[0]) <= output_image.shape[1] and \
			#         0 <= int(aruco_2d[1]) and int(aruco_2d[1]) <= output_image.shape[0]:
			#     cv2.circle(output_image, (int(aruco_2d[0]), int(aruco_2d[1])), 3, (255, 255, 0), 3)

			logging.debug(f"{i.get_id()}: {aruco_2d}")

			# check that the marker when backprojected is within the bounds of the output_image
			if aruco_2d[0] < 0 or aruco_2d[0] > os.environ.get("ORIGINAL_FRAME_WIDTH", 1920) or aruco_2d[1] < 0 or aruco_2d[
				1] > os.environ.get("ORIGINAL_FRAME_WIDTH", 1920):
				# not a valid marker location, ignore
				remove_aruco_vectors.append(index)
				continue

			aruco_tvecs_plot.append(i.tvec[:, 0])
			aruco_tvecs[i.get_id()] = i.tvec[:, 0]

			__valid_shelf_markers_dict = {
				val: key for key, val in enumerate(
					set(os.environ.get("VALID_BIN_MARKERS", {110, 120, 130, 210, 220, 230, 310, 320, 330, 410, 420, 430, 510, 520, 530, 610, 620, 630})) \
			  		| set(os.environ.get("VALID_LOCALIZATION_MARKERS", {111, 121, 131, 211, 221, 231, 311, 321, 331, 411, 421, 431, 511, 521, 531, 611, 621, 631})))
			}

			# assign x, y
			htk_output_vector[__valid_shelf_markers_dict[i.get_id()] * 2 + os.environ.get('ARUCO_MARKER_HTK_OFFSET', 0)] = aruco_2d[
				0]
			htk_output_vector[__valid_shelf_markers_dict[i.get_id()] * 2 + os.environ.get('ARUCO_MARKER_HTK_OFFSET', 0) + 1] = aruco_2d[1]

	# ----------------------------------OPTICAL FLOW DETECTION--------------------------------------------------
	# mag, ang = optical_flow_algo.calc(output_image)
	#
	# mag, ang = mag.mean(), ang.mean()

	# optical flow detection not enabled
	mag, ang = 0, 0

	logging.debug(f"Optical flow: {mag} {ang}")

	htk_output_vector[os.environ.get('OPTICAL_FLOW_HTK_OFFSET', 352)] = mag
	htk_output_vector[os.environ.get('OPTICAL_FLOW_HTK_OFFSET', 352) + 1] = ang

	# output_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
	# To improve performance, optionally mark the output_image as not writeable to
	# pass by reference.
	image.flags.writeable = False
	results = __hands.process(image)

	# -----------------------------------------MEDIAPIPE HAND DETECTION-----------------------------------------
	# Draw the hand annotations on the output_image.
	# output_image.flags.writeable = True
	# use bgr for processing
	image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
	first_hand_points = []
	if results.multi_hand_landmarks:
		# get which hand we're looking at (look at the first hand that is detected)
		handedness = MessageToDict(results.multi_handedness[0])['classification'][0]['label']
		for hand_landmarks in results.multi_hand_landmarks:
			for i in hand_landmarks.landmark[:21]:
				first_hand_points.append([i.x, i.y, i.z])
				
	if len(first_hand_points) >= 21:
		# find current location of hand, (x, y)
		curr_hand_loc = hand_pos(first_hand_points, image)

		htk_output_vector[os.environ.get("s", 354)] = curr_hand_loc[0]
		htk_output_vector[os.environ.get("HAND_POS_HTK_OFFSET", 354) + 1] = curr_hand_loc[1]

		# ---------------------------------PROCESSING COLOR MODEL---------------------------------------------------
		min_x = float('inf')
		min_y = float('inf')
		max_x = 0
		max_y = 0
		for point in first_hand_points:
			x, y, _ = point
			# mediapipe outputs as a ratio
			x = int(x * os.environ.get("ORIGINAL_FRAME_WIDTH", 1920))
			y = int(y * os.environ.get("ORIGINAL_FRAME_HEIGHT", 1080))
			if x < min_x:
				min_x = x
			if y < min_y:
				min_y = y
			if x > max_x:
				max_x = x
			if y > max_y:
				max_y = y

		# use the original image to get the colors (original image is in RGB)
		cropped_hand_image = image[min_y:max_y, min_x:max_x, :]

		# cv2.imshow('MediaPipe Hands', cropped_hand_image)
		# cv2.waitKey(0)

		histsat = get_hs_bins(cropped_hand_image)

		# hand might not be detected so number of pixels might be 0
		if histsat is not []:
			for i in range(len(histsat)):
				# set the values in the htk_output_vector to the proportion of cropped points in each bin
				htk_output_vector[os.environ.get("COLOR_BIN_HTK_OFFSET", 72) + i] = histsat[i]

	# --------------------------------WRITING OUT HTK VECTOR TO FILE-----------------------------------------
	logging.info(htk_output_vector)
	# htk_output_vector = [str(i) for i in htk_output_vector]

	__end = 0
	if os.environ.get("DEBUG_TIME", True):
		__end = time.time()

	return htk_output_vector, __end - __start

service = Service(
		"feature-extractor",
		lambda id, shape, image: str((id, *extract_features(np.frombuffer(image.read(), dtype=np.uint8).reshape([int(i) for i in shape.split(",")])))),
		{
			'form': ['id', 'shape'],
			'files': ['image'],
		}
	).create_service(init_cors=True)
if __name__ == '__main__':
	service.run(
		host=os.environ.get("HOST", "0.0.0.0"),
		port=os.environ.get("PORT", 5000)
	)
