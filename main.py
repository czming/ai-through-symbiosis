import cv2
import mediapipe as mp
from gesture_recognition_functions import *
from google.protobuf.json_format import MessageToDict
import time
import random

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(min_detection_confidence=0.9, min_tracking_confidence=0.5, max_num_hands=1)
#live video feed from camera 0
cap = cv2.VideoCapture(0)

#error for detecting direction the hand is moving
MOVEMENT_MOE = 10

prev_hand_loc = ()
curr_hand_loc = ()
#used to keep track of time between frames for movement detection
prev_time = time.time()

objects = []

curr_moving = -1

added = 0

while cap.isOpened():
	success, image = cap.read()
	if not success:
		print("Ignoring empty camera frame.")
		# If loading a video, use 'break' instead of 'continue'.
		continue

	# Flip the image horizontally for a later selfie-view display, and convert
	# the BGR image to RGB.
	image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
	# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	# To improve performance, optionally mark the image as not writeable to
	# pass by reference.
	image.flags.writeable = False
	results = hands.process(image)

	# Draw the hand annotations on the image.
	image.flags.writeable = True
	image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
	first_hand_points = []
	if results.multi_hand_landmarks:
		#get which hand we're looking at (look at the first hand that is detected)
		handedness = MessageToDict(results.multi_handedness[0])['classification'][0]['label']
		for hand_landmarks in results.multi_hand_landmarks:
			for i in hand_landmarks.landmark[:21]:
				first_hand_points.append([i.x, i.y])
				#image = cv2.circle(image, (int(i.x * image.shape[1]), int(i.y * image.shape[0])), 2, (0, 0, 255), 3)
				mp_drawing.draw_landmarks(
				   image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

	if len(first_hand_points) >= 21:

		#detect motion of hand
		curr_hand_loc = hand_pos(first_hand_points, image)[0]
		curr_time = time.time()
		time_change = curr_time - prev_time

		if time_change > 0.05:
			print("\r", improved_gesture_recognition(first_hand_points, handedness, image), handedness, end=" ")

			if prev_hand_loc != () and (curr_hand_loc[0] > prev_hand_loc[0] + MOVEMENT_MOE * time_change / 0.05):
				#curr_hand is more than 5 pixels to the right of prev_hand_loc
				print ("Moving right", end = "")
				if all(improved_gesture_recognition(first_hand_points, handedness, image)):
					#deletes objects if hand is open and quickly move to the right
					for index in range(len(objects)):
						if get_distance(curr_hand_loc, objects[index]) < 75:
							del(objects[index])
							if curr_moving == index:
								curr_moving = -1
							break
			elif prev_hand_loc != () and (curr_hand_loc[0] < prev_hand_loc[0] - MOVEMENT_MOE * time_change / 0.05):
				print ("Moving left", end = " ")

			if curr_moving != -1:
				index = curr_moving
				i = objects[curr_moving]
				image = hand_pos(first_hand_points, image)[1]
				curr_moving = -1
				if curr_hand_loc != () and get_distance(curr_hand_loc, i[:2]) < 50 * time_change / 0.02 and not any(
						improved_gesture_recognition(first_hand_points, handedness, image)):
					objects[index] = [int(curr_hand_loc[0]), int(curr_hand_loc[1]), i[2], i[3]]
					curr_moving = index
			else:
				for index in range(len(objects)):
					i = objects[index]
					image = hand_pos(first_hand_points, image)[1]
					if curr_hand_loc != () and get_distance(curr_hand_loc, i[:2]) < 50 and not any(
							improved_gesture_recognition(first_hand_points, handedness, image)):
						objects[index] = [int(curr_hand_loc[0]), int(curr_hand_loc[1]), i[2], i[3]]
						curr_moving = index
						break
			if curr_moving == -1 and not any(improved_gesture_recognition(first_hand_points, handedness, image)):
				objects.append([int(curr_hand_loc[0]), int(curr_hand_loc[1]), 50, (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))])

			prev_hand_loc = curr_hand_loc
			prev_time = curr_time
	else:
		# reset prev palm since there's a frame where the hand was missing
		prev_hand_loc = ()
	for i in objects:
		image = cv2.circle(image, tuple(i[:2]), i[2], i[3], -1)

	cv2.imshow('MediaPipe Hands', image)
	if cv2.waitKey(5) & 0xFF == 27:
		break
hands.close()
cap.release()