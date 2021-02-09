import cv2
import mediapipe as mp
from gesture_recognition_functions import *
from google.protobuf.json_format import MessageToDict

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.6, max_num_hands=1)
#live video feed from camera 0
cap = cv2.VideoCapture(0)
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
	cv2.imshow('MediaPipe Hands', image)
	if len(first_hand_points) >= 21:
		print ("\r", improved_gesture_recognition(first_hand_points, handedness, image), handedness, end = "")
	if cv2.waitKey(5) & 0xFF == 27:
		break
hands.close()
cap.release()