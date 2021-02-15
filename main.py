import cv2
import mediapipe as mp
from gesture_recognition_functions import *
from google.protobuf.json_format import MessageToDict
import time

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.6, max_num_hands=1)
#live video feed from camera 0
cap = cv2.VideoCapture(0)

MOVEMENT_MOE = 5
REFRESH_TIME = 0.05

prev_hand_loc = ()
curr_hand_loc = ()
prev_time = time.time()


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
			#look only at the first hand detected if there are multiple hands
			for i in hand_landmarks.landmark[:21]:
				first_hand_points.append([i.x, i.y])
				#image = cv2.circle(image, (int(i.x * image.shape[1]), int(i.y * image.shape[0])), 2, (0, 0, 255), 3)
				mp_drawing.draw_landmarks(
				   image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
	if len(first_hand_points) >= 21:
		curr_time = time.time()
		time_change = curr_time - prev_time

		# find current location of hand
		curr_hand_loc = hand_pos(first_hand_points, image)
		cv2.circle(image, (int(curr_hand_loc[0]), int(curr_hand_loc[1])), 3, (255, 0, 255), 3)

		# find bounding box of hand
		bounding_box, bounding_box_size = hand_bounding_box(first_hand_points, image)
		for point_index in range(len(bounding_box)):
			#use the %len(bounding_box) to wrap around back to the start
			cv2.line(image, bounding_box[point_index], bounding_box[(point_index + 1) % len(bounding_box)],
							 (0, 255, 0), 4)

		# motion related stuff that is checked eveyr fixed period (otherwise different frame refresh rates lead to
		# inconsistencies in measuring movement in pixels)
		if (time_change > REFRESH_TIME):
			# checks when the time_change is greater than the REFRESH TIME

			print ("\r", improved_gesture_recognition(first_hand_points, handedness, image), handedness, end = "")
			if prev_hand_loc != () and (curr_hand_loc[0] > prev_hand_loc[0] + MOVEMENT_MOE * time_change / REFRESH_TIME):
				# curr_hand is more than MOVEMENT_MOE pixels to the right of prev_hand_loc
				print("Moving right", end="")

			elif prev_hand_loc != () and (curr_hand_loc[0] < prev_hand_loc[0] - MOVEMENT_MOE * time_change / REFRESH_TIME):
				print("Moving left", end=" ")

			prev_hand_loc = curr_hand_loc
			prev_time = curr_time

	cv2.imshow('MediaPipe Hands', image)
	if cv2.waitKey(5) & 0xFF == 27:
		break
hands.close()
cap.release()