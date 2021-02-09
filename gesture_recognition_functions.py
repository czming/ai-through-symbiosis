import math
import numpy as np
import cv2

def get_distance(point1, point2, image_shape):
    # get euclidean distance between point1 (x1, y1) and point2 (x2, y2)
    return math.sqrt(((point1[0]  - point2[0]) * image_shape[1]) ** 2 + ((point1[1] - point2[1]) * image_shape[0]) ** 2)


def improved_gesture_recognition(landmarks, handedness, image):
    """
    Gesture recognition method based on detecting whether each finger is open or closed.

    Determines if thumb is open or closed based on relative proximity between tip of thumb and base of palm and
    proximity between base of pinky and base of palm

    Determines if other fingers are open based on relatively proximity between each finger landmark and the base
    (a straight finger should have increasing keypoints further from the base of the palm)
    :param landmarks: array of 21 landmarks, each landmark denoted by [x,y] coordinates
    :param handedness: left or right hand
    :param image: image that is being detected from
    :return: array of 5 elements denoting if each finger is open, 0 = thumb, 1 = index and so on
    """
    fingers = [[np.array(landmarks[i]) for i in range(j * 4 + 1, j * 4 + 5)] for j in range(5)]
    fingers_open = [0] * 5
    #thumb detection, tip of thumb is closer to base of pinky than base of second finger
    # fingers_open[0] = get_distance(fingers[0][-1], fingers[1][0], image.shape) < \
    #                   get_distance(fingers[0][-1], fingers[4][0], image.shape)

    #thumb detection, base of palm is closer to base of pinky than tip of thumb to base of pinky
    fingers_open[0] = get_distance(landmarks[0], fingers[4][0], image.shape) < \
                      get_distance(fingers[0][-1], fingers[4][0], image.shape)

    for finger_index in range(1, 5):
        finger = fingers[finger_index]
        #finger is open if the tip is further from the base of the palm than the base of the finger to the base of the
        # palm
        finger_open = True
        for i in range(len(finger) - 1):
            #check all points in the finger, if the tip is closer to the base of the palm than any finger, it's closed
            if get_distance(finger[len(finger) - 1], landmarks[0], image.shape) < \
                    get_distance(finger[i], landmarks[0], image.shape):
                finger_open = False
        fingers_open[finger_index] = finger_open
    return fingers_open