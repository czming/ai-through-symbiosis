import math
import numpy as np
import cv2


def get_distance(point1, point2, image_shape=None):
    """
    get pixel distance between two points
    :param point1: (x, y) of first point, where x and y are ratios of the point's location to respective image param
    :param point2: (x, y) of second point, where x and y are ratios of the point's location to respective image param
    :param image_shape: shape of the image
    :return: pixel distance between the two points
    """

    if image_shape == None:
        # no image shape passed in, assume to scale
        image_shape = (1, 1)
    # get euclidean distance between point1 (x1, y1) and point2 (x2, y2)
    return math.sqrt(((point1[0] - point2[0]) * image_shape[1]) ** 2 + ((point1[1] - point2[1]) * image_shape[0]) ** 2)


def improved_gesture_recognition(landmarks, handedness, image):
    """
    Gesture recognition method based on detecting whether each finger is open or closed.
    Determines if thumb is open or closed based on relative proximity between tip of thumb and base of palm and
    proximity between base of pinky and base of palm
    Determines if other fingers are open based on relatively proximity between each finger landmark and the base
    (a straight finger should have increasing keypoints further from the base of the palm)
    :param landmarks: (x,y) of each landmark in an array as a ratio of image's shape
    :param handedness: left or right hand
    :param image: image that is being detected from
    :return: array of 5 elements denoting if each finger is open, 0 = thumb, 1 = index and so on
    """
    fingers = [[np.array(landmarks[i]) for i in range(j * 4 + 1, j * 4 + 5)] for j in range(5)]
    fingers_open = [0] * 5
    # thumb detection, tip of thumb is closer to base of pinky than base of second finger
    # fingers_open[0] = get_distance(fingers[0][-1], fingers[1][0], image.shape) < \
    #                   get_distance(fingers[0][-1], fingers[4][0], image.shape)

    # thumb detection, base of palm is closer to base of pinky than tip of thumb to base of pinky
    fingers_open[0] = int(get_distance(landmarks[0], fingers[4][0], image.shape) < \
                      get_distance(fingers[0][-1], fingers[4][0], image.shape))

    for finger_index in range(1, 5):
        finger = fingers[finger_index]
        # finger is open if the tip is further from the base of the palm than the base of the finger to the base of the
        # palm
        finger_open = 1
        for i in range(len(finger) - 1):
            # check all points in the finger, if the tip is closer to the base of the palm than any finger, it's closed
            if get_distance(finger[len(finger) - 1], landmarks[0], image.shape) < \
                    get_distance(finger[i], landmarks[0], image.shape):
                finger_open = 0
        fingers_open[finger_index] = finger_open
    return fingers_open


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

def hand_bounding_box(landmarks, image):
    """
    gets the bounding box of the hand and the side of the bounding box
    :param landmarks: (x,y) of each landmark in an array as a ratio of image's shape
    :param image: image the landmarks are on (used for image.shape)
    :return: (bounding box of hand in image (top left at index 0, and clockwise from there),
     size of the bounding box (height, width))
    """

    left = image.shape[1]
    right = -1
    top = image.shape[0]
    bottom = -1
    for (x, y, z) in landmarks:
        #adjust for pixel values
        x = x * image.shape[1]
        y = y * image.shape[0]
        if x < left:
            # point further left than current left
            left = x
        if x > right:
            # point further right than current right
            right = x
        if y < top:
            #point that is higher than top
            top = y
        if y > bottom:
            bottom = y

    #add a little buffer for the rest of the finger (the landmarks aren't exactly at the tip)
    top, bottom, left, right = int(top - 20), int(bottom + 20), int(left - 20), int(right + 20)
    # return points (top left and clockwise from there, (x, y) for points), size (height, width)
    return ((left, top), (right, top), (right, bottom), (left, bottom)), (bottom - top, right - left)

def calculate_new_mean_variance(old_mean, old_variance, num_readings, new_reading):
    # num readings is exclusive of the current new_reading being inputted
    if num_readings == 0:
        # new mean is the new reading and the variance is just 0 for now, num_readings is 1 after this
        return new_reading, 0

    # old variance = (1 / (num_readings - 1)) * sum(x ** 2) - (num_readings / (num_readings - 1)) * (old_mean ** 2))
    # old_variance * (num_readings - 1) = sum(x ** 2) - num_readings * (old_mean ** 2)
    # old_mean ** 2
    old_squared_mean = old_mean ** 2
    # sum(x ** 2)
    sum_squared_x = old_variance * (num_readings - 1) + num_readings * old_squared_mean

    #find new mean by finding sum of all readings then dividing by num_readings + 1
    new_mean = (num_readings * old_mean + new_reading) / (num_readings + 1)

    #use new_mean and add new_reading squared to sum_squared_x to get new variance
    new_variance = (1 / num_readings) * (sum_squared_x + new_reading ** 2) - \
                   ((num_readings + 1) / num_readings) * (new_mean ** 2)

    return new_mean, new_variance
