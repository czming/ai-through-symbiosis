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
    """
    updating the mean and variance of a tracked value

    :param old_mean: previous mean
    :param old_variance: previous variance
    :param num_readings: number of previous readings (i.e. exclusive of current reading)
    :param new_reading: the value of the current reading
    :return: (new mean, new variance)
    """
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


def get_best_fit_plane(points):
    """
    Find the best fit plane for the points, in the form z = a + bx + cy, using OLS linear regression
    :param points: Array of 3-D points, [x, y, z]
    :return: numpy array of [a, b, c], where z = a + bx + cy
    """
    points = np.array(points)
    num_points = len(points)
    #A = [1 x y]
    A = np.concatenate((np.ones((num_points, 1)), points[:, :2]), axis=1)
    #x_pred = (A^T * A) ^ (-1) * A^T * b, OLS
    best_fit_param = np.linalg.inv(A.T @ A) @ A.T @ points[:, 2]

    return best_fit_param

def test_best_fit_plane():
    import matplotlib.pyplot as plt

    x = np.linspace(0, 10, 10)
    y = np.linspace(0, 10, 10)
    X, Y = np.meshgrid(x, y)

    x_val = [1, 4, 2, 3, 7, 5, 6, 7, 8, 10]
    y_val = [3, 2, 6, 1, 8, 9, 10, 3, 2, 5]
    z_val = [6, 4, 6, 2, 8, 10, 3, 4, 5, 2]
    points = np.concatenate((np.expand_dims(np.array(x_val), axis=-1), np.expand_dims(np.array(y_val), axis=-1),
                             np.expand_dims(np.array(z_val), axis=-1)), axis=1)

    best_fit_plane = get_best_fit_plane(points)

    Z = best_fit_plane[0] + best_fit_plane[1] * X + best_fit_plane[2] * Y

    plt.figure()
    ax = plt.axes(projection = "3d")
    ax.plot_surface(X, Y, Z, alpha = 0.5)
    ax.scatter3D(x_val, y_val, z_val)
    plt.show()


def get_best_fit_plane(points):
    """
    Find the best fit plane for the points, in the form z = a + bx + cy, using OLS linear regression
    :param points: Array of 3-D points, [x, y, z]
    :return: numpy array of [a, b, c], where z = a + bx + cy
    """
    points = np.array(points)
    num_points = len(points)
    #A = [1 x y]
    A = np.concatenate((np.ones((num_points, 1)), points[:, :2]), axis=1)
    #x_pred = (A^T * A) ^ (-1) * A^T * b, OLS
    best_fit_param = np.linalg.inv(A.T @ A) @ A.T @ points[:, 2]

    return best_fit_param

def best_fit_point(plane, point_coords):
    """
    finds coordinates of foot of perpendicular from point to plane
    :param plane: [n1, n2, n3, d], a \dot n = d, n = [n1; n2; n3], n1 \dot x + n2 \dot y + n3 \dot z = d
    :param point_coords: [x, y, z]
    :return: [x, y, z] coords of the foot of perpendicular from point_coords to plane
    """
    #the actual point and foot of perpendicular lie on a line with unit vector n and passes through the point
    #let foot of perpendicular be a, a \dot n = d, a = [x + \beta n1; y + \beta n2; z + \beta n3]
    #(x + \beta n1) \dot n1 + ... for y and z = d, trying to find \beta
    #LHS: \beta n1 \dot n1 + \beta n2 \dot n2 + \beta n3 \dot n3
    LHS = plane[0] ** 2 + plane[1] ** 2 + plane[2] ** 2 #LHS in terms of beta
    #RHS: d - x \dot n1 - y \dot n2 - z \dot n3
    RHS = plane[3] - point_coords[0] * plane[0] - point_coords[1] * plane[1] - point_coords[2] * plane[2]

    #get beta
    B = RHS / LHS

    #now just use line equation to return the best fit point
    return [point_coords[0] + B * plane[0], point_coords[1] + B * plane[1], point_coords[2] + B * plane[2]]

def get_unit_vector(input_vector):
    """
    returns the unit vector in direction of input_vector
    :param input_vector: [x, y, z] vector, or np.array([x, y, z])
    :return: vector of length 1 in direction of input_vector
    """
    input_vector = np.array(input_vector)

    input_vector_length = (input_vector[0] ** 2 + input_vector[1] ** 2 + input_vector[2] ** 2) ** 0.5

    #divide all the vector lengths in different dimensions by the length of the vector
    return input_vector / input_vector_length

def in_basis_vector(vector, *basis_vectors):
    """
    returns the linear combination of weights of basis vectors that sum to the vector (the vector in
    coordinate system using basis_vectors as the basis vectors

    (basis vectors and vector must be in the same coordinate system when passed into the function)

    :param vector: vector in the basis vectors in the form [x, y, z, ...]
    :param basis_vectors: array of basis vectors, must have same dimensionality as vector
    :return: linear combination of weights of basis_vectors that sum to vector
    """

    vector = np.array(vector)
    #take transpose so each column is a basis vector and each row is looking at the combination of weights
    #in the original basis vectors (which the vectors are currently denoted in)
    basis_vectors = np.array(basis_vectors).T

    if len(basis_vectors.shape) == 1:
        #only one vector, need to expand the dimensions to wrap in another layer of array
        basis_vectors = np.expand_dims(basis_vectors, axis=0)

    #make sure they have the same dimensionality
    assert basis_vectors.shape[0] == vector.shape[0], "Vectors do not have the same dimensionality"

    #shouldn't have singular matrix error because the two basis vectors should be perpendicular
    return np.linalg.lstsq(basis_vectors, vector, rcond=None)[0]

# def test_best_fit_plane():
#     import matplotlib.pyplot as plt
#
#     x = np.linspace(0, 10, 10)
#     y = np.linspace(0, 10, 10)
#     X, Y = np.meshgrid(x, y)
#
#     x_val = [1, 4, 2, 3, 7, 5, 6, 7, 8, 10]
#     y_val = [3, 2, 6, 1, 8, 9, 10, 3, 2, 5]
#     z_val = [6, 4, 6, 2, 8, 10, 3, 4, 5, 2]
#     points = np.concatenate((np.expand_dims(np.array(x_val), axis=-1), np.expand_dims(np.array(y_val), axis=-1),
#                              np.expand_dims(np.array(z_val), axis=-1)), axis=1)
#
#     best_fit_plane = get_best_fit_plane(points)
#
#     Z = best_fit_plane[0] + best_fit_plane[1] * X + best_fit_plane[2] * Y
#
#     plt.figure()
#     ax = plt.axes(projection = "3d")
#     ax.plot_surface(X, Y, Z, alpha = 0.5)
#     ax.scatter3D(x_val, y_val, z_val)
#     plt.show()