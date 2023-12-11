"""

functions to extract features from video frames in parallel

"""

import argparse
import copy

import logging

import cv2
import mediapipe as mp
from gesture_recognition_functions import *
from optical_flow import *
from google.protobuf.json_format import MessageToDict
import time
import matplotlib.pyplot as plt

import skimage.color

from multiprocessing import Process, Manager
import time
from mlsocket import MLSocket

HOST = "127.0.0.1"
PORT = 48293

# from scripts.utils import load_yaml_config


# requirements: opencv-python, opencv-contrib-python

class ArUcoDetector:
    def __init__(self, intrinsic: np.ndarray, distortion: np.ndarray, aruco_dict, square_length=1):
        assert intrinsic.shape == (3, 3)
        self.intrinsic = intrinsic
        self.distortion = distortion
        self.aruco_dict = aruco_dict
        self.square_length = square_length

    def getCorners(self, image: np.ndarray):
        # corners seem to be (x, y)
        corners, ids, rejected = cv2.aruco.detectMarkers(image, self.aruco_dict, cameraMatrix=self.intrinsic,
                                                         distCoeff=self.distortion)

        if corners:
            # try and find all 4 points
            rvecs, tvecs, markerPoints = cv2.aruco.estimatePoseSingleMarkers(corners, self.square_length,
                                                                             self.intrinsic, self.distortion)

            ids = ids.flatten()

            # ensure that the 3D points and the 2D points are aligned
            assert len(rvecs) == len(corners)

            # only include those that are not in the horizontal margin (use top left point of marker as the reference)
            return [ArUcoMarker(corners[i][0], int(ids[i]), rvecs[i].reshape((3, 1)), tvecs[i].reshape((3, 1))) for i in
                    range(len(corners))]
        return []


class ArUcoMarker(object):
    def __init__(self, corners: np.ndarray, _id: int, rvec: np.ndarray, tvec: np.ndarray):
        """
        params:
            - corners: pixel positions it was found in
            - _id: ArUco marker number
            - rvec: Rodriguez rotational 3d unit vector
            - tvec: Rodriguez
        """
        assert rvec.shape == tvec.shape == (3, 1)

        self.corners = corners
        self._id = _id
        self.rvec = rvec
        self.tvec = tvec

    def get_id(self):
        return self._id


def get_hand_corners(hand_points: list) -> np.ndarray:
    """
	Get pseudo ArUco landmarks for hand positions

	params:
		- hand_points: list of MediaPipe hand positions

	return:
		- ArUco corners
	"""
    top_left = np.asarray(hand_points[17][:2])  # pinky
    top_right = np.asarray(hand_points[5][:2])  # index
    bottom_right = np.asarray(hand_points[1][:2])  # index
    bottom_left = top_left + (bottom_right - top_right)

    return np.vstack([top_left, top_right, bottom_right, bottom_left])


def parse_picklist(file):
    """
    Parses through pick list as a CSV file and returns an array of dictionaries representing each action in the pick
    list
    :param file: CSV file which contains pick list
    :return: array of dictionaries which contain {"item_index", "horizontal_location", "vertical_location", "picked"}.
    horizontal_location = -1 if left, 0 if center and 1 if right, vertical_location = -1 if bottom, 0 if middle, and 1
    if top, picked is 1 if the current action is picking up the object and 0 is the current action is depositing the
    object
    """

    infile = open(file).readlines()
    picklist = []
    pick_indices = ["item_index", "horizontal_location", "vertical_location"]
    # initially object not in hand, so we need to pick
    picking = True
    for line in infile:
        line = [int(i) for i in line.strip().split(",")]
        curr_pick = dict(zip(pick_indices, line))
        curr_pick["picked"] = picking
        picking = not picking
        picklist.append(curr_pick)
    return picklist


def get_hs_bins(cropped_hand_image):
    """
    returns the hue and saturation bins for cropped_hand_image (bins are proportion of pixels within the image)
    :param cropped_hand_image: image to calculate the hue and saturation over
    :return: [hue_bins[:10], saturation_bins[10:20]], [] if cropped_hand_image is empty
    """
    number_of_pixels = cropped_hand_image.shape[0] * cropped_hand_image.shape[1]

    # hand might not be detected so number of pixels might be 0
    if number_of_pixels == 0:
        return []

    # cv2.imshow("", cropped_hand_image)
    # cv2.waitKey(0)
    cropped_hand_image = cv2.cvtColor(cropped_hand_image, cv2.COLOR_BGR2RGB)
    hsv_image = skimage.color.rgb2hsv(cropped_hand_image)
    dims = hsv_image.shape
    hues = []
    saturations = []
    for i in range(0, dims[0]):
        for j in range(0, dims[1]):
            # subsample
            if i % 1 == 0:
                # BGR
                hsv_value = np.array([[hsv_image[i, j, 0],
                                       hsv_image[i, j, 1],
                                       hsv_image[i, j, 2]]])
                # rgb_value = np.array([[color_image[i, j, 0],
                #                        color_image[i, j, 1],
                #                        color_image[i, j, 2]]]) / 255.0

                hues.append(hsv_value[0][0])
                saturations.append(hsv_value[0][1])

    if np.array(hues).max() > 1 or np.array(hues).min() < 0 or np.array(saturations).max() > 1 or np.array(
            saturations).min() < 0:
        raise Exception("Hue or saturation not in range [0, 1]")

    # visualizing color model

    # f, axarr = plt.subplots(2, 2)
    #
    # # axarr[0, 0].imshow(cropped_hand_image)
    # h = sum(hues) / len(hues)
    # s = sum(saturations) / len(saturations)
    # # print(max(set(hues), key=hues.count)) #mode
    # # print(max(set(saturations), key=saturations.count)) #mode
    # V = np.array([[h, s]])
    # origin = np.array([[0, 0, 0], [0, 0, 0]])  # origin point
    # # axarr[1].set_xlim([0, 10])
    # # axarr[1].set_ylim([0, 10])
    # axarr[0, 1].quiver(*origin, V[:, 0], V[:, 1], color=['r'], scale=10)
    # circle1 = plt.Circle((0, 0), 1 / 21, fill=False)
    # axarr[0, 1].add_patch(circle1)
    # axarr[1, 0].set_xlim([0, 1])
    # # axarr[1,0].set_title("Hue")

    # print(hist_n, hist_bins)
    # print(sat_n, sat_bins)

    # calculate the histograms
    hist_n, hist_bins = np.histogram(hues, bins=180, range=(0, 1))
    sat_n, sat_bins = np.histogram(saturations, bins=100, range=(0, 1))

    histsat = np.concatenate((hist_n, sat_n)) / number_of_pixels
    mystring = str(histsat).replace("[", "").replace("]", "").replace("\n", "")
    newstring = ' '.join(mystring.split())
    logging.debug(f"color vector: {str(newstring)}")

    return histsat

def extract_features(image, output_dict, counter):
    """

    :param image: array containing the image to be processed
    :param output_dict: dict to store the output
    :param counter: index to store the output at (index for the output dict)
    :return:
    """

    htk_output_vector = [0 for i in range(356)]

    image = cv2.resize(image, (ORIGINAL_FRAME_WIDTH, ORIGINAL_FRAME_HEIGHT))

    if TO_FLIP:
        # Flip the image horizontally for a later selfie-view display, and convert
        # the BGR image to RGB.
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # image = cv2.resize(image, (1920, 1080))

    # undistort image
    # image = cv2.undistort(image, initial_intrinsic, initial_distortion, None)

    output_image = np.copy(image)

    # the actual aruco positions directly from the aruco_detection
    aruco_vectors = aruco_detector.getCorners(image)

    # scatter plot figure objects
    aruco_sp_points = None
    aruco_sp_predicted_points = None
    aruco_plot_plane = None
    hand_sp_points = None

    logging.debug("\rnumber of markers: " + str(len(aruco_vectors)))

        # draw axis for the aruco markers
        # cv2.aruco.drawAxis(image, newcameramtx, distortion, marker.rvec, marker.tvec, 0.05)

    # id to tvec mapping for aruco markers
    aruco_tvecs = {}

    # gathering the tvecs for plotting using matplotlib
    aruco_tvecs_plot = []

    # aruco marker for hand
    aruco_hand_tvec = None

    logging.debug("Actual: ")

    # stores indices of aruco markers in aruco_vectors to be removed if the backprojection gives invalid values
    remove_aruco_vectors = []

    # plot aruco markers in image and scatter plot
    for index in range(len(aruco_vectors)):
        i = aruco_vectors[index]
        if i.get_id() in VALID_SHELF_MARKERS:
            # make sure the aruco marker is valid shelf marker
            logging.debug(f"{i.get_id()}: {i.tvec[:, 0]}")

            aruco_2d = \
                cv2.projectPoints(np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]), i.tvec[:, 0], intrinsic,
                                  distortion)[0][0][0]
            # objectPoints might be the actual 3D points, tvecs and rvecs are the estimated ones from the camera coord system
            # print (str(i.get_id()) + " " + str(aruco_2d))
            if 0 <= int(aruco_2d[0]) and int(aruco_2d[0]) <= output_image.shape[1] and \
                    0 <= int(aruco_2d[1]) and int(aruco_2d[1]) <= output_image.shape[0]:
                cv2.circle(output_image, (int(aruco_2d[0]), int(aruco_2d[1])), 3, (255, 255, 0), 3)

            logging.debug(f"{i.get_id()}: {aruco_2d}")

            # check that the marker when backprojected is within the bounds of the output_image
            if aruco_2d[0] < 0 or aruco_2d[0] > ORIGINAL_FRAME_WIDTH or aruco_2d[1] < 0 or aruco_2d[
                1] > ORIGINAL_FRAME_WIDTH:
                # not a valid marker location, ignore
                remove_aruco_vectors.append(index)
                continue

            aruco_tvecs_plot.append(i.tvec[:, 0])
            aruco_tvecs[i.get_id()] = i.tvec[:, 0]

            # assign x, y
            htk_output_vector[SORTED_VALID_SHELF_MARKERS_DICT[i.get_id()] * 2 + ARUCO_MARKER_HTK_OFFSET] = aruco_2d[
                0]
            htk_output_vector[SORTED_VALID_SHELF_MARKERS_DICT[i.get_id()] * 2 + ARUCO_MARKER_HTK_OFFSET + 1] = \
                aruco_2d[1]

        elif i.get_id() == 0:
            # if hand found, hand aruco marker should have id 0
            aruco_hand_tvec = np.array(i.tvec[:, 0])

    # ----------------------------------OPTICAL FLOW DETECTION--------------------------------------------------
    # mag, ang = optical_flow_algo.calc(output_image)
    #
    # mag, ang = mag.mean(), ang.mean()

    # optical flow detection not enabled
    mag, ang = 0, 0

    logging.debug(f"Optical flow: {mag} {ang}")

    htk_output_vector[OPTICAL_FLOW_HTK_OFFSET] = mag
    htk_output_vector[OPTICAL_FLOW_HTK_OFFSET + 1] = ang

    # output_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
    # To improve performance, optionally mark the output_image as not writeable to
    # pass by reference.
    output_image.flags.writeable = False
    results = hands.process(output_image)

    # -----------------------------------------MEDIAPIPE HAND DETECTION-----------------------------------------
    # Draw the hand annotations on the output_image.
    output_image.flags.writeable = True
    # use bgr for processing
    output_image = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)
    first_hand_points = []
    if results.multi_hand_landmarks:
        # get which hand we're looking at (look at the first hand that is detected)
        handedness = MessageToDict(results.multi_handedness[0])['classification'][0]['label']
        for hand_landmarks in results.multi_hand_landmarks:
            #        #look only at the first hand detected if there are multiple hands
            #         used_landmarks = hand_landmarks.landmark[:21]
            #         for i in range(21):
            #             landmark_pos = [used_landmarks[i].x, used_landmarks[i].y, used_landmarks[i].z]
            #             first_hand_points.append(landmark_pos)
            #             htk_output_vector[HAND_LANDMARK_OFFSET + 3 * i : HAND_LANDMARK_OFFSET + (3 * i) + 3] = landmark_pos
            #             #output_image = cv2.circle(output_image, (int(i.x * output_image.shape[1]), int(i.y * output_image.shape[0])), 2, (0, 0, 255), 3)
            #             mp_drawing.draw_landmarks(output_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            # else:
            #     #what should we put hand landmarks at if not visible? I'm putting [-1, -1, -1] for now
            #     htk_output_vector[HAND_LANDMARK_OFFSET : HAND_LANDMARK_OFFSET + 63] = [-1 for i in range(63)]

            for i in hand_landmarks.landmark[:21]:
                first_hand_points.append([i.x, i.y, i.z])
                # output_image = cv2.circle(output_image, (int(i.x * output_image.shape[1]), int(i.y * output_image.shape[0])), 2, (0, 0, 255), 3)
                mp_drawing.draw_landmarks(output_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    #        8   12  16  20
    #        |   |   |   |
    #        7   11  15  19
    #    4   |   |   |   |
    #    |   6   10  14  18
    #    3   |   |   |   |
    #    |   5---9---13--17
    #    2    \         /
    #     \    \       /
    #      1    \     /
    #       \    \   /
    #        ------0-
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (5, 6), (6, 7), (7, 8),
        (9, 10), (10, 11), (11, 12),
        (13, 14), (14, 15), (15, 16),
        (17, 18), (18, 19), (19, 20),
        (0, 5), (5, 9), (9, 13), (13, 17), (0, 17)
    ]

    if len(first_hand_points) >= 21:
        curr_time = time.time()
        # time_change = curr_time - prev_time

        # find current location of hand, (x, y)
        curr_hand_loc = hand_pos(first_hand_points, output_image)

        htk_output_vector[HAND_POS_HTK_OFFSET] = curr_hand_loc[0]
        htk_output_vector[HAND_POS_HTK_OFFSET + 1] = curr_hand_loc[1]

        cv2.circle(output_image, (int(curr_hand_loc[0]), int(curr_hand_loc[1])), 3, (255, 0, 255), 3)

        # calculate hand pos distribution stats
        bounding_box, bounding_box_size = hand_bounding_box(first_hand_points, output_image)
        for point_index in range(len(bounding_box)):
            # use the %len(bounding_box) to wrap around back to the start
            cv2.line(output_image, bounding_box[point_index], bounding_box[(point_index + 1) % len(bounding_box)],
                     (0, 0, 255), 4)

        # write text
        fingers_open = improved_gesture_recognition(first_hand_points, handedness, output_image)
        cv2.putText(output_image, f"{fingers_open}", (bounding_box[0][0], bounding_box[0][1] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1, color=(0, 0, 255), thickness=3)

        # uncomment when using fingers openclose data
        # #add to features
        # htk_output_vector[FINGERS_OFFSET : FINGERS_OFFSET + 5] = fingers_open
        # htk_output_vector[FINGERS_OFFSET + 5 : FINGERS_OFFSET + 10] = [1 if sum(fingers_open) > i else 0 for i in range(5)]

        # resize output_image to be a little smaller
        # output_image = cv2.resize(output_image, (int(output_image.shape[1] * 0.75), int(output_image.shape[0] * 0.75)))

        # ---------------------------------PROCESSING COLOR MODEL---------------------------------------------------
        min_x = float('inf')
        min_y = float('inf')
        max_x = 0
        max_y = 0
        for point in first_hand_points:
            x, y, z = point
            # mediapipe outputs as a ratio
            x = int(x * ORIGINAL_FRAME_WIDTH)
            y = int(y * ORIGINAL_FRAME_HEIGHT)
            if x < min_x:
                min_x = x
            if y < min_y:
                min_y = y
            if x > max_x:
                max_x = x
            if y > max_y:
                max_y = y

        cropped_out_points = [(min_x, min_y), (min_x, max_y), (max_x, max_y), (min_x, max_y), (min_x, min_y)]
        cv2.rectangle(output_image, (int(min_x), int(max_y)), (int(max_x), int(min_y)), (255, 255, 0), 3)

        # use the original image to get the colors (original image is in RGB)
        cropped_hand_image = image[min_y:max_y, min_x:max_x, :]

        # cv2.imshow('MediaPipe Hands', cropped_hand_image)
        # cv2.waitKey(0)

        histsat = get_hs_bins(cropped_hand_image)

        # hand might not be detected so number of pixels might be 0
        if histsat != []:
            for i in range(len(histsat)):
                # set the values in the htk_output_vector to the proportion of cropped points in each bin
                htk_output_vector[COLOR_BIN_HTK_OFFSET + i] = histsat[i]

    # --------------------------------WRITING OUT HTK VECTOR TO FILE-----------------------------------------
    logging.info(counter)
    logging.info(htk_output_vector)
    htk_output_vector = [str(i) for i in htk_output_vector]

    # store output
    output_dict[counter] = htk_output_vector

# ----------------------------------------DECLARING CONSTANTS----------------------------------------------------
# change this to visualize the detections in the image
DISPLAY_VISUAL = False

ORIGINAL_FRAME_WIDTH = 1920
ORIGINAL_FRAME_HEIGHT = 1080

OUTPUT_FRAME_WIDTH = 1280
OUTPUT_FRAME_HEIGHT = 720

VALID_BIN_MARKERS = {110, 120, 130, 210, 220, 230, 310, 320, 330, 410, 420, 430, 510, 520, 530, 610, 620, 630}

VALID_LOCALIZATION_MARKERS = {111, 121, 131, 211, 221, 231, 311, 321, 331, 411, 421, 431, 511, 521, 531, 611, 621,
                              631}

VALID_SHELF_MARKERS = VALID_BIN_MARKERS | VALID_LOCALIZATION_MARKERS

SORTED_VALID_SHELF_MARKERS_DICT = dict(
    zip(sorted(list(VALID_SHELF_MARKERS)), list(range(len(VALID_SHELF_MARKERS)))))

print(SORTED_VALID_SHELF_MARKERS_DICT)

# offsets in the HTK output array
ARUCO_MARKER_HTK_OFFSET = 0
COLOR_BIN_HTK_OFFSET = 72
OPTICAL_FLOW_HTK_OFFSET = 352
HAND_POS_HTK_OFFSET = 354

# don't flip camera view unless you want selfie view
TO_FLIP = False

intrinsic = np.array([[900., 0, 640], [0, 900, 360], [0, 0, 1]])
distortion = np.array([[0., 0., 0., 0., 0.]])

arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_1000)

# need to declare outside the if __name__ statement otherwise the process doesn't seem to have access to this
# as a global var
# create aruco detector
aruco_detector = ArUcoDetector(intrinsic, distortion, arucoDict, square_length=0.0254)

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.3, max_num_hands=1)

picked = []

frames = 0

# Hand landmark and fingers data - not in master yet apparently
# HAND_LANDMARK_OFFSET = 356 #21 * 3
# FINGERS_OFFSET = 419 #[fingers 0..4, >0 fingers, >1 finger, >2, >3, >4]

# aruco camera matrices are after image is distorted, focal length shouldn't matter since everything is adjusted
# proportionally


# to undistort the image
# initial_intrinsic = np.load("intrinsic_gopro.npy")
# initial_distortion = np.load("distortion_gopro.npy")

# all the markers that we know of so far
markers = set()

if __name__ == "__main__":
    # logging.getLogger().setLevel(logging.DEBUG)

    parser = argparse.ArgumentParser()

    # can load from configs
    parser.add_argument("--video", "-v", type=str, help="Path to input video", required=True)
    # fill in for video output
    parser.add_argument("--outfile", "-o", type=str, default="",
                        help="Path to video outfile to save to. If not provided, will not create video")  # 'hand_detection_output.mp4'

    args = parser.parse_args()

    OUTPUT_FILE = os.path.basename(args.video.split(".")[0] + ".txt")

    print(OUTPUT_FILE)

    with open(OUTPUT_FILE, "w") as outfile:
        # clear the output file
        pass


    # #live video feed from camera 0
    # hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.6, max_num_hands=1)
    # cap = cv2.VideoCapture(0)
    # #flip video horizontally
    # TO_FLIP = True

    # seeing only hands from video

    cap = cv2.VideoCapture(args.video)



    # get optimal camera matrix to undistort the image
    # newcameramtx, roi = cv2.getOptimalNewCameraMatrix(intrinsic, distortion, (1920, 1080), 1,
    #                                                (1920, 1080))

    # intrisic c_x, c_y is off

    plt.show(block=False)

    dist = 1

    # optical_flow_algo = FarnebackFlow(frame_distance=dist)

    # to write videos
    if args.outfile:
        out = cv2.VideoWriter(args.outfile, cv2.VideoWriter_fourcc(*"mp4v"), 60,
                              (OUTPUT_FRAME_WIDTH, OUTPUT_FRAME_HEIGHT))
    plt.ion()
    plt.show()

    counter = 0

    with MLSocket() as socket:
        with Manager() as manager:
            # to do the multiprocessing

            # store the result
            result = manager.dict({})

            while cap.isOpened():
                # aruco_marker = [x, y]
                # aruco_marker = [is_there, x, y]
                # output vector for use with htk, [aruco_marker[:72], color_bins[72:92], optical_flow_avg[92:94], hand_loc[94:96]], 96 dim version
                # [aruco_marker[:72], color_bins[72:352], optical_flow_avg[352:354], hand_loc[354:356]] 356-dimversion

                # update to 429 when using hand landmark + fingers openclose data
                # htk_output_vector = [0 for i in range(429)]

                success, image = cap.read()
                if not success:
                    print("Ignoring empty camera frame.")
                    # If loading a video, use 'break' instead of 'continue'.
                    break

                logging.debug(f"Counter: {counter}")

                counter += 1

                # image = cv2.undistort(image, newcameramtx, distortion, None)

                # reduce resolution of image

                # get the htk output vector from the image
                # htk_output_vector = extract_features(image, hands)

                # call to process
                p = Process(target=extract_features, args=(image, result, frames))

                # with open(OUTPUT_FILE, "a") as outfile:
                #     outfile.write(" ".join(htk_output_vector) + "\n")

                # #removing distortion on output_image
                # output_image = cv2.undistort(output_image, intrinsic, distortion, None)

                # horizontal margin to be removed from the video (one for left and one for right so total fraction removed is
                # double this proportion)
                horizontal_margin = 0

                if DISPLAY_VISUAL:
                    output_image = output_image[:, int(output_image.shape[1] * horizontal_margin): int(
                        output_image.shape[1] * (1 - horizontal_margin))]

                    output_image = cv2.resize(output_image, (OUTPUT_FRAME_WIDTH, OUTPUT_FRAME_HEIGHT))

                    cv2.imshow('MediaPipe Hands', output_image)

                    if args.outfile:
                        out.write(output_image)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break



                print (f"{frames} start: {time.time()}")
                p.start()
                print(f"{frames} after start: {time.time()}")

                frames += 1

                print(frames)
            hands.close()
            cap.release()

            if args.outfile:
                out.release()

            # Closes all the frames
            cv2.destroyAllWindows()

