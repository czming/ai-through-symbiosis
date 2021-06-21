import cv2
import mediapipe as mp
from gesture_recognition_functions import *
from google.protobuf.json_format import MessageToDict
import time
import matplotlib.pyplot as plt

class ArUcoDetector:
    def __init__(self, intrinsic: np.ndarray, distortion: np.ndarray, aruco_dict, square_length=1.):
        assert intrinsic.shape == (3,3)
        self.intrinsic = intrinsic
        self.distortion = distortion
        self.aruco_dict = aruco_dict
        self.square_length = square_length

    def getCorners(self, image: np.ndarray):
        corners, ids, rejected = cv2.aruco.detectMarkers(image, self.aruco_dict, cameraMatrix=self.intrinsic, distCoeff=self.distortion)
        if corners:
            #try and find all 4 points
            rvecs, tvecs, markerPoints = cv2.aruco.estimatePoseSingleMarkers(corners, self.square_length, self.intrinsic, self.distortion)
            ids = ids.flatten()
            return [ArUcoMarker(corners[i][0], int(ids[i]), rvecs[i].reshape((3,1)), tvecs[i].reshape((3,1))) for i in range(len(corners))]
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
        assert rvec.shape == tvec.shape == (3,1)

        self.corners = corners
        self._id = _id
        self.rvec = rvec
        self.tvec = tvec

    def get_id(self):
        return self._id


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
    #initially object not in hand, so we need to pick
    picking = True
    for line in infile:
        line = [int(i) for i in line.strip().split(",")]
        curr_pick = dict(zip(pick_indices, line))
        curr_pick["picked"] = picking
        picking = not picking
        picklist.append(curr_pick)
    return picklist


mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# #live video feed from camera 0
# hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.6, max_num_hands=1)
# cap = cv2.VideoCapture(0)
# #flip video horizontally
# TO_FLIP = True

#seeing only hands from video
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.2, max_num_hands=1)
cap = cv2.VideoCapture("C:/Users/chngz/Documents/AI through Symbiosis/AI through Symbiosis/Red 1 ArUco.mp4")
#don't flip camera view unless you want selfie view
TO_FLIP = False

#contains one instruction for pick and another instruction for drop
pick_list = parse_picklist("C:/Users/chngz/Documents/AI through Symbiosis/AI through Symbiosis/picklist.csv")

picked = []

#start out open
hand_closed = False
hand_open = True
curr_holding = False
last_change = 0
#give a buffer of time due to inaccuracy in detection
TIME_CHANGE_BUFFER = 5

MARKER_TIME_BUFFER = 0.5


MOVEMENT_MOE = 5
REFRESH_TIME = 0.05

prev_hand_loc = ()
curr_hand_loc = ()
prev_time = time.time()

hand_mean_x = 0
hand_mean_y = 0
hand_var_x = 0
hand_var_y = 0

closed_hand_mean_x = 0
closed_hand_mean_y = 0
closed_hand_var_x = 0
closed_hand_var_y = 0
closed_hand_frames = 0

#-1 for uninitialized
left_center_right = -1
top_center_bottom = -1

#max distance between hand and average so far during pick attempt
max_hand_discrepancy = 0

#all the markers that we know of so far
markers = set()

#the relative tvec locations between markers

frames = 0

FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

intrinsic = np.load("intrinsic.npy")
distortion = np.load("distortion.npy")

arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_1000)

#get optimal camera matrix to undistort the image
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(intrinsic, distortion, (1920, 1080), 1,
                                                  (1920, 1080))

#create aruco detector
aruco_detector = ArUcoDetector(intrinsic, distortion,arucoDict, square_length=0.05)

curr_figure = None

#setting up plotting for the tvecs
aruco_fig = plt.figure()

aruco_tvec_ax = aruco_fig.add_axes(projection="3d")

aruco_tvec_sp = aruco_fig.add_subplot(projection='3d')

aruco_tvec_sp.set_xlim(-1,1)
aruco_tvec_sp.set_ylim(-1,1)
aruco_tvec_sp.set_zlim(0,10)

plt.ion()
plt.show()


# ##to write videos
# out = cv2.VideoWriter('hand_detection_output.mp4', cv2.VideoWriter_fourcc(*"mp4v"), 10,
# 					  (FRAME_WIDTH, FRAME_HEIGHT))

while cap.isOpened():
    success, image = cap.read()

    curr_detected_markers = set()

    if not success:
        print("Ignoring empty camera frame.")
        # If loading a video, use 'break' instead of 'continue'.
        break

    if TO_FLIP:
        # Flip the image horizontally for a later selfie-view display, and convert
        # the BGR image to RGB.
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = cv2.resize(image, (1920, 1080))

    aruco_vectors = aruco_detector.getCorners(image)

    #carriage return to delete previous line
    print ("\r", "number of markers: " + str(len(aruco_vectors)), end = ' ')

    for marker in aruco_vectors:
        curr_detected_markers.add(marker.get_id())
        markers.add(marker.get_id())

        #draw axis for the aruco markers
        cv2.aruco.drawAxis(image, intrinsic, distortion, marker.rvec, marker.tvec, 0.05)

    print (curr_detected_markers, end = " ")
    print (markers, end = " ")



    #print (len(picked), end = ' ')

    aruco_tvecs = []

    for i in aruco_vectors:
        #aruco has an extra dimension added so we're removing it here
        aruco_tvecs.append([j[0] for j in i.tvec])

        points = i.corners

        # plot the lines between the corners of the aruco markers
        cv2.line(image, (int(points[0][0]), points[0][1]), (int(points[1][0]), points[1][1]), (0, 255, 0), 2)
        cv2.line(image, (int(points[1][0]), points[1][1]), (int(points[2][0]), points[2][1]), (0, 255, 0), 2)
        cv2.line(image, (int(points[2][0]), points[2][1]), (int(points[3][0]), points[3][1]), (0, 255, 0), 2)
        cv2.line(image, (int(points[3][0]), points[3][1]), (int(points[0][0]), points[0][1]), (0, 255, 0), 2)

        cv2.putText(image, f"id: {i.get_id()}", (int(points[0][0]), int(points[0][1] - 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 255), thickness=3)

    #print (i.tvec)

    #want at least 4 points
    if len(aruco_tvecs) >= 4:
        aruco_tvecs_np = np.array(aruco_tvecs)
        aruco_plot_plane = None
        try:
            aruco_plane = get_best_fit_plane(aruco_tvecs_np)
            x = np.linspace(-1, 1, 10)
            y = np.linspace(-1, 1, 10)
            X, Y = np.meshgrid(x, y)
            Z = aruco_plane[0] + aruco_plane[1] * X + aruco_plane[2] * Y

            aruco_plot_plane = aruco_tvec_sp.plot_surface(X, Y, Z, alpha=0.5)

        except np.linalg.LinAlgError:
            #insufficient points, singular matrix
            pass

        #plot x, y, z
        points = aruco_tvec_sp.scatter3D(aruco_tvecs_np[:,0], aruco_tvecs_np[:,1], aruco_tvecs_np[:,2])

        aruco_fig.canvas.draw()

        points.remove()
        if aruco_plot_plane:
            aruco_plot_plane.remove()

    #removing distortion on image
    image = cv2.undistort(image, intrinsic, distortion, None, newcameramtx)


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
                first_hand_points.append([i.x, i.y, i.z])
                #image = cv2.circle(image, (int(i.x * image.shape[1]), int(i.y * image.shape[0])), 2, (0, 0, 255), 3)
                mp_drawing.draw_landmarks(
                   image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    if len(first_hand_points) >= 21:
        curr_time = time.time()
        time_change = curr_time - prev_time

        # find current location of hand, (x, y)
        curr_hand_loc = hand_pos(first_hand_points, image)
        cv2.circle(image, (int(curr_hand_loc[0]), int(curr_hand_loc[1])), 3, (255, 0, 255), 3)

        # calculate hand pos distribution stats
        hand_mean_x, hand_var_x = calculate_new_mean_variance(hand_mean_x, hand_var_x, frames, curr_hand_loc[0])
        hand_mean_y, hand_var_y = calculate_new_mean_variance(hand_mean_y, hand_var_y, frames, curr_hand_loc[1])

        hand_sd_x = int(hand_var_x ** 0.5)
        hand_sd_y = int(hand_var_y ** 0.5)
        #euclidean_sd = (hand_var_x + hand_var_y) ** (1/2)

        cv2.circle(image, (int(hand_mean_x), int(hand_mean_y)), 3, (255, 255, 0), 3)
        #cv2.circle(image, (int(hand_mean_x), int(hand_mean_y)), int(euclidean_sd), (255, 255, 0), 3)
        cv2.ellipse(image, (int(hand_mean_x), int(hand_mean_y)), (2 * hand_sd_x, 2 * hand_sd_y), 0, 0, 360, (255, 255, 0), 3)

        # find bounding box of hand
        bounding_box, bounding_box_size = hand_bounding_box(first_hand_points, image)
        for point_index in range(len(bounding_box)):
            #use the %len(bounding_box) to wrap around back to the start
            cv2.line(image, bounding_box[point_index], bounding_box[(point_index + 1) % len(bounding_box)],
                             (0, 255, 0), 4)

        #write text
        fingers_open = improved_gesture_recognition(first_hand_points, handedness, image)
        cv2.putText(image, f"{fingers_open}", (bounding_box[0][0], bounding_box[0][1] - 20) , cv2.FONT_HERSHEY_SIMPLEX,
                                               fontScale = 1, color = (0,0, 255), thickness = 3)

        if time.time() - last_change >= TIME_CHANGE_BUFFER and sum(fingers_open) == 5 and not hand_open:
            #detected that hand has reopened, process the pick
            hand_open = True
            #pick the stuff after the whole thing has been processed
            picked.append(pick_list.pop(0))
            # if closed_hand_mean_x > hand_mean_x + hand_sd_x:
            # 	# to the right
            # 	left_center_right = 1
            # elif closed_hand_mean_x < hand_mean_x - hand_sd_x:
            # 	left_center_right = -1
            print (f"\n{top_center_bottom}{left_center_right}")
            left_center_right = -1
            top_center_bottom = -1
            max_hand_discrepancy = 0

            # #reset closed hand variables
            # closed_hand_mean_x = 0
            # closed_hand_mean_y = 0
            # closed_hand_var_x = 0
            # closed_hand_var_y = 0
            # closed_hand_frames = 0

        elif time.time() - last_change >= TIME_CHANGE_BUFFER and sum(fingers_open) <= 2 and hand_open:
            #closed palms seem harder to detect due to angle, so only need to see 3 closed fingers
            #hand was previously open, so something is being picked up/dropped

            #change state of curr_holding
            curr_holding = not curr_holding
            hand_open = False
            last_change = time.time()

        if time.time() - last_change < TIME_CHANGE_BUFFER and sum(fingers_open) <= 2:
            #during transition time and hand is closed, find average location
            # closed_hand_mean_x, closed_hand_var_x = calculate_new_mean_variance(closed_hand_mean_x,
            # 																	closed_hand_var_x,
            # 																	closed_hand_frames,
            # 																	curr_hand_loc[0])
            # closed_hand_mean_y, closed_hand_var_y = calculate_new_mean_variance(closed_hand_mean_y,
            # 																	closed_hand_var_y,
            # 																	closed_hand_frames,
            # 																	curr_hand_loc[1])
            #
            # frames += 1


            if max_hand_discrepancy < get_distance(curr_hand_loc, (hand_mean_x, hand_mean_y)):
                # hand is at greatest distance from mean detected so far, detect hand position relative to fiducials
                #initialize closest_distance as an integer greater than size of image
                closest_distance = 10000
                max_hand_discrepancy = get_distance(curr_hand_loc, (hand_mean_x, hand_mean_y))
                for marker_index in range(len(aruco_vectors)):
                    #later replace corners with a persistent measure of fiducial location
                    marker = aruco_vectors[marker_index]
                    marker_corners = marker.corners

                    # top left point of marker should be to the left and underneath the hand
                    if marker_corners[0][0] < curr_hand_loc[0] and marker_corners[0][1] > curr_hand_loc[1] and \
                        get_distance(marker_corners[0], curr_hand_loc) < closest_distance:

                    # #this version just finds nearest fiducial marker, problematic since picker might pick towards one
                    #side of the bin
                    # if get_distance(marker[0], curr_hand_loc) < closest_distance:

                        closest_distance = get_distance(marker_corners[0], curr_hand_loc)
                        #get left center right based on marker index, ids of aruco stored as int
                        left_center_right = marker.get_id() % 10
                        top_center_bottom = marker.get_id()// 10


        # motion related stuff that is checked every fixed period (otherwise different frame refresh rates lead to
        # inconsistencies in measuring movement in pixels)
        # if (time_change > REFRESH_TIME):
        #     # checks when the time_change is greater than the REFRESH TIME
        #
        #     print (f"curr_holding: {curr_holding}", handedness, end = " ")
        #     if prev_hand_loc != () and (curr_hand_loc[0] > prev_hand_loc[0] + MOVEMENT_MOE * time_change / REFRESH_TIME):
        #         # curr_hand is more than MOVEMENT_MOE pixels to the right of prev_hand_loc
        #         print("Moving right", end=" ")
        #     elif prev_hand_loc != () and (curr_hand_loc[0] < prev_hand_loc[0] - MOVEMENT_MOE * time_change / REFRESH_TIME):
        #         print("Moving left", end=" ")
        #
        #     if prev_hand_loc != () and (curr_hand_loc[1] > prev_hand_loc[1] + MOVEMENT_MOE * time_change / REFRESH_TIME):
        #         # curr_hand is more than MOVEMENT_MOE pixels to the bottom of prev_hand_loc
        #         print("Moving down", end=" ")
        #     elif prev_hand_loc != () and (curr_hand_loc[1] < prev_hand_loc[1] - MOVEMENT_MOE * time_change / REFRESH_TIME):
        #         print("Moving up", end=" ")
        #
        #
        #
        #     prev_hand_loc = curr_hand_loc
        #     prev_time = curr_time

        frames += 1

    #resize image to be a little smaller
    #image = cv2.resize(image, (int(image.shape[1] * 0.75), int(image.shape[0] * 0.75)))

    cv2.imshow('MediaPipe Hands', image)

    #out.write(image)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

hands.close()
cap.release()

#out.release()

# Closes all the frames
cv2.destroyAllWindows()