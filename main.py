import argparse

import cv2
import mediapipe as mp
from gesture_recognition_functions import *
from google.protobuf.json_format import MessageToDict
import time
import matplotlib.pyplot as plt

class ArUcoDetector:
    def __init__(self, intrinsic: np.ndarray, distortion: np.ndarray, aruco_dict, square_length=1):
        assert intrinsic.shape == (3,3)
        self.intrinsic = intrinsic
        self.distortion = distortion
        self.aruco_dict = aruco_dict
        self.square_length = square_length

    def getCorners(self, image: np.ndarray):
        #corners seem to be (x, y)
        corners, ids, rejected = cv2.aruco.detectMarkers(image, self.aruco_dict, cameraMatrix=self.intrinsic, distCoeff=self.distortion)

        if corners:
            #try and find all 4 points
            rvecs, tvecs, markerPoints = cv2.aruco.estimatePoseSingleMarkers(corners, self.square_length, self.intrinsic, self.distortion)
            ids = ids.flatten()

            #ensure that the 3D points and the 2D points are aligned
            assert len(rvecs) == len(corners)

            #only include those that are not in the horizontal margin (use top left point of marker as the reference)
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

      
def get_hand_corners(hand_points:list) -> np.ndarray:
    """
	Get pseudo ArUco landmarks for hand positions

	params:
		- hand_points: list of MediaPipe hand positions

	return:
		- ArUco corners
	"""
    top_left = np.asarray(hand_points[17][:2]) # pinky
    top_right = np.asarray(hand_points[5][:2]) # index
    bottom_right = np.asarray(hand_points[1][:2]) # index
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
    #initially object not in hand, so we need to pick
    picking = True
    for line in infile:
        line = [int(i) for i in line.strip().split(",")]
        curr_pick = dict(zip(pick_indices, line))
        curr_pick["picked"] = picking
        picking = not picking
        picklist.append(curr_pick)
    return picklist


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--video", "-v", type=str, default="G:/My Drive/Georgia Tech/AI Through Symbiosis/GOPR3788.MP4", help="Path to input video")
    parser.add_argument("--pickpath", "-pp", type=str, default="C:/Users/chngz/Documents/AI through Symbiosis/AI through Symbiosis/picklist.csv", help="Path to picklist")
    parser.add_argument("--outfile", "-o", type=str, help="Path to video outfile to save to. If not provided, will not create video") # 'hand_detection_output.mp4'
    parser.add_argument("--check_pickpath", "-cp", action="store_false", help="Add this flag if you want to work with a picklist. Also be sure to pass a --pickpath argument")

    args = parser.parse_args()

    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands

    # #live video feed from camera 0
    # hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.6, max_num_hands=1)
    # cap = cv2.VideoCapture(0)
    # #flip video horizontally
    # TO_FLIP = True

    #seeing only hands from video
    hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.3, max_num_hands=1)
    cap = cv2.VideoCapture(args.video)
    #don't flip camera view unless you want selfie view
    TO_FLIP = False

    #contains one instruction for pick and another instruction for drop
    pick_list = parse_picklist(args.pickpath) if args.check_pickpath else None

    picked = []

    #start out open
    hand_closed = False
    hand_open = True
    curr_holding = False
    last_change = 0

    TIME_CHANGE_BUFFER = 5

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

    frames = 0

    FRAME_WIDTH = 1280
    FRAME_HEIGHT = 720

    VALID_SHELF_MARKERS = {1, 2, 3, 11, 12, 13, 21, 22, 23, 31, 32, 33, 41, 42, 43}

    intrinsic = np.load("intrinsic_gopro.npy")
    distortion = np.load("distortion_gopro.npy")
  
    #all the markers that we know of so far
    markers = set()

    #stores the relative position of the markers (markers_rel_pos[id1][id2][id3]) which gives the vector between
    #id1 and id3 in terms of [u, v], where u is the vector between id1 and id2 and v is the vector perpendicular
    #to u and the normal vector of the plane
    markers_rel_pos = {}

    #the relative tvec locations between markers

    frames = 0

    arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_1000)

    #get optimal camera matrix to undistort the image
    #newcameramtx, roi = cv2.getOptimalNewCameraMatrix(intrinsic, distortion, (1920, 1080), 1,
    #                                                (1920, 1080))

    #create aruco detector
    aruco_detector = ArUcoDetector(intrinsic, distortion, arucoDict, square_length=0.0254)

    curr_figure = None

    #setting up plotting for the tvecs
    aruco_fig = plt.figure()

    aruco_tvec_ax = aruco_fig.add_axes(projection="3d")

    #scatter plot of aruco_tvecs
    aruco_tvec_sp = aruco_fig.add_subplot(projection='3d')

    #set limits for the scatter plot
    aruco_tvec_sp.set_xlim(-2,2)
    aruco_tvec_sp.set_ylim(0,3)
    aruco_tvec_sp.set_zlim(0,3)

    plt.show(block=False)

    #to write videos
    if args.outfile:
        out = cv2.VideoWriter(args.outfile, cv2.VideoWriter_fourcc(*"mp4v"), 10,
                          (FRAME_WIDTH, FRAME_HEIGHT))
    plt.ion()
    plt.show()

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            break

        #image = cv2.undistort(image, newcameramtx, distortion, None)

        if TO_FLIP:
            # Flip the image horizontally for a later selfie-view display, and convert
            # the BGR image to RGB.
            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        #image = cv2.resize(image, (1920, 1080))

        aruco_vectors = aruco_detector.getCorners(image)

        #scatter plot figure objects
        aruco_sp_points = None
        aruco_sp_predicted_points = None
        aruco_plot_plane = None
        hand_sp_points = None

        print ("\rnumber of markers: " + str(len(aruco_vectors)), end = " ")
    
        for marker in aruco_vectors:
            if marker.get_id() not in markers and marker.get_id() in VALID_SHELF_MARKERS:
                #new marker
                markers.add(marker.get_id())
                markers_rel_pos[marker.get_id()] = {}

            #draw axis for the aruco markers
            #cv2.aruco.drawAxis(image, newcameramtx, distortion, marker.rvec, marker.tvec, 0.05)

        #id to tvec mapping for aruco markers
        aruco_tvecs = {}

        #gathering the tvecs for plotting using matplotlib
        aruco_tvecs_plot = []

        #aruco marker for hand
        aruco_hand_tvec = None

        for i in aruco_vectors:
            if i.get_id() in VALID_SHELF_MARKERS:
                #make sure the aruco marker is valid shelf marker
                aruco_tvecs_plot.append(i.tvec[:,0])
                aruco_tvecs[i.get_id()] = i.tvec[:, 0]
                aruco_2d = cv2.projectPoints(np.array([0.0,0.0,0.0]), np.array([0.0,0.0,0.0]), i.tvec[:,0], intrinsic, distortion)[0][0][0]
                #objectPoints might be the actual 3D points, tvecs and rvecs are the estimated ones from the camera coord system
                #print (str(i.get_id()) + " " + str(aruco_2d))
                if 0 <= int(aruco_2d[0]) and int(aruco_2d[0]) <= image.shape[1] and \
                        0 <= int(aruco_2d[1]) and int(aruco_2d[1]) <= image.shape[0]:
                    cv2.circle(image, (int(aruco_2d[0]), int(aruco_2d[1])), 3, (255,255,255), 3)
            elif i.get_id() == 0:
                #if hand found, hand aruco marker should have id 0
                aruco_hand_tvec = np.array(i.tvec[:,0])

        #print(aruco_tvecs.keys(), end=" ")
        #print(markers, end=" ")

        #want at least 4 points
        if len(aruco_tvecs) >= 4:
            #aruco_tvecs numpy array
            aruco_tvecs_plot_np = np.array(aruco_tvecs_plot)
            aruco_plot_plane = None
            try:
                aruco_plane = get_best_fit_plane(aruco_tvecs_plot_np)
                #vector that is perpendicular to the plane
                aruco_plane_normal_vector = [-aruco_plane[1], -aruco_plane[2], 1]
                x = np.linspace(-1.5, 1.5, 10)
                y = np.linspace(-1.5, 1.5, 10)
                X, Y = np.meshgrid(x, y)
                Z = aruco_plane[0] + aruco_plane[1] * X + aruco_plane[2] * Y

                aruco_plot_plane = aruco_tvec_sp.plot_surface(X, Y, Z, alpha=0.3, color = (0,0,1))

            except np.linalg.LinAlgError:
                #insufficient points, singular matrix
                continue

            #update the adjusted positions for aruco_tvecs
            for i in aruco_tvecs.keys():
                aruco_tvecs[i] = best_fit_point([-aruco_plane[1], -aruco_plane[2], 1, aruco_plane[0]], aruco_tvecs[i])

            adjusted_aruco_tvecs_plot_np = []

            for i in aruco_tvecs.keys():
                # get the adjusted marker positions for plotting
                adjusted_aruco_tvecs_plot_np.append(aruco_tvecs[i])

            #get the sorted ids
            sorted_aruco_ids = sorted(aruco_tvecs.keys())

            for i in range(len(sorted_aruco_ids)):
                #looking at each aruco marker
                for j in range(i + 1, len(sorted_aruco_ids)):
                    #looking at each aruco marker after that (i < j, assuming that aruco markers are unique)
                    aruco_id1 = sorted_aruco_ids[i]
                    aruco_id2 = sorted_aruco_ids[j]
                    assert aruco_id1 != aruco_id2, "Aruco marker ids must be unique"

                    if aruco_id2 not in markers_rel_pos[aruco_id1].keys():
                        markers_rel_pos[aruco_id1][aruco_id2] = {}

                    # use u and v which as basis vectors (calculated using aruco_id1 and aruco_id2) for the vector
                    # between aruco_id1 and aruco_id3

                    #u is the vector from aruco_id1 to aruco_id2
                    u = np.array(aruco_tvecs[aruco_id2]) - np.array(aruco_tvecs[aruco_id1])

                    #v is the vector that is perpendicular to u in the plane (i.e. perpendicular to both u and the normal vector
                    #for the plane)
                    v = np.cross(u, aruco_plane_normal_vector)

                    #check that the vectors are perpendicular to each other
                    assert abs(np.sum(u * v)) < 0.0000001, "Vectors are not perpendicular"
                    assert abs(np.sum(u * np.array(aruco_plane_normal_vector))) < 0.0000001, "Vectors are not perpendicular"
                    assert abs(np.sum(v * np.array(aruco_plane_normal_vector))) < 0.0000001, "Vectors are not perpendicular"

                    for k in range(len(sorted_aruco_ids)):
                        if i == k or j == k:
                            #ignore if the same vector because it will be [1, 0] since the vector between them is
                            #one of the basis vector, or if the start id is the id itself
                            continue

                        aruco_id3 = sorted_aruco_ids[k]

                        #vector bewteen id3 and id1
                        curr_vector = np.array(aruco_tvecs[aruco_id3]) - np.array(aruco_tvecs[aruco_id1])

                        #change of base of relative position of marker2 from marker1 to the basis vectors
                        #of the vector from marker1 to marker2 and the vector perpendicular to that in the plane
                        u_v_coords = in_basis_vector(curr_vector, u, v)

                        #print (f"{aruco_id1} {aruco_id2} {aruco_id3}: {u_v_coords}")

                        if aruco_id3 not in markers_rel_pos[aruco_id1][aruco_id2].keys():
                            #[x_mean, y_mean, num_readings], ignoring the variance
                            markers_rel_pos[aruco_id1][aruco_id2][aruco_id3] = [0, 0, 0]

                        curr_coords = markers_rel_pos[aruco_id1][aruco_id2][aruco_id3]

                        #take x as the one in the direction of the id1 to id2 vector, fill 0 as variance since not using it
                        x_coords = calculate_new_mean_variance(curr_coords[0], 0, curr_coords[2], u_v_coords[0])[0]

                        y_coords = calculate_new_mean_variance(curr_coords[1], 0, curr_coords[2], u_v_coords[1])[0]

                        markers_rel_pos[aruco_id1][aruco_id2][aruco_id3] = [x_coords, y_coords, curr_coords[2] + 1]

            adjusted_aruco_tvecs_plot_np = np.array(adjusted_aruco_tvecs_plot_np)

            predicted_aruco_tvecs = {}
            predicted_aruco_tvecs_plot = []

            for marker in markers:
                #if marker in aruco_tvecs.keys():
                    #if marker is detected in the current frame, ignore

                    #continue

                #looking at the undetected markers (marker is an undetected marker)

                #[x, y, z, num_readings], ignoring variance, only looking at mean predicted location
                predicted_loc = [0,0,0,0]

                for aruco_id1 in markers_rel_pos.keys():
                    if aruco_id1 not in aruco_tvecs.keys():
                        #if the aruco marker is not detected then won't be used (can add functionality to detect second
                        #degree predictions, e.g. use the predicted marker to predict another marker's location, but that
                        #is not implemented here)
                        continue
                    for aruco_id2 in markers_rel_pos[aruco_id1].keys():
                        if aruco_id2 not in aruco_tvecs.keys():
                            continue
                        if marker in markers_rel_pos[aruco_id1][aruco_id2].keys():
                            #found the marker

                            #getting the location in u and v (u and v are the basis vectors)
                            u_v_loc = markers_rel_pos[aruco_id1][aruco_id2][marker][:2]

                            num_readings = markers_rel_pos[aruco_id1][aruco_id2][marker][2]

                            # u is the vector from marker to aruco_id2
                            u = np.array(aruco_tvecs[aruco_id2]) - np.array(aruco_tvecs[aruco_id1])

                            # v is the vector that is perpendicular to u in the plane (i.e. perpendicular to both u and the normal vector
                            # for the plane)
                            v = np.cross(u, aruco_plane_normal_vector)

                            #the location predicted by current
                            #determined by original location of marker, then the vector to aruco_id2 after a change of
                            #base back to x, y, z

                            curr_predicted = u * u_v_loc[0] + v * u_v_loc[1] + aruco_tvecs[aruco_id1]

                            predicted_loc[:3] = predicted_loc[:3] + curr_predicted[:3] * num_readings

                            # for i in range(num_readings):
                            #     #repeat based on number of readings for this current combination
                            #     predicted_loc[0] = calculate_new_mean_variance(predicted_loc[0], 0, predicted_loc[3] + i,
                            #                                                    curr_predicted[0])[0]
                            #     predicted_loc[1] = calculate_new_mean_variance(predicted_loc[1], 0, predicted_loc[3] + i,
                            #                                                    curr_predicted[1])[0]
                            #     predicted_loc[2] = calculate_new_mean_variance(predicted_loc[2], 0, predicted_loc[3] + i,
                            #                                                    curr_predicted[2])[0]

                            #update num readings
                            predicted_loc[3] = predicted_loc[3] + num_readings

                if predicted_loc[3] != 0:
                    #there are detected readings
                    #take the simple average based on number of readings across the different markers
                    predicted_aruco_tvecs[marker] = [i / predicted_loc[3] for i in predicted_loc[:3]]
                    predicted_aruco_tvecs_plot.append(predicted_aruco_tvecs[marker])

                    #print (f"{marker}: {predicted_aruco_tvecs[marker]}")

            if len(predicted_aruco_tvecs) > 0:
                predicted_aruco_tvecs_plot_np = np.array(predicted_aruco_tvecs_plot)
                aruco_sp_predicted_points = aruco_tvec_sp.scatter3D(predicted_aruco_tvecs_plot_np[:,0], predicted_aruco_tvecs_plot_np[:,1], predicted_aruco_tvecs_plot_np[:,2],
                                                      c=[(0,0,0) for i in predicted_aruco_tvecs_plot_np])
                for i in predicted_aruco_tvecs_plot_np:
                    #draw the aruco points back on the image
                    aruco_2d = cv2.projectPoints(np.array([0.0,0.0,0.0]), np.array([0.0,0.0,0.0]), i, intrinsic, distortion)[0][0][0]
                    if 0 <= int(aruco_2d[0]) and int(aruco_2d[0]) <= image.shape[1] and \
                        0 <= int(aruco_2d[1]) and int(aruco_2d[1]) <= image.shape[0]:
                        cv2.circle(image, (int(aruco_2d[0]), int(aruco_2d[1])), 3, (0, 255, 0), 3)

            #plot x, y, z for the aruco markers
            aruco_sp_points = aruco_tvec_sp.scatter3D(aruco_tvecs_plot_np[:,0], aruco_tvecs_plot_np[:,1],   aruco_tvecs_plot_np[:,2],
                                                      c=[(1,0,0) for i in aruco_tvecs_plot_np])
            #print (aruco_tvecs_np)


        if aruco_hand_tvec is not None:
            #if the hand is detected, check if the hand is in front or behind the plane
            if aruco_plane[0] + aruco_plane[1] * aruco_hand_tvec[0] + aruco_plane[2] * aruco_hand_tvec[1] > aruco_hand_tvec[2]:
                aruco_hand_sp_points = aruco_tvec_sp.scatter3D(aruco_hand_tvec[0], aruco_hand_tvec[1],
                                                               aruco_hand_tvec[2], color=(0,1,0))
            else:
                aruco_hand_sp_points = aruco_tvec_sp.scatter3D(aruco_hand_tvec[0], aruco_hand_tvec[1], aruco_hand_tvec[2],
                                                           color = (0,0,0))
            print (aruco_hand_tvec, end = " ")
            print (aruco_plane[0] + aruco_plane[1] * aruco_hand_tvec[0] + aruco_plane[2] * aruco_hand_tvec[1], aruco_hand_tvec[2], end = " ")

        for marker_index in range(len(aruco_vectors)):
            points = aruco_vectors[marker_index].corners
            #plot the lines between the corners of the aruco markers

            #print (aruco_vectors[marker_index].get_id(), points[0])
            cv2.line(image, (int(points[0][0]), int(points[0][1])), (int(points[1][0]), int(points[1][1])),
                     (0, 255, 0), 2)
            cv2.line(image, (int(points[1][0]), int(points[1][1])), (int(points[2][0]), int(points[2][1])),
                     (0, 255, 0), 2)
            cv2.line(image, (int(points[2][0]), int(points[2][1])), (int(points[3][0]), int(points[3][1])),
                     (0, 255, 0), 2)
            cv2.line(image, (int(points[3][0]), int(points[3][1])), (int(points[0][0]), int(points[0][1])),
                     (0, 255, 0), 2)

            marker_id = aruco_vectors[marker_index].get_id()
            #use the top left point as the marker's coordinates
            marker_coords = points[0]

            cv2.putText(image, f"id: {marker_id}", (int(marker_coords[0]), int(marker_coords[1] - 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0,0, 255), thickness=3)


        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        results = hands.process(image)

        #-----------------------------------------MEDIAPIPE HAND DETECTION-----------------------------------------
        # # Draw the hand annotations on the image.
        # image.flags.writeable = True
        # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # first_hand_points = []
        # if results.multi_hand_landmarks:
        #     #get which hand we're looking at (look at the first hand that is detected)
        #     handedness = MessageToDict(results.multi_handedness[0])['classification'][0]['label']
        #     for hand_landmarks in results.multi_hand_landmarks:
        #         #look only at the first hand detected if there are multiple hands
        #         for i in hand_landmarks.landmark[:21]:
        #             first_hand_points.append([i.x, i.y, i.z])
        #             #image = cv2.circle(image, (int(i.x * image.shape[1]), int(i.y * image.shape[0])), 2, (0, 0, 255), 3)
        #             mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        #
        # if len(first_hand_points) >= 21:
        #     curr_time = time.time()
        #     time_change = curr_time - prev_time
        #
        #     # find current location of hand, (x, y)
        #     curr_hand_loc = hand_pos(first_hand_points, image)
        #     cv2.circle(image, (int(curr_hand_loc[0]), int(curr_hand_loc[1])), 3, (255, 0, 255), 3)
        #
        #     # calculate hand pos distribution stats
        #     hand_mean_x, hand_var_x = calculate_new_mean_variance(hand_mean_x, hand_var_x, frames, curr_hand_loc[0])
        #     hand_mean_y, hand_var_y = calculate_new_mean_variance(hand_mean_y, hand_var_y, frames, curr_hand_loc[1])
        #
        #     hand_sd_x = int(hand_var_x ** 0.5)
        #     hand_sd_y = int(hand_var_y ** 0.5)
        #     #euclidean_sd = (hand_var_x + hand_var_y) ** (1/2)
        #
        #     cv2.circle(image, (int(hand_mean_x), int(hand_mean_y)), 3, (255, 255, 0), 3)
        #     #cv2.circle(image, (int(hand_mean_x), int(hand_mean_y)), int(euclidean_sd), (255, 255, 0), 3)
        #     cv2.ellipse(image, (int(hand_mean_x), int(hand_mean_y)), (2 * hand_sd_x, 2 * hand_sd_y), 0, 0, 360, (255, 255, 0), 3)
        #
        #     # find bounding box of hand
        #     bounding_box, bounding_box_size = hand_bounding_box(first_hand_points, image)
        #     for point_index in range(len(bounding_box)):
        #         #use the %len(bounding_box) to wrap around back to the start
        #         cv2.line(image, bounding_box[point_index], bounding_box[(point_index + 1) % len(bounding_box)],
        #                         (0, 0, 255), 4)
        #
        #     #write text
        #     fingers_open = improved_gesture_recognition(first_hand_points, handedness, image)
        #     cv2.putText(image, f"{fingers_open}", (bounding_box[0][0], bounding_box[0][1] - 20) , cv2.FONT_HERSHEY_SIMPLEX,
        #                                         fontScale = 1, color = (0,0, 255), thickness = 3)
        #
        #     if time.time() - last_change >= TIME_CHANGE_BUFFER and sum(fingers_open) == 5 and not hand_open:
        #         #detected that hand has reopened, process the pick
        #         hand_open = True
        #         #pick the stuff after the whole thing has been processed
        #         if pick_list:
        #             picked.append(pick_list.pop(0))
        #         print (f"\n{top_center_bottom}{left_center_right}", end = " ")
        #         left_center_right = -1
        #         top_center_bottom = -1
        #         max_hand_discrepancy = 0
        #
        #     elif time.time() - last_change >= TIME_CHANGE_BUFFER and sum(fingers_open) <= 2 and hand_open:
        #         #closed palms seem harder to detect due to angle, so only need to see 3 closed fingers
        #         #hand was previously open, so something is being picked up/dropped
        #
        #         #change state of curr_holding
        #         curr_holding = not curr_holding
        #         hand_open = False
        #         last_change = time.time()
        #
        #     if time.time() - last_change < TIME_CHANGE_BUFFER and sum(fingers_open) <= 2:
        #         if max_hand_discrepancy < get_distance(curr_hand_loc, (hand_mean_x, hand_mean_y)):
        #             # hand is at greatest distance from mean detected so far, detect hand position relative to fiducials
        #             #initialize closest_distance as an integer greater than size of image
        #             closest_distance = 10000
        #             max_hand_discrepancy = get_distance(curr_hand_loc, (hand_mean_x, hand_mean_y))
        #             for marker_index in range(len(aruco_vectors)):
        #                 #later replace corners with a persistent measure of fiducial location
        #                 marker = aruco_vectors[marker_index].corners
        #
        #                 # top left point of marker should be to the left and underneath the hand
        #                 if marker[0][0] < curr_hand_loc[0] and marker[0][1] > curr_hand_loc[1] \
        #                     and	get_distance(marker[0], curr_hand_loc) < closest_distance:
        #
        #                 # #this version just finds nearest fiducial marker, problematic since picker might pick towards one
        #                 #side of the bin
        #                 # if get_distance(marker[0], curr_hand_loc) < closest_distance:
        #
        #                     closest_distance = get_distance(marker[0], curr_hand_loc)
        #                     #get left center right based on marker index, ids of aruco stored as int
        #                     left_center_right = aruco_vectors[marker_index].get_id() % 10
        #                     top_center_bottom = aruco_vectors[marker_index].get_id() // 10
        #
        #
        #     # motion related stuff that is checked every fixed period (otherwise different frame refresh rates lead to
        #     # inconsistencies in measuring movement in pixels)
        #     if (time_change > REFRESH_TIME):
        #         # checks when the time_change is greater than the REFRESH TIME
        #
        #         print (f"curr_holding: {curr_holding}", handedness, end = " ")
        #         if prev_hand_loc != () and (curr_hand_loc[0] > prev_hand_loc[0] + MOVEMENT_MOE * time_change / REFRESH_TIME):
        #             # curr_hand is more than MOVEMENT_MOE pixels to the right of prev_hand_loc
        #             print("Moving right", end=" ")
        #         elif prev_hand_loc != () and (curr_hand_loc[0] < prev_hand_loc[0] - MOVEMENT_MOE * time_change / REFRESH_TIME):
        #             print("Moving left", end=" ")
        #
        #         if prev_hand_loc != () and (curr_hand_loc[1] > prev_hand_loc[1] + MOVEMENT_MOE * time_change / REFRESH_TIME):
        #             # curr_hand is more than MOVEMENT_MOE pixels to the bottom of prev_hand_loc
        #             print("Moving down", end=" ")
        #         elif prev_hand_loc != () and (curr_hand_loc[1] < prev_hand_loc[1] - MOVEMENT_MOE * time_change / REFRESH_TIME):
        #             print("Moving up", end=" ")
        #
        #         print (len(picked), end = " ")
        #
        #
        #         prev_hand_loc = curr_hand_loc
        #         prev_time = curr_time
        #
        #     frames += 1

            #resize image to be a little smaller
            #image = cv2.resize(image, (int(image.shape[1] * 0.75), int(image.shape[0] * 0.75)))

        #removing distortion on image
        image = cv2.undistort(image, intrinsic, distortion, None)

        #horizontal margin to be removed from the video (one for left and one for right so total fraction removed is
        #double this proportion)
        horizontal_margin = 0

        image = image[:, int(image.shape[1] * horizontal_margin): int(image.shape[1] * (1 - horizontal_margin))]

        image = cv2.resize(image, (int(1280 * (1 - horizontal_margin * 2)),720))

        cv2.imshow('MediaPipe Hands', image)

        if args.outfile:
            out.write(image)

        aruco_fig.canvas.draw()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if aruco_sp_points:
            aruco_sp_points.remove()
        if aruco_sp_predicted_points:
            aruco_sp_predicted_points.remove()
        if aruco_hand_tvec is not None:
            aruco_hand_sp_points.remove()
        if aruco_plot_plane:
            aruco_plot_plane.remove()
        if hand_sp_points:
            hand_sp_points.remove()
    hands.close()
    cap.release()

    if args.outfile:
        out.release()

    # Closes all the frames
    cv2.destroyAllWindows()

