import os
import cv2
import argparse
from src.hand_tracker import HandTracker

# USAGE: python extract_hand.py --3d [true/false]
ap = argparse.ArgumentParser()
ap.add_argument("--3d", required=True,
	help="Check for type of detection")
args = vars(ap.parse_args())

WINDOW = "Hand Tracking"
PALM_MODEL_PATH = "models/palm_detection_without_custom_op.tflite"
LANDMARK_MODEL_PATH = "models/hand_landmark.tflite"
ANCHORS_PATH = "models/anchors.csv"

POINT_COLOR = (0, 255, 0)
CONNECTION_COLOR = (255, 0, 0)
THICKNESS = 4

cv2.namedWindow(WINDOW)
#Use Camera
#capture = cv2.VideoCapture(0)
video_name = "picklist_17"
dataset_path = '../2022Spring/dataset/'

capture = cv2.VideoCapture(dataset_path + video_name + '.MP4')
#green2
#Use Video
fps = capture.get(cv2.CAP_PROP_FPS)      # OpenCV2 version 2 used "CV_CAP_PROP_FPS"
print(fps)
frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
print(frame_count)

duration = frame_count/fps
print(duration*1000)

if capture.isOpened():
    hasFrame, frame = capture.read()
else:
    hasFrame = False

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

hand_3d = args["3d"]

detector = HandTracker(
    hand_3d,
    PALM_MODEL_PATH,
    LANDMARK_MODEL_PATH,
    ANCHORS_PATH,
    box_shift=0.2,
    box_enlarge=1
)
count = 0
path = os.path.join(dataset_path, video_name)
if not os.path.exists(path):
    os.mkdir(path)
while hasFrame:
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    points, bbox = detector(image)
    print(count)
    if points is not None:
        if hand_3d == "True":
            min_x = 100000
            min_y = 100000
            max_x = 0
            max_y = 0
            for point in [points[i] for i in [0,1,5,9,13,17]]:
                x, y = point
                if x < min_x:
                    min_x = x
                if y < min_y:
                    min_y = y
                if x > max_x:
                    max_x = x
                if y > max_y:
                    max_y = y
            #     cv2.circle(frame, (int(x), int(y)), THICKNESS * 2, POINT_COLOR, THICKNESS)
            # cv2.rectangle(frame, (int(min_x), int(min_y)), (int(max_x),int(max_y)), POINT_COLOR, THICKNESS)
            frame = frame[int(min_y): int(max_y), int(min_x): int(max_x)]

            # if len(frame) < 1000 and len(frame) > 70:
            cv2.imshow(WINDOW, frame)

            cv2.imwrite(path + "/" + str(count) + ".jpg", frame)
            key = cv2.waitKey(1)
            if key == 27:
                break
            for connection in connections:
                x0, y0 = points[connection[0]]
                x1, y1 = points[connection[1]]
                cv2.line(frame, (int(x0), int(y0)), (int(x1), int(y1)), CONNECTION_COLOR, THICKNESS)
        else:
            cv2.line(frame, (int(bbox[0][0]), int(bbox[0][1])), (int(bbox[1][0]), int(bbox[1][1])), CONNECTION_COLOR, THICKNESS)
            cv2.line(frame, (int(bbox[1][0]), int(bbox[1][1])), (int(bbox[2][0]), int(bbox[2][1])), CONNECTION_COLOR, THICKNESS)
            cv2.line(frame, (int(bbox[2][0]), int(bbox[2][1])), (int(bbox[3][0]), int(bbox[3][1])), CONNECTION_COLOR, THICKNESS)
            cv2.line(frame, (int(bbox[3][0]), int(bbox[3][1])), (int(bbox[0][0]), int(bbox[0][1])), CONNECTION_COLOR, THICKNESS)
    count += 5  # i.e. at 30 fps, this advances one second
    capture.set(1, count)
    hasFrame, frame = capture.read()


capture.release()
cv2.destroyAllWindows()
