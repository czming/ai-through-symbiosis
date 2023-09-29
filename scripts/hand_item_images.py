import utils
import argparse
import random
import os
import cv2
# from mediapipe import HandTracker
from utils.hand_tracker import HandTracker
import numpy as np

#stolen from extract_hand
PALM_MODEL_PATH = os.path.join(os.getcwd(), "scripts", "utils", "models", "palm_detection_without_custom_op.tflite")
LANDMARK_MODEL_PATH =  os.path.join(os.getcwd(), "scripts", "utils", "models", "hand_landmark.tflite")
ANCHORS_PATH = os.path.join(os.getcwd(), "scripts", "utils", "models", "anchors.csv")
detector = HandTracker(
    "True", #True = hand, False = palm i think
    PALM_MODEL_PATH,
    LANDMARK_MODEL_PATH,
    ANCHORS_PATH,
    box_shift=0.2,
    box_enlarge=1
)

# https://stackoverflow.com/a/58126805
def resize_img(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=inter)

'''
Get a specific frame from a capture. Returns the image itself.
'''
def get_frame(cap, frame = 0):
    curr = cap.get(cv2.CAP_PROP_POS_FRAMES)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame - 1)
    ret, img = cap.read()
    cap.set(cv2.CAP_PROP_POS_FRAMES, curr - 1)
    if ret:
        return img 
    else:
        return RuntimeError("could not read frame number " + int(frame) + " from capture.")

'''
Display an image in a computer-friendly way (resize to normal screen dims)
'''
def show_frame(image, name = "", dims = [1280, 720]):
    image = resize_img(image, dims[0], dims[1])

    cv2.imshow(name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    

'''
Takes in a frame (image), show hand crop
'''
def extract_hand_region_from_frame(frame):
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    points, bbox = detector(image)
    if points is not None:
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
        
        hand_item_crop = frame[int(min_y): int(max_y), int(min_x): int(max_x)]
        # return hand_item_crop
        # print(bbox)

        frame = cv2.line(frame, (int(bbox[0][0]), int(bbox[0][1])), (int(bbox[1][0]), int(bbox[1][1])), (255,0,0), 3)
        frame = cv2.line(frame, (int(bbox[1][0]), int(bbox[1][1])), (int(bbox[2][0]), int(bbox[2][1])), (255,0,0), 3)
        frame = cv2.line(frame, (int(bbox[2][0]), int(bbox[2][1])), (int(bbox[3][0]), int(bbox[3][1])), (255,0,0), 3)
        frame = cv2.line(frame, (int(bbox[3][0]), int(bbox[3][1])), (int(bbox[0][0]), int(bbox[0][1])), (255,0,0), 3) 
        return frame

        # rect = cv2.minAreaRect(bbox)
        # src_pts = cv2.boxPoints(rect)
        # width = int(rect[1][0])
        # height = int(rect[1][1])
        # dst_pts = np.array([[0, height-1],
        #             [0, 0],
        #             [width-1, 0],
        #             [width-1, height-1]], dtype="float32")
        # M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        # warped = cv2.warpPerspective(frame, M, (width, height))
        # return warped


        
        
    return None

if __name__ == "__main__":

    DEFAULT_VIDEOS_FILE_PATH = 'C:\\Users\\calix\\Desktop\\GT\\aits_temp\\aits_videos'
    VIDEO_FILE_PREFIX = 'picklist_'

    DEFAULT_HTK_OUTPUT_FILE_PATH = 'C:\\Users\\calix\\Desktop\\GT\\aits_temp\\htk_output'
    HTK_OUTPUT_FILE_PREFIX = 'results-'

    DEFAULT_START_PICKLIST = 41
    DEFAULT_END_PICKLIST = 50
    # DEFAULT_END_PICKLIST = 90

    parser = argparse.ArgumentParser(
                    prog='Hand Item Script')
    
    parser.add_argument('--videopath', default = DEFAULT_VIDEOS_FILE_PATH) 
    parser.add_argument('--htkpath', default = DEFAULT_HTK_OUTPUT_FILE_PATH)
    parser.add_argument('--videoprefix', default = VIDEO_FILE_PREFIX)
    parser.add_argument('--htkprefix', default = HTK_OUTPUT_FILE_PREFIX)
    parser.add_argument('-s', '--start', default = DEFAULT_START_PICKLIST)
    parser.add_argument('-e', '--end', default = DEFAULT_END_PICKLIST)

    args = parser.parse_args()

    for i in range(args.start, args.end):
        print("Picklist " + str(i))
        video_full_path= os.path.join(args.videopath, args.videoprefix + str(i) + ".mp4")
        htk_output_full_path = os.path.join(args.htkpath, args.htkprefix + str(i))

        # https://stackoverflow.com/a/38368198
        cap = cv2.VideoCapture(video_full_path)
        # print(cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        # frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        # print(frame_count)

        # total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        

        #find each carry item sequence - in the aeim grammar, it's 'e'
        boundaries = utils.get_htk_boundaries(htk_output_full_path, fps = fps)

        #[[start0, end0], [start1, end1], ...]
        carry_sequences_frames = [[int(boundaries['e'][i] * fps), int(boundaries['e'][i + 1] * fps)] for i in range(0, int(len(boundaries['e'])/2), 2)]
        # print(carry_sequences_frames)

        #display time stamp in video (print first)
        for start, end in carry_sequences_frames:
            #show the middle image (print time stamp)
            midpt = int((start + end) / 2)
            image = get_frame(cap, midpt)

            show_frame(image, str((midpt / fps)) + "s")

            hand_region_image = extract_hand_region_from_frame(image)
            show_frame(hand_region_image, "hand region crop", [480, 480])

            print(midpt, ": " + str(midpt / fps) + "s")

            
            # cv2.imwrite("path_where_to_save_image", frame)
        