import cv2
import mediapipe as mp
import numpy as np
from score_video import normalized_to_pixel

def parse_action_label_file(path_to_file):
    data = []
    with open(path_to_file, 'r') as infile:
        data = [line.strip().split(' ') for line in infile.readlines()]
        infile.close()
    keys = ['pick', 'carry_item', 'place', 'carry_empty', 'empty']
    data_dict = dict()
    for key in keys:
        data_dict[key] = []
    for line in data:
        data_dict[line[0]].append([int(line[1]), int(line[2])])

    return data_dict


def get_hand_crop_from_frame(vid_path, frame_num):
    cap = cv2.VideoCapture(vid_path)
    frame_width, frame_height = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    hasFrame, frame = False, None

    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path='./models/hand_landmarker.task'),
        running_mode=VisionRunningMode.IMAGE,
        min_tracking_confidence = 1.0,
        min_hand_detection_confidence = 0.4,
        min_hand_presence_confidence = 0.4)

    landmarker = HandLandmarker.create_from_options(options) 
    handedness = None

    HAND_CONNECTIONS = mp.solutions.hands_connections.HAND_CONNECTIONS

    cap.set(1, frame_num)
    hasFrame, frame = cap.read()

    image = mp.Image(image_format = mp.ImageFormat.SRGB, data = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    # detection_res = landmarker.detect_for_video(image, int(mspf * frame_counter))
    detection_res = landmarker.detect(image)


    #instantiate metrics ig
    points_in_frame = 0

    #get hand landmarks in frame
    hand_landmarks_list = detection_res.hand_landmarks
    world_hand_landmarks_list = detection_res.hand_world_landmarks
    
    #if we detected any hands:
    if hand_landmarks_list is not None and len(hand_landmarks_list) > 0:
        #only care about first hand
        hand_landmarks = hand_landmarks_list[0] 
        world_landmarks = world_hand_landmarks_list[0]
        pick_handedness = handedness

        #handedness matters for calculating palm vector - empirically mp will get it around half of the time when only part of the hand is in frame
        if handedness is None:
            pick_handedness = detection_res.handedness[0][0].category_name

        normalized_hand_landmarks = np.array([[hand_landmarks[k].x, hand_landmarks[k].y] for k in range(21)])
        image_hand_landmarks = normalized_to_pixel((frame_width, frame_height), normalized_hand_landmarks)

        # points_in_frame = in_frame((frame_width, frame_height), image_hand_landmarks)
        # score_metrics[frame_counter - start_frame, 1] = np.sum(points_in_frame / 21) #normalize to [0, 1] range instead of [0, 21]

        #world landmarks in meters using the hand geometric center as origin
        world_hand_landmarks = np.array([[world_landmarks[k].x, world_landmarks[k].y, world_landmarks[k].z] for k in range(21)])

        """Perform transformation to orient the hand in frame"""
        #get rotated bbox
        orig = image_hand_landmarks[9][::-1] #base of middle finger

        #we want the vector from orig to the wrist to be +y, or 90 deg
        pos_y = (image_hand_landmarks[0][::-1] - orig)
        theta = np.arctan2(pos_y[0], pos_y[1]) * 180 / np.pi
        center = tuple(orig.astype(np.float32))[::-1]
        rot_mat = cv2.getRotationMatrix2D(center, theta - 90, 1)
        # rotated_image = cv2.warpAffine(src = frame, M = rot_mat) 

        size_reverse = np.array(frame.shape[1::-1]) # swap x with y
        MM = np.absolute(rot_mat[:,:2])
        size_new = MM @ size_reverse
        rot_mat[:,-1] += (size_new - size_reverse) / 2.
        new_size = tuple(size_new.astype(int))
        rotated_frame = cv2.warpAffine(frame, rot_mat, new_size)
        rotated_keypoints = (rot_mat @ np.concatenate((image_hand_landmarks, np.ones((image_hand_landmarks.shape[0], 1))), axis = 1).T).T.astype(np.int32)
        
        """Testing to make sure transformation done correctly"""
        # for connection in HAND_CONNECTIONS:
        #     cv2.line(rotated_frame, tuple(rotated_keypoints[connection[0]].astype(np.int32)), tuple(rotated_keypoints[connection[1]].astype(np.int32)), color = (0, 255, 0), thickness = 3)
        # cv2.imshow('tmp', cv2.resize(rotated_frame, (960, 540)))
        # if cv2.waitKey(-1) & 0xFF == ord('q'):
        #     pass

        """Get tight hand bbox as a cropped image"""
        #bbox in rotated frame is min area bbox 
        #TODO: consider expanding the bbox as this cuts off the hand sometimes
        min_x, min_y, max_x, max_y = np.min(rotated_keypoints[:, 0]), np.min(rotated_keypoints[:, 1]), np.max(rotated_keypoints[:, 0]), np.max(rotated_keypoints[:, 1])

        hand_crop = rotated_frame[min_y : max_y, min_x : max_x]

        #get hand crop HSV bins?
        return hand_crop

    #If no hand crop detected, return None    
    return None

def get_avg_hsv_bins_video(vid_path, frame_list):
    cap = cv2.VideoCapture(vid_path)
    frame_width, frame_height = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    hasFrame, frame = False, None

    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path='./models/hand_landmarker.task'),
        running_mode=VisionRunningMode.IMAGE,
        min_tracking_confidence = 1.0,
        min_hand_detection_confidence = 0.4,
        min_hand_presence_confidence = 0.4)

    landmarker = HandLandmarker.create_from_options(options) 
    handedness = None

    HAND_CONNECTIONS = mp.solutions.hands_connections.HAND_CONNECTIONS

    for frame_num in frame_list:
        cap.set(1, frame_num)
        hasFrame, frame = cap.read()
   
        image = mp.Image(image_format = mp.ImageFormat.SRGB, data = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
       
        # detection_res = landmarker.detect_for_video(image, int(mspf * frame_counter))
        detection_res = landmarker.detect(image)


        #instantiate metrics ig
        points_in_frame = 0

        #get hand landmarks in frame
        hand_landmarks_list = detection_res.hand_landmarks
        world_hand_landmarks_list = detection_res.hand_world_landmarks
        
        #if we detected any hands:
        if hand_landmarks_list is not None and len(hand_landmarks_list) > 0:
            #only care about first hand
            hand_landmarks = hand_landmarks_list[0] 
            world_landmarks = world_hand_landmarks_list[0]
            pick_handedness = handedness

            #handedness matters for calculating palm vector - empirically mp will get it around half of the time when only part of the hand is in frame
            if handedness is None:
                pick_handedness = detection_res.handedness[0][0].category_name

            normalized_hand_landmarks = np.array([[hand_landmarks[k].x, hand_landmarks[k].y] for k in range(21)])
            image_hand_landmarks = normalized_to_pixel((frame_width, frame_height), normalized_hand_landmarks)

            # points_in_frame = in_frame((frame_width, frame_height), image_hand_landmarks)
            # score_metrics[frame_counter - start_frame, 1] = np.sum(points_in_frame / 21) #normalize to [0, 1] range instead of [0, 21]

            #world landmarks in meters using the hand geometric center as origin
            world_hand_landmarks = np.array([[world_landmarks[k].x, world_landmarks[k].y, world_landmarks[k].z] for k in range(21)])

            """Perform transformation to orient the hand in frame"""
            #get rotated bbox
            orig = image_hand_landmarks[9][::-1] #base of middle finger

            #we want the vector from orig to the wrist to be +y, or 90 deg
            pos_y = (image_hand_landmarks[0][::-1] - orig)
            theta = np.arctan2(pos_y[0], pos_y[1]) * 180 / np.pi
            center = tuple(orig.astype(np.float32))[::-1]
            rot_mat = cv2.getRotationMatrix2D(center, theta - 90, 1)
            # rotated_image = cv2.warpAffine(src = frame, M = rot_mat) 

            size_reverse = np.array(frame.shape[1::-1]) # swap x with y
            MM = np.absolute(rot_mat[:,:2])
            size_new = MM @ size_reverse
            rot_mat[:,-1] += (size_new - size_reverse) / 2.
            new_size = tuple(size_new.astype(int))
            rotated_frame = cv2.warpAffine(frame, rot_mat, new_size)
            rotated_keypoints = (rot_mat @ np.concatenate((image_hand_landmarks, np.ones((image_hand_landmarks.shape[0], 1))), axis = 1).T).T.astype(np.int32)
            
            """Testing to make sure transformation done correctly"""
            # for connection in HAND_CONNECTIONS:
            #     cv2.line(rotated_frame, tuple(rotated_keypoints[connection[0]].astype(np.int32)), tuple(rotated_keypoints[connection[1]].astype(np.int32)), color = (0, 255, 0), thickness = 3)
            # cv2.imshow('tmp', cv2.resize(rotated_frame, (960, 540)))
            # if cv2.waitKey(-1) & 0xFF == ord('q'):
            #     pass

            """Get tight hand bbox as a cropped image"""
            #bbox in rotated frame is min area bbox 
            #TODO: consider expanding the bbox as this cuts off the hand sometimes
            min_x, min_y, max_x, max_y = np.min(rotated_keypoints[:, 0]), np.min(rotated_keypoints[:, 1]), np.max(rotated_keypoints[:, 0]), np.max(rotated_keypoints[:, 1])

            hand_crop = rotated_frame[min_y : max_y, min_x : max_x]

            #get hand crop HSV bins?

            print(frame_num)
            cv2.imshow('e', hand_crop)
            if cv2.waitKey(-1) & 0xFF == ord('q'):
                continue