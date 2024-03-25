import os
import cv2
import argparse

import sys
from pathlib import Path # if you haven't already done so
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))
from utils.hand_tracker import HandTracker
import numpy as np
import mediapipe as mp

import time

"""native MP impl stuff"""

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Create a hand landmarker instance with the video mode:
# options = HandLandmarkerOptions(
#     base_options=BaseOptions(model_asset_path='./models/hand_landmarker.task'),
#     running_mode=VisionRunningMode.VIDEO,
#     min_tracking_confidence = 0.3,
#     min_hand_detection_confidence = 0.3,
#     min_hand_presence_confidence = 0.2)

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='./models/hand_landmarker.task'),
    running_mode=VisionRunningMode.IMAGE,
    min_tracking_confidence = 1.0,
    min_hand_detection_confidence = 0.4,
    min_hand_presence_confidence = 0.4)

landmarker = None

HAND_CONNECTIONS = mp.solutions.hands_connections.HAND_CONNECTIONS

#higher mean = less blurry
#TODO: tune size param - originally 60
def detect_blur_fft(image, size=20):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    (h, w) = image.shape
    (cx, cy) = (int(w / 2.0), int(h / 2.0))
    fft = np.fft.fft2(image)
    fftShift = np.fft.fftshift(fft)

    fftShift[cy - size : cy + size, cx - size : cx + size] = 0
    fftShift = np.fft.ifftshift(fftShift)
    recon = np.fft.ifft2(fftShift)

    magnitude = np.log(np.abs(recon))
    mean = np.mean(magnitude)
    return mean

def kanjar_image_score(image):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    (h, w) = image.shape
    # (cx, cy) = (int(w / 2.0), int(h / 2.0))
    #Step 1: compute fft 
    fft = np.fft.fft2(image)
    #Step 2: FFT shift to center
    fftShift = np.fft.fftshift(fft)
    #Step 3 & 4 - get max of absolute value of fft shift
    M = np.max(np.abs(fftShift))
    #Step 5 - get num of pixels valued above threshold
    th = np.sum(fft > M / 2000)
    #Step 6 - normalize to a proportion of image size (so range is [0, 1])
    return th / (h * w)


def normalized_to_pixel(frame_shape, points):
    frame = np.array([[frame_shape[0], frame_shape[1]]])
    return np.clip((frame * points).astype(int), np.zeros((1, 2)), frame - 1).astype(int)

#width, height and (x, y)s
def in_frame(frame_shape, points):
    # if point[0] > 0 and point[0] < frame_shape[0] and point[1] > 0 and point[1] < frame_shape[1]:
    #     return True
    # else:
    #     return False 
    bounds = np.array([[0, 0], [frame_shape[0], frame_shape[1]]])
    # return np.bitwise_and(np.bitwise_and(points >= bounds[0][:, 0], points >= bounds[0][:, 1]), np.bitwise_and(points < bounds[1][:, 0], points < bounds[1][:, 1]))
    return np.bitwise_and(np.bitwise_and((points >= bounds[0])[:, 0], (points >= bounds[0])[:, 1]), np.bitwise_and((points < bounds[1])[:, 0], (points < bounds[1])[:, 1]))


#return one frame per carry sequence
def get_frame_scores(video_folder_path, picklist_name, label_path, out_path):
    data = []
    try:
        with open(os.path.join(label_path, picklist_name + ".txt"), "r") as infile:
            data = infile.readlines()
        data = [d.strip().split(' ') for d in data]
    except:
        print("Picklist " + picklist_name + " label file not found. No frame scores calculated for picklist " + picklist_name)
        return

    #get all carry sequences
    carry_seq = np.array([d[1 : ] for d in data if d[0] == 'carry_item']).astype(int)

    for i in range(carry_seq.shape[0]):
        sweep_video_mp(video_folder_path, picklist_name, out_path, carry_seq[i], seq_num = i, type = 'carry_item')

     #get all carry empty sequences
    carry_seq = np.array([d[1 : ] for d in data if d[0] == 'carry_empty']).astype(int)

    for i in range(carry_seq.shape[0]):
        sweep_video_mp(video_folder_path, picklist_name, out_path, carry_seq[i], seq_num = i, type = 'carry_empty')

def sweep_video_mp(video_folder_path, picklist_name, out_path, carry_sequence, seq_num = None, handedness = None, type = 'carry_item'):
    [start_frame, end_frame] = carry_sequence
    # vid_out_name = picklist_name + "_" + str(start_frame) + "_" + str(end_frame)
    video_path = os.path.join(video_folder_path, picklist_name + ".mp4")
    # vid_out_path = os.path.join(out_path, vid_out_name + "_bbox_mp.mp4")
    res_out_path = os.path.join(out_path, picklist_name)
    if not os.path.exists(res_out_path):
        os.mkdir(res_out_path)
    
    if seq_num is not None:
        picklist_name = picklist_name + "_" + str(seq_num)
    
    
    cap = cv2.VideoCapture(video_path)
    cap.set(1, start_frame)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_ct = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    mspf = 1000 / fps
    frame_width, frame_height = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_counter = start_frame

    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # out_writer = cv2.VideoWriter(os.path.abspath(vid_out_path), fourcc, fps, (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    
    if cap.isOpened():
        hasFrame, frame = cap.read()
    else:
        hasFrame = False

    #_, tmp, points visible, blur score?,  
    # best_frame, best_score, best_pts_visible, best_blur = None, 0, 0, float('-inf')
    # worst_frame, worst_score, worst_pts_visible, worst_blur = None, float('inf'), float('inf'), float('inf')
    
    #frame num, points visible (in frame), palm vector, normalized bbox area, blur metric, contrast metrics, overall score   
    score_metrics = np.zeros((end_frame - start_frame + 1, 7))
    score_weighting = np.asarray([0, 2, 10, 10, 3]) #mediapipe always think there's 21 points in frame...
    score_metrics[:, 0] = np.arange(start_frame, end_frame + 1)

    #TODO: should we only include frames where all 21 points are in frame?

    #set up testing for metrics
    # best_blur, best_blur_frame, best_blur_frame_num, = float('-inf'), None, -1,
    # worst_blur, worst_blur_frame, worst_blur_frame_num, = float('inf'), None, -1
    #https://en.wikipedia.org/wiki/Contrast_(vision) TODO
    # best_weber, best_weber_frame, best_weber_frame_num = float('-inf'), None, -1
    # worst_weber, worst_weber_frame, worst_weber_frame_num = float('inf'), None, -1
    # best_michelson, best_michelson_frame, best_michelson_frame_num = float('-inf'), None, -1
    # worst_michelson, worst_michelson_frame, worst_michelson_frame_num = float('inf'), None, -1
    # best_rms, best_rms_frame, best_rms_frame_num = float('-inf'), None, -1
    # worst_rms, worst_rms_frame, worst_rms_frame_num = float('inf'), None, -1

    #only calculate on eligible frames
    best_score, best_frame_num, best_frame = 0, -1, None
    worst_score, worst_frame_num, worst_frame = float('inf'), -1, None

    while hasFrame and frame_counter <= end_frame:
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

            points_in_frame = in_frame((frame_width, frame_height), image_hand_landmarks)
            score_metrics[frame_counter - start_frame, 1] = np.sum(points_in_frame / 21) #normalize to [0, 1] range instead of [0, 21]

            #world landmarks in meters using the hand geometric center as origin
            world_hand_landmarks = np.array([[world_landmarks[k].x, world_landmarks[k].y, world_landmarks[k].z] for k in range(21)])

            """Get the z component (depth) of the vector pointing out of the palm."""
            #We assume that the plane of the palm can be determined by the points at indices 0, 5, 13 (wrist, base index, base ring). 
            #Taking the correct two vectors (depending on handedness) derived from these points and doing cross product gives palm plane vector
            #Normalize result and take the z direction. Should give a result between -1 and 1
            palm_vec = None
            if pick_handedness == 'Right':
                palm_vec = np.cross(world_hand_landmarks[0] - world_hand_landmarks[5], world_hand_landmarks[13] - world_hand_landmarks[5])
            else:
                palm_vec = np.cross(world_hand_landmarks[0] - world_hand_landmarks[13], world_hand_landmarks[5] - world_hand_landmarks[13])
            palm_vec /= np.sqrt(np.sum(palm_vec ** 2))
            z_val = palm_vec[2] #calculated metric - range in [-1, 1] with 1 being good (palm up)
            score_metrics[frame_counter - start_frame, 2] = z_val

            # fig = plt.figure()
            # ax = fig.add_subplot(projection='3d')
            # ax.scatter3D(world_hand_landmarks[:, 0], world_hand_landmarks[:, 1], world_hand_landmarks[:, 2], color='b')
            # ax.scatter3D(world_hand_landmarks[(0, 5, 13), 0], world_hand_landmarks[(0, 5, 13), 1], world_hand_landmarks[(0, 5, 13), 2], color ='r')
            # ax.plot(world_hand_landmarks[(0, 5), 0], world_hand_landmarks[(0, 5), 1], world_hand_landmarks[(0, 5), 2])
            # ax.plot(world_hand_landmarks[(0, 13), 0], world_hand_landmarks[(0, 13), 1], world_hand_landmarks[(0, 13), 2])
            # ax.plot(world_hand_landmarks[(13, 5), 0], world_hand_landmarks[(13, 5), 1], world_hand_landmarks[(13, 5), 2])
            # ax.quiver3D(world_hand_landmarks[5, 0], world_hand_landmarks[5, 1], world_hand_landmarks[5, 2], world_hand_landmarks[5, 0] + palm_vec[0], world_hand_landmarks[5, 1] + palm_vec[1], world_hand_landmarks[5, 2] + palm_vec[2], length = 0.03)
            # plt.show()


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
            if hand_crop.shape[0] <= 0 or hand_crop.shape[1] <= 0:
                score_metrics[frame_counter - start_frame, 1:] = np.asarray([0, 0, 0, 0, 0, float('-inf')])
                frame_counter += 1
                hasFrame, frame = cap.read()
                continue
            """Area of hand crop normalized to whole image"""
            score_metrics[frame_counter - start_frame, 3] = (max_y - min_y) * (max_x - min_x) / (np.prod(frame.shape[:2]))

            # cv2.imshow('e', hand_crop)
            # if cv2.waitKey(-1) & 0xFF == ord('q'):
            #     pass

            """Calculate blur metric""" 
            #blur metric for hand crop and resized hand crop
            #no need to resize since it relative blur is the same
            # blur = detect_blur_fft(cv2.resize(hand_crop, (128, 128)))
            # score_metrics[frame_counter - start_frame, 3] = blur
            # print(hand_crop.shape)
            blur = kanjar_image_score(cv2.resize(hand_crop, (128, 128)))
            score_metrics[frame_counter - start_frame, 4] = blur
            # if blur < worst_blur:
            #     worst_blur_frame = hand_crop
            #     worst_blur = blur
            #     worst_blur_frame_num = frame_counter
            # if blur > best_blur:
            #     best_blur_frame = hand_crop
            #     best_blur = blur
            #     best_blur_frame_num = frame_counter

            """Calculate contrast metrics"""

            #looks like michelson gives garbage values consistently
            # #michelson
            # Y_channel = cv2.cvtColor(hand_crop, cv2.COLOR_BGR2YUV)[:,:,0]
            # # compute min and max of Y
            # min, max = np.min(Y_channel), np.max(Y_channel)
            # # compute contrast
            # michelson_contrast = (max-min)/(max+min)
            # score_metrics[frame_counter - start_frame, 4] = michelson_contrast
            # if michelson_contrast > best_michelson:
            #     best_michelson_frame = hand_crop
            #     best_michelson = michelson_contrast
            #     best_michelson_frame_num = frame_counter
            # if michelson_contrast < worst_michelson:
            #     worst_michelson_frame = hand_crop
            #     worst_michelson = michelson_contrast
            #     worst_michelson_frame_num = frame_counter

            #rms contrast
            img_grey = cv2.normalize(cv2.cvtColor(hand_crop, cv2.COLOR_BGR2GRAY), None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            rms_contrast = img_grey.std()
            score_metrics[frame_counter - start_frame, 5] = rms_contrast

            # if rms_contrast > best_rms:
            #     best_rms_frame = hand_crop
            #     best_rms = rms_contrast
            #     best_rms_frame_num = frame_counter
            # if rms_contrast < worst_rms:
            #     worst_rms_frame = hand_crop
            #     worst_rms = rms_contrast
            #     worst_rms_frame_num = frame_counter

            frame_score = np.dot(score_metrics[frame_counter - start_frame, 1:6], score_weighting)
            score_metrics[frame_counter - start_frame, 6] = frame_score
            if frame_score > best_score:
                best_score, best_frame_num, best_frame = frame_score, frame_counter, hand_crop
            if frame_score < worst_score:
                worst_score, worst_frame_num, worst_frame = frame_score, frame_counter, hand_crop
        else:
            # print("no hands detected on frame {}".format(frame_counter))
            score_metrics[frame_counter - start_frame, 6] = float('-inf') #TODO figure out a better way to handle this probably

        #move on to next frame
        frame_counter += 1
        hasFrame, frame = cap.read()

    # if best_frame is None:
    #     print("Couldn't detect any hands in the carry sequence from frames {} to {}".format(start_frame, end_frame))
    
    #process score and save things
    #TODO remove eventually
    # cv2.imwrite(os.path.join(res_out_path, picklist_name + "_best_blur_{}.jpg".format(best_blur_frame_num)), best_blur_frame)
    # cv2.imwrite(os.path.join(res_out_path, picklist_name + "_worst_blur_{}.jpg".format(worst_blur_frame_num)), worst_blur_frame)

    # cv2.imwrite(os.path.join(res_out_path, picklist_name + "_best_michelson_{}.jpg".format(best_michelson_frame_num)), best_michelson_frame)
    # cv2.imwrite(os.path.join(res_out_path, picklist_name + "_worst_michelson_{}.jpg".format(worst_michelson_frame_num)), worst_michelson_frame)
    # cv2.imwrite(os.path.join(res_out_path, picklist_name + "_best_rms_{}.jpg".format(best_rms_frame_num)), best_rms_frame)
    # cv2.imwrite(os.path.join(res_out_path, picklist_name + "_worst_rms_{}.jpg".format(worst_rms_frame_num)), worst_rms_frame)
    cv2.imwrite(os.path.join(res_out_path, picklist_name + "_" + type + "_best_frame_{}.jpg".format(best_frame_num)), best_frame)
    cv2.imwrite(os.path.join(res_out_path, picklist_name + "_" + type + "_worst_frame_{}.jpg".format(worst_frame_num)), worst_frame)
    
    # cv2.imshow('best frame', best_frame)
    # if cv2.waitKey(-1) & 0xFF == ord('q'):
    #     pass
    # cv2.imshow('worst frame', worst_frame)
    # if cv2.waitKey(-1) & 0xFF == ord('q'):
    #     pass

    np.save(os.path.join(res_out_path, picklist_name + '_' + type), score_metrics)


    # cap.release()
    # out_writer.release()
    cv2.destroyAllWindows()




if __name__ == '__main__':
    landmarker = HandLandmarker.create_from_options(options) 
    start_time = time.perf_counter()
    for i in range(136, 140):
        print(i)
        get_frame_scores('../../../thesis_dataset', 'picklist_{}'.format(i), '../../../sim_labels', '../../../thesis_results')
    end_time = time.perf_counter()
    print(end_time - start_time)
    
    #TODO 
        #find a way to rank frames based on some combination (maybe a weighted sum?) of metrics  
    
    # np.set_printoptions(suppress=True)
    # a = np.load('../../../thesis_results/picklist_275/picklist_275_0.npy')
    # print(a)
    # a = np.load('../../../thesis_results/picklist_275/picklist_275_1.npy')
    # print(a)
    # a = np.load('../../../thesis_results/picklist_275/picklist_275_2.npy')
    # print(a)