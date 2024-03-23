import mediapipe as mp
import cv2
import numpy as np 
import argparse
import os

from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2


BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Create a hand landmarker instance with the video mode:
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='./models/hand_landmarker.task'),
    running_mode=VisionRunningMode.VIDEO,
    min_tracking_confidence = 0.3,
    min_hand_detection_confidence = 0.2,
    min_hand_presence_confidence = 0.2)

HAND_CONNECTIONS = mp.solutions.hands_connections.HAND_CONNECTIONS


def gen_landmarks(vid_path, outvid_path = None, draw_landmarks = True, draw_live = True):

    #file_object = open("example.txt", "a")

    # landmarks = {}
    
    with HandLandmarker.create_from_options(options) as landmarker:
        cap = cv2.VideoCapture(vid_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_ct = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        mspf = 1000 / fps
        i = 0

        fourcc, out = cv2.VideoWriter_fourcc(*'mp4v'), None

        outvid_path = os.path.abspath(outvid_path)
    
        if outvid_path:
            print("Writing results to {}".format(outvid_path))
            out = cv2.VideoWriter(outvid_path, fourcc, fps, (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))


        while cap.isOpened() and i < frame_ct:
            if i % 100 == 0:
                print("Processing frame {} of {}".format(i, int(frame_ct)))

            ret, frame = cap.read()
            # if frame is read correctly ret is True
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break

            image = mp.Image(image_format = mp.ImageFormat.SRGB, data = frame)


            detection_res = landmarker.detect_for_video(image, int(mspf * i))
            # num_box, joint_pos, keypoint_2d, mediapipe_wrist_rot = detector.detect(rgb)
            # print(str(detection_res))
            hands, handedness = len(detection_res.handedness), [elem[0].category_name for elem in detection_res.handedness]
            hand_landmarks_list = detection_res.hand_landmarks
            world_landmarks_list = detection_res.hand_world_landmarks
            print(hand_landmarks_list)

            # hand_landmarks_dict = {i : {'x': x, } for i in range(len(hand_landmarks_list))}

            """
            Write landmarks to object
            
            landmarks[i] = { j : {
                    'handedness': handedness[j],
                    'hand_landmarks': {k : {'x' : hand_landmarks_list[j][k].x, 'y' : hand_landmarks_list[j][k].y, 
                                            'z' : hand_landmarks_list[j][k].z} for k in range(21)}, 
                    'world_landmarks': {k : {'x' : world_landmarks_list[j][k].x, 'y' : world_landmarks_list[j][k].y, 
                                            'z' : world_landmarks_list[j][k].z} for k in range(21)}, 
                    } for j in range(hands)
            }
            """
            
            """
            Draw and display the frame if we wanted to
            """
            if draw_landmarks or outvid_path:
                annotated_image = np.copy(frame)

                for idx in range(len(hand_landmarks_list)):
                    hand_landmarks = hand_landmarks_list[idx]

                    # Draw the hand landmarks.
                    hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                    hand_landmarks_proto.landmark.extend([
                        landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
                    ])

                    solutions.drawing_utils.draw_landmarks(
                        annotated_image,
                        hand_landmarks_proto,
                        solutions.hands.HAND_CONNECTIONS,
                        solutions.drawing_styles.get_default_hand_landmarks_style(),
                        solutions.drawing_styles.get_default_hand_connections_style()
                    )
            
                if draw_live:
                    cv2.imshow('annotated image', annotated_image)
                
                if outvid_path:
                    out.write(annotated_image)

            # if cv2.waitKey(int(mspf)) & 0xFF == ord('q'):
            #     break

            i += 1

        if draw_landmarks:
            cap.release()
            cv2.destroyAllWindows()

        if outvid_path:
            out.release()
            print("Wrote to video successfully")


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--show_inference', action='store_true')
    args = arg_parser.parse_args()

    gen_landmarks('../../../thesis_dataset/picklist_275.mp4', outvid_path = '../../../thesis_dataset/picklist_275_results2.mp4', draw_live = args.show_inference)
    # gen_landmarks('./3bld_pr.mp4', '3bld_results', '3bld_annotated.mp4')
            