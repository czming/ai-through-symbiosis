import cv2
import mediapipe as mp
import skimage.color
import numpy as np
import os


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
                hues.append(hsv_value[0][0])
                saturations.append(hsv_value[0][1])
    hist_n, hist_bins = np.histogram(hues, bins=10, range=(0, 1))
    sat_n, sat_bins = np.histogram(saturations, bins=10, range=(0, 1))

    histsat = np.concatenate((hist_n, sat_n)) / number_of_pixels
    mystring = str(histsat).replace("[", "").replace("]", "").replace("\n", "")
    newstring = ' '.join(mystring.split())
    return histsat

picklist = "picklist_2"
picklist_nums = [11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40]
# picklist_nums = list(range(4,41))
for picklist_num in picklist_nums:
    print(picklist_num)
    mphands = mp.solutions.hands
    hands = mphands.Hands()
    mp_drawing = mp.solutions.drawing_utils
    picklist_filename = "picklist_" + str(picklist_num) + ".MP4"
    video_path = os.path.join("/Users/jonathanwomack/projects/ai-through-symbiosis/github-repo/videos/", picklist_filename)
    cap = cv2.VideoCapture(video_path)
    _, frame = cap.read()
    h, w, c = frame.shape
    count = 0
    while True:
        _, frame = cap.read()
        count += 1
        if frame is None:
            break
        framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(framergb)
        hand_landmarks = result.multi_hand_landmarks
        if hand_landmarks:
            for handLMs in hand_landmarks:
                x_max = 0
                y_max = 0
                x_min = w
                y_min = h
                for lm in handLMs.landmark:
                    x, y = int(lm.x * w), int(lm.y * h)
                    if x > x_max:
                        x_max = x
                    if x < x_min:
                        x_min = x
                    if y > y_max:
                        y_max = y
                    if y < y_min:
                        y_min = y
                # cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                # mp_drawing.draw_landmarks(frame, handLMs, mphands.HAND_CONNECTIONS)
            frame = frame[y_min:y_max, x_min:x_max]
            hs_vector = [str(i) for i in get_hs_bins(frame)]
            if len(hs_vector) == 0:
                hs_vector = previous_output
            previous_output = hs_vector
        
        #     with open("../experiments/show-hand-color/data/picklist_" + str(picklist_num) + ".txt-test", "a") as outfile:
        #         outfile.write(" ".join(hs_vector) + "\n")
        # else:
        #     with open("../experiments/show-hand-color/data/picklist_" + str(picklist_num) + ".txt-test", "a") as outfile:
        #         outfile.write(" ".join(previous_output) + "\n")
        
        cv2.imshow("Frame", frame)
        cv2.waitKey(1)