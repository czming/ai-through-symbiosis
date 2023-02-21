import math
import numpy as np

def get_avg_hsv_bin_frames(hsv_inputs, start_frame, end_frame):
    # get the average hsv bin in the period and return the number of frames where the hand was present (hand not present
    # would lead to hsv bins being 0)

    # look only at hue bins (first 10 bins out of hsv bin length 20 vector)
    hsv_bin_sum = np.array([0 for i in range(10)]).astype(float)
    frame_count = 0

    print(np.asarray(hsv_inputs).shape)
    print(start_frame)
    print(end_frame)
    for j in range(start_frame, end_frame):
        # see whether the hand was detected (if hand was not detected, all bins would be 0)
        hand_detected = False
        for k in range(len(hsv_bin_sum)):
            # sum up the current values
            # k + 72 in both instances when looking at original
            hand_detected = hand_detected or float(hsv_inputs[j][k + 72]) != 0
            hsv_bin_sum[k] += float(hsv_inputs[j][k + 72])
        frame_count += hand_detected
    return hsv_bin_sum / frame_count, frame_count

def avg_hsv_bins(hsv_inputs, elan_boundaries, action_label):
    # sums up the hsv bins in the frames that are demarcated by the action label in elan_boundaries
    frame_count = 0
    hsv_bin_sum = np.array([0 for i in range(10)]).astype(float)
    for i in range(0, len(elan_boundaries[action_label]), 2):
        # collect the red frames
        start_frame = math.ceil(float(elan_boundaries[action_label][i]) * 59.97)
        end_frame = math.ceil(float(elan_boundaries[action_label][i + 1]) * 59.97)

        avg_hsv_bin, curr_frame_count = get_avg_hsv_bin_frames(hsv_inputs, start_frame, end_frame)

        # need the sum across all of the frames so this helps to weight by number of frames
        hsv_bin_sum += (avg_hsv_bin * curr_frame_count)
        frame_count += curr_frame_count


    if frame_count == 0:
        # no detected frames
        return hsv_bin_sum

    for k in range(len(hsv_bin_sum)):
        hsv_bin_sum[k] = hsv_bin_sum[k] / frame_count

    return hsv_bin_sum
