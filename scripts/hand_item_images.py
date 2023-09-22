import utils
import argparse
import random
import os
import cv2

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

#goes to a specific frame number in a capture and displays it.
def show_frame(cap, frame = 0):
    fps = cap.get(cv2.CAP_PROP_FPS)

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame - 1)
    ret, v_frame = cap.read()
    if ret:
        v_frame = resize_img(v_frame, 1280, 720)

        cv2.imshow(str(frame / fps) + "s", v_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        raise IndexError("Can't read frame number " + int(frame) + " from capture.")

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
            show_frame(cap, midpt)
            print(midpt, ": " + str(midpt / fps) + "s")
            
            # cv2.imwrite("path_where_to_save_image", frame)
        