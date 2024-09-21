# to subsample videos to 30 fps

import cv2

file_name = "picklist_142"

VIDEO_FILE = f"C:/Users/chngz/OneDrive/Georgia Tech/AI Through Symbiosis/pick_list_dataset/Videos/{file_name}.MP4"


cap = cv2.VideoCapture(VIDEO_FILE)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(f'{file_name}.mp4', cv2.VideoWriter_fourcc("H", "2", "6", "4"), 29.97, (int(cap.get(3)), int(cap.get(4))))

subsample_rate = 2

count = 0

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        # If loading a video, use 'break' instead of 'continue'.
        break

    count += 1

    if count % subsample_rate == 0:
        # skip this frame
        continue

    print (count)

    out.write(image)

    # cv2.imshow("image", image)
    cv2.waitKey(1)

out.release()
