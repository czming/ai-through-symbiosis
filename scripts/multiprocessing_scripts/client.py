import cv2
from mlsocket import MLSocket
import numpy as np
import argparse

HOST = "127.0.0.1"
PORT = 48294

ORIGINAL_FRAME_WIDTH = 1920
ORIGINAL_FRAME_HEIGHT = 1080

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # get the video file to read the images from
    parser.add_argument("--video", "-v", type=str, help="Path to input video", required=True)

    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)

    with MLSocket() as socket:
        socket.connect((HOST, PORT))

        while cap.isOpened():
            success, image = cap.read()
            # image = cv2.imread("image.jpg")

            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                break

            image = cv2.resize(image, (ORIGINAL_FRAME_WIDTH, ORIGINAL_FRAME_HEIGHT))

            socket.send(image)
        # send final empty image
        socket.send(np.zeros((1080, 1920, 3)))