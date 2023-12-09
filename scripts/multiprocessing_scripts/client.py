import cv2
from mlsocket import MLSocket
import numpy as np

HOST = "127.0.0.1"
PORT = 48293

if __name__ == "__main__":
    with MLSocket() as socket:
        socket.connect((HOST, PORT))
        image = cv2.imread("image.jpg")
        for i in range(500):
            socket.send(image)
        socket.send(np.zeros((1080, 1920, 3)))