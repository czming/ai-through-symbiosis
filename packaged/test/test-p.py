import requests
import cv2
import time
import numpy as np
import sys
import threading
import concurrent.futures
from ast import literal_eval


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: test-p.py <filename>")


    vid = cv2.VideoCapture(sys.argv[1])

    while True:
        ret, frame = vid.read()
        if not ret:
            break

        shape = ','.join([str(i) for i in image.shape])
        url = 'http://localhost:5000/feature-extractor'
        files = {
            'image': ('unnamed.png', image),
        }
        payload = {
            'id': '1234',
            'shape': shape
        }
        headers = {'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.75 Safari/537.36'}

        requests.post(url, files=files, data=payload)
