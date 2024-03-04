import requests
import cv2
import time
import numpy as np
import sys
import threading
import concurrent.futures
from ast import literal_eval

# docker run --rm --name extract_features -p 5000:5000 extract_features
# docker run --rm --name preprocessing -p 5000:5001 preprocessing

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: test-video.py <filename>")
        exit()

    start = time.time()

    cap = cv2.VideoCapture(sys.argv[1])
    fcount = 0
    features = []

    while True:
        if fcount > 5:
            break
        print("Frame ", fcount)
        ret, frame = cap.read()
        if not ret:
            break
        # Load image
        image = frame
        bytes = image.tobytes
        shape = ','.join([str(i) for i in image.shape])
        url = 'http://localhost:5000/feature-extractor'
        files = {
            'image': ('unnamed.png', image),
        }
        payload = {
            'id': str(fcount),
            'shape': shape
        }
        headers = {'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.75 Safari/537.36'}
        # cv2.imshow('Frame', frame)
        content = requests.post(url, files=files, data=payload, headers=headers)

        fcount += 1

        if content.status_code == 200:
            id, feature, runtime = literal_eval(content.content.decode())
            features.append(feature)
        
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    features = np.array(features)

    shape = ','.join([str(i) for i in features.shape])
    url = 'http://localhost:5001/preprocessing'
    files = {
        'data': ('data.npy', features.tobytes())
    }
    payload = {
        'id': str(0),
        'shape': shape
    }
    headers = {'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.75 Safari/537.36'}
    content = requests.post(url, files=files, data=payload, headers=headers)

    if content.status_code == 200:
        print(content.content.decode())
        id, serialized, dtype, shape, runtime = literal_eval(content.content.decode())
        output = np.frombuffer(serialized, dtype=dtype).reshape(shape)

    end = time.time()
    print(output.shape)
    print(output, "\n", 'Response took', end-start, 'seconds.')

print("END")
    