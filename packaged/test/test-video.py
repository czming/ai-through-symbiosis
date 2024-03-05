import requests
import cv2
import time
import numpy as np
import sys
import threading
import concurrent.futures
from ast import literal_eval
import signal
import os
from pathlib import Path
import tempfile
from multiprocessing import Process, Queue, Manager, Pool

# docker run --rm --name extract_features -p 5000:5000 extract_features
# docker run --rm --name preprocessing -p 5000:5001 preprocessing

stop_flag = False


def consumer(fcount, image):
    while True:
        print("Consumer", fcount)
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

        if content.status_code == 200:
            id, feature, runtime = literal_eval(content.content.decode())
            return (id, feature)
        else:
            print("Error", content.status_code)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: test-video.py <filename>")
        exit()

    pool = Pool()
    cap = cv2.VideoCapture(sys.argv[1])

    fcount = 0
    features = []

    while True:
        print("Frame ", fcount)
        ret, frame = cap.read()
        if not ret:
            stop_flag = True
            break
        # Load image
        # features.append(consumer(fcount, frame))
        pool.apply_async(consumer, args=(fcount, frame), callback=lambda x: features.append(x))
        fcount += 1
    
    pool.close()
    pool.join()

    features = sorted(features, key=lambda x: x[0])
    features = [f[1] for f in features]
    # print(features)
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
        # print(content.content.decode())
        id, serialized, dtype, shape, runtime = literal_eval(content.content.decode())
        output = np.frombuffer(serialized, dtype=dtype).reshape(shape)

    print(output.shape)

    tmpdir = "../tmp"
    Path.mkdir(Path("../tmp/data"), exist_ok=True, parents=True, mode=0o777)

    np.savetxt(os.path.join(tmpdir, "data/output1"), output, delimiter=" ")
    np.savetxt(os.path.join(tmpdir, "data/output2"), output, delimiter=" ")
    np.savetxt(os.path.join(tmpdir, "data/output3"), output, delimiter=" ")

    url = 'http://localhost:5002/htk'
    payload = {
        'id': str(0),
        '_dir': "/tmp",
        'num_folds': 1,
        'split_ratio': 0.7
    }
    headers = {'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.75 Safari/537.36'}
    content = requests.post(url, files=files, data=payload, headers=headers)

    if content.status_code == 200:
        # print(content.content.decode())
        errorcode = literal_eval(content.content.decode())
        print(errorcode)




    


print("END")
    
