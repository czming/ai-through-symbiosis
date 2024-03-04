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
        print("Usage: test-fe.py <filename>")
        exit()

    image = cv2.imread(sys.argv[1])
    bytes = image.tobytes
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

    if len(sys.argv) > 2:
        if sys.argv[2] == 'agg':
            if len(sys.argv) == 4:
                num = int(sys.argv[3])
            else:
                num = 60 * 10
            times = []
            for i in range(num):
                print("Request", i, end='                 \r')
                start = time.time()
                requests.post(url, files=files, data=payload)
                times += [time.time() - start]
            print()
            times = np.array(times)
            print("Response time for", num, "images (",num // 60,"s @ 60FPS) per request (in seconds).\n")
            print("Mean:", times.mean())
            print("Max:", times.max())
            print("Min:", times.min())
            print("Std Dev:", times.std())
        if sys.argv[2] == 'multi':
            if len(sys.argv) == 4:
                num = int(sys.argv[3])
            else:
                num = 60 * 10
            times = []
            threads = []
            lock = threading.Lock()
            start = time.time()
            def make_request():
                res = requests.post(url, files=files, data=payload, headers=headers)
                
                if res.status_code == 200:
                    with lock:
                        times.append(float(literal_eval(res.content.decode())[2]))
            with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
                # Submit tasks to the executor
                futures = [executor.submit(make_request) for _ in range(num)]

                # Wait for all tasks to complete
                concurrent.futures.wait(futures)
            end = time.time()
            times = np.array(times)
            print("Response (server) time for", num, "images (", num // 60, "s @ 60FPS) per request (in seconds). Client took ", end - start,"s to complete requests.\n")
            print("Mean:", times.mean())
            print("Max:", times.max())
            print("Min:", times.min())
            print("Std Dev:", times.std())
            print("Count: ", len(times))
    else:
        start = time.time()
        content = requests.post(url, files=files, data=payload, headers=headers)
        end = time.time()
        print(content.content, "\n", 'Response took', end-start, 'seconds.')

print("END")
    