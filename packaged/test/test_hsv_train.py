import numpy as np
import sys
import requests


if __name__ == '__main__':
    # if len(sys.argv) < 2:
    #     print("Usage: test_hsv_train.py")
    #     exit()
    
    url = 'http://localhost:5003/hsv_train'
    payload = {
        'id': '1234',
        'picklist_nos': str([i for i in range(136, 235)])
    }
    headers = {'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.75 Safari/537.36'}

    print (requests.post(url, data=payload).content)