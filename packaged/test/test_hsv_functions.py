import numpy as np
import sys
import requests
import ast
import json

# run test hsv train first to get a model
if __name__ == '__main__':
    # if len(sys.argv) < 2:
    #     print("Usage: test_hsv_train.py")
    #     exit()
    
    url = 'http://localhost:5003/hsv_train'
    payload = {
        'id': '1234',
        'picklist_nos': str([i for i in range(136, 235)])
    }
    # headers = {'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.75 Safari/537.36'}

    request_id, hsv_avg_mean, hsv_avg_std = ast.literal_eval(requests.post(url, data=payload).content.decode())

    print (hsv_avg_mean)

    url = 'http://localhost:5005/hsv_train_iterative'
    payload = {
        'id': '1234',
        'picklist_no': 136,
        'hsv_avg_mean': json.dumps(hsv_avg_mean),
        'hsv_avg_std': json.dumps(hsv_avg_std)
    }

    request_id, hsv_avg_mean, hsv_avg_std = ast.literal_eval(requests.post(url, data=payload).content.decode())

    print (hsv_avg_mean)

    url = 'http://localhost:5004/hsv_test'
    payload = {
        'id': '1234',
        'picklist_nos': str([136]),
        'hsv_avg_mean': json.dumps(hsv_avg_mean),
        'hsv_avg_std': json.dumps(hsv_avg_std)
    }

    request_id, predictions = ast.literal_eval(requests.post(url, data=payload).content.decode())

    print (predictions)
    