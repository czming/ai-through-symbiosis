import requests
import cv2
import time
import numpy as np
import sys
import threading
import concurrent.futures
from ast import literal_eval

if __name__ == '__main__':
    print("Started")
    url = 'http://htk_test:5000/htk_test'
    files = {
        'mlf': open("../shared/pre/labels.mlf_tri_internal", "rb").read(),
        'ext': open("../shared/pre/result.mlf_letter0", "rb").read(),
        'dict': open("../shared/pre/dict_letter2letter_ai_general", "rb").read(),
        'models': open("../shared/pre/newMacros", "rb").read(),
        'commands': open("../shared/pre/commands_letter_isolated_ai_general", "rb").read()
    }
    payload = {
        'id': '1234',
        'picklist': '1000'
    }
    headers = {'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.75 Safari/537.36'}

    if True:
        start = time.time()
        content = requests.post(url, files=files, data=payload, headers=headers)
        end = time.time()

        print(content.content, "\n", 'Response took', end-start, 'seconds.')

print("END")
    
