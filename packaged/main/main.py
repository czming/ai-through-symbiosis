import requests
import cv2
import time
import numpy as np
import sys
from ast import literal_eval
from requests_toolbelt import MultipartDecoder
import concurrent.futures
import os
import json

stop_flag = False
Global_Headers = {'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.75 Safari/537.36'}

class Recorder:
    def _time(self):
        import time
        return str(int(time.time() * 1000.0))
    def __init__(self):
        self.id = self._time()
        self.fd = open(f"/shared/logs/{self.id}.log", "a")
    
    def record(self, event):
        self.fd.write(f"{self._time()}: {event}\n")
        self.fd.flush()
    def end(self):
        self.fd.close()

gl_rec = Recorder()

class ParallelBufferManager:
    def __init__(self, workers, name="Parallel Buffer Manager"):
        self.lookup_map = dict()
        self.counter = 0
        self.name = name
        self.workers = workers
        self.pool = concurrent.futures.ThreadPoolExecutor(max_workers=workers)

    def construct_request(self, data):
        raise NotImplementedError("Have not implemented request construction")

    def process_response(self, result):
        raise NotImplementedError("Have not implemented response processing")

    def flush(self, callback):
        print("Flushing in", self.name)
        self.pool.shutdown(wait=True)
        self.pool = concurrent.futures.ThreadPoolExecutor(max_workers=self.workers)
        print("(flush) rebuilt pool in", self.name)
        if callback and 0 < len(self.lookup_map.keys()):
            callback(self.lookup_map)
        self.lookup_map = dict()
        self.counter = 0

    def push(self, data):
        print("Pushing in", self.name)
        self.pool.submit(
                lambda: self.construct_request(data, self.counter)
        ).add_done_callback(
                lambda future: self.process_response(future.result())
        )
        self.counter += 1


class FrameFeatureManager(ParallelBufferManager):
    def __init__(self):
        super().__init__(100, "Frame Feature Manager")
    
    def construct_request(self, data, counter):
        url = "http://ef:5000/feature-extractor"
        files = {
            'image': ('unnamed.png', data)
        }
        payload = {
            'id': str(counter),
            'shape': ','.join(str(i) for i in data.shape)
        }
        gl_rec.record(f"Start Frame Feature {counter} {data.shape}")
        ret = requests.post(url, files=files, data=payload, headers=Global_Headers)
        gl_rec.record(f"End Frame Feature {counter}")
        return ret

    def process_response(self, result):
        if result.status_code != 200:
            print("Errored out on frame")
            return
        _id, vector, runtime = literal_eval(result.content.decode())
        self.lookup_map[_id] = vector
    
class PreprocessorManager(ParallelBufferManager):
    def __init__(self):
        super().__init__(3, "Preprocessor Manager")

    def construct_request(self, feature_vector, counter):
        url = 'http://preprocessing:5000/preprocessing'
        files = {
            'data': ('data.npy', feature_vector.tobytes())
        }
        payload = {
            'id': str(counter),
            'shape': ','.join(str(i) for i in feature_vector.shape)
        }
        gl_rec.record(f"Start Preprocess {counter}")
        ret = requests.post(url, files=files, data=payload, headers=Global_Headers)
        gl_rec.record(f"End Preprocess {counter}")
        return ret

    def process_response(self, result):
        if result.status_code != 200:
            print("Errored on picklist!")
            return
        _id, serialized, dtype, shape, runtime = literal_eval(result.content.decode())
        self.lookup_map[_id] = np.frombuffer(serialized, dtype=dtype).reshape(shape)

def htk_inferencer(pool, picklist, hsv, counter):
    global curr_htk_model
    print("HTK Inference Started")
    if curr_htk_model == dict():
        print("No HTK Model Trained. Skipping Inference")
        return
    url = "http://htk_test:5000/htk_test"
    payload = {
        'id': str(picklist),
        'picklist': str(picklist)
    }
    headers = {'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.75 Safari/537.36'}
    files = {
            i: curr_htk_model[i] for i in curr_htk_model if i in ['mlf', 'ext', 'models', 'commands', 'dict']
    }
    def on_htk_test(result):
        if result.status_code != 200:
            print("HTK Inference Errored")
            return
        gl_rec.record(f"End HTK Inference {counter}")
        data = MultipartDecoder.from_response(result)
        _id = None
        result = None
        for i in data.parts:
            if i.headers[b'Content-Id'] == b'id':
                _id = i.text
            if i.headers[b'Content-Id'] == b'result':
                result = i.content
        with open(f"/shared/htk_outputs/results-{_id}", "wb") as f:
                f.write(result)
        hsv.test(_id)
        print("HTK Inference Complete")
    gl_rec.record(f"Start HTK Inference {counter}")
    pool.submit(
        lambda: requests.post(url, data=payload, headers=headers, files=files)
    ).add_done_callback(
        lambda future: on_htk_test(future.result())
    )
class HTKManager:
    def __init__(self, hsv):
        self.picklist_counter = 1000
        self.pool = concurrent.futures.ThreadPoolExecutor(max_workers=10)
        self.hsv = hsv

    def add_picklist(self, picklist_data):
        print("Preproecssing Done")
        init_counter = self.picklist_counter
        counter = 0
        for i in picklist_data:
            data = picklist_data[i][:, 0]
            with open(f"/shared/data/picklist_{self.picklist_counter}", "w") as f:
                f.write("\n".join(str(i) for i in data))
                htk_inferencer(self.pool, self.picklist_counter, self.hsv, counter)
                counter += 1
                self.picklist_counter += 1
        print("HTK Data Written. Wrote", self.picklist_counter - init_counter, "picklists")

class HSVManager:
    def __init__(self):
        self.counter = 0
        self.pool = concurrent.futures.ThreadPoolExecutor(max_workers=60)
        self.model = dict()
        self.inters = dict()

    def train(self, added_list=1000):
        print("HSV Train Started")
        url = 'http://hsv_train:5000/hsv_train'
        payload = {
            'id': 'test',
            'picklist_nos': str([*list(range(136, 235)), *list(range(1000,added_list))])
        }

        def on_hsv_train(result):
            if result.status_code != 200:
                print("HSV Train Errored")
                return 
            _, m, s = literal_eval(result.content.decode())
            self.model = dict(hsv_avg_mean=m, hsv_avg_std=s)
            print (f"self.model={self.model}")
            self.inters = dict()
            print("HSV Train Complete")
        gl_rec.record("Start HSV Train")
        on_hsv_train(requests.post(url, data=payload))
        gl_rec.record("End HSV Train")
        #self.pool.submit(lambda: requests.post(url, data=payload)).add_done_callback(lambda future: on_hsv_train(future.result()))

    def train_iterative(self, picklist, predicted_labels):
        print("HSV Train Iterative Started")
        if len(self.model.keys()) == 0:
            print("HSV Model needs to be trained first, before training iterative")
            return
        url = 'http://hsv_train_iterative:5000/hsv_train_iterative'
        payload = dict(
            id=picklist,
            picklist_no=picklist,
            hsv_avg_mean=json.dumps(self.model['hsv_avg_mean']),
            hsv_avg_std=json.dumps(self.model['hsv_avg_std']),
            predicted_labels=predicted_labels
        )
        def on_hsv_train_iterative(result):
            if result.status_code != 200:
                print("HSV Train Iterative Errored")
                return
            picklist, m, s = literal_eval(result.content.decode())
            self.inters[picklist] = dict(hsv_avg_mean=m, hsv_avg_std=s)
            print("HSV Train Iterative Complete")
        gl_rec.record("Start HSV Iterative Train")
        on_hsv_train_iterative(requests.post(url, data=payload))
        gl_rec.record("End HSV Iterative Train")
        #self.pool.submit(lambda: requests.post(url, data=payload)).add_done_callback(lambda future: on_hsv_train_iterative(future.result()))

    def test(self, picklist):
        global gl_rec
        print("HSV Test Started")
        if self.counter % 5 == 0:
            print("Triggering Train HSV")
            self.train()
        picklist = str(picklist)
        model = self.model
        url = 'http://hsv_test:5000/hsv_test'
        payload = dict(id=picklist, picklist_nos=str([picklist]), hsv_avg_mean=json.dumps(model['hsv_avg_mean']), hsv_avg_std=json.dumps(model['hsv_avg_std']))
        def on_hsv_inference(result):
            global gl_rec
            if result.status_code != 200:
                print("HSV Test Errored")
                return
            gl_rec.record("End HSV Test")
            gl_rec.end()
            gl_rec = Recorder()
            print(result.content)
            print("HSV Test Complete")
            if picklist not in self.inters:
                # train hsv iterative after prediction
                print("Training HSV Iterative:", picklist)
                print(literal_eval(result.content.decode()))
                self.train_iterative(picklist, json.dumps(literal_eval(result.content.decode())[1])) # get the dictionary of predicted labels
            model = self.inters[int(picklist)]
        gl_rec.record("Start HSV Test")
        self.pool.submit(lambda: requests.post(url, data=payload)).add_done_callback(lambda future: on_hsv_inference(future.result()))
        self.counter += 1

htk_train_state = 0
htk_train_pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
curr_htk_model = dict()
def htk_trainer(htk_mgr):
    global htk_train_state
    global curr_htk_model
    if htk_mgr.picklist_counter == htk_train_state:
        return
    if (htk_mgr.picklist_counter - 1000) % 5 != 0:
        return
    print("Training HTK")
    url = 'http://htk_train:5000/htk'
    payload = {
        'id': 'test',
        'picklists': ','.join([
            *[str(i) for i in range(1, 41)],
            *[str(1000 + i) for i in range(htk_mgr.picklist_counter)]
        ])
    }
    headers = {'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.75 Safari/537.36'}
    def on_htk_train(result):
        if result.status_code != 200:
            print("HTK Train Errored")
            return
        gl_rec.record("End HTK Train")
        data = MultipartDecoder.from_response(result)
        for part in data.parts:
            curr_htk_model[part.headers[b'Content-Id'].decode()] = part.content
        print("HTK Processed")

    gl_rec.record("Start HTK Train")

    htk_train_pool.submit(
            lambda: requests.post(url, data=payload, headers=headers)
    ).add_done_callback(
            lambda future: on_htk_train(future.result())
    )
    htk_train_state = htk_mgr.picklist_counter



def preprocess(mgr, picklist, htk_mgr):
    print("PICKLIST COLLECTED:", mgr.counter)
    feature_vector = np.array([picklist[f'{i}'] for i in sorted(int(i) for i in picklist.keys())])
    np.savetxt(f"/shared/htk_inputs/picklist_{htk_mgr.picklist_counter}.txt", feature_vector, delimiter=" ")
    mgr.push(feature_vector)
    preprocessor_manager.flush(lambda x: htk_mgr.add_picklist(x))

if __name__ == '__main__':
    feature_manager = FrameFeatureManager()
    preprocessor_manager = PreprocessorManager()
    hsv_manager = HSVManager()
    htk_manager = HTKManager(hsv_manager)
    cap = cv2.VideoCapture()
    print("Started")
    while True:
        # this is problematic since rtmp connection is taking too much time
        # todo - make a new connection listener and have that multithreaded as well! have an endpoint with flask for which 
        # use nginx to redirect :)
        cap.open("rtmp://nginx-rtmp/live/test")
        print("Connected")
        gl_rec = Recorder()
        while cap.isOpened():
            ret, frame = cap.read()
            htk_trainer(htk_manager)
            if ret and frame is not None:
                feature_manager.push(frame)
            else: # TODO: confirm there is no better way to do this - what if RTMP dropped frame in a picklist?
                break

        feature_manager.flush(lambda x: preprocess(preprocessor_manager, x, htk_manager))
        print("Disconnected")
        cap.release()


