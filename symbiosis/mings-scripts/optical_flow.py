import os
from abc import ABC, abstractmethod
from collections import deque

import numpy as np
import cv2


class Flow(ABC):

    def __init__(self, *params, frame_distance=1, **kwargs):
        self.queue = deque(maxlen=frame_distance)
        self.frame_distance = frame_distance

        self.params = params

        self.algo = None

    def _ready(self):
        return len(self.queue) == self.frame_distance

    def calc(self, frame:np.ndarray, *params, to_gray=False) -> np.ndarray:
        if to_gray:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        ret_val = np.zeros((frame.shape[0], frame.shape[1])),np.zeros((frame.shape[0], frame.shape[1]))
        if self._ready():
            assert frame.shape == self.queue[0].shape
            assert frame.dtype == self.queue[0].dtype
            params = params if params else self.params
            flow = self.algo(self.queue[0], frame, None, *params)
            ret_val = cv2.cartToPolar(flow[:,:, 0], flow[:,:, 1]) #mag, ang

        self.queue.append(frame)

        return ret_val

class RLOFlow(Flow):

    def __init__(self, *params, frame_distance=1, **kwargs):
        super(RLOFlow, self).__init__(*params, frame_distance=frame_distance, **kwargs)
        algo = cv2.optflow.DenseRLOFOpticalFlow_create()
        self.algo = algo.calc

    def calc(self, frame:np.ndarray, *params) -> np.ndarray:
        return super().calc(frame, *params)



class FarnebackFlow(Flow):

    def __init__(self, *params, frame_distance=1, **kwargs):
        super(FarnebackFlow, self).__init__(*params, frame_distance=frame_distance, **kwargs)

        algo = cv2.FarnebackOpticalFlow_create(*params, **kwargs)
        self.algo = algo.calc

    def calc(self, frame:np.ndarray, *params) -> np.ndarray:
        if len(frame.shape) > 2 and frame.shape[2] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        return super().calc(frame, *params)


def global_optical_flow(videopath:str, dist:int=1, scale:float=1., outpath:str=None):
    cap = cv2.VideoCapture(videopath)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    flow_algo = FarnebackFlow(frame_distance=dist)
    flows = []

    for i in tqdm(range(num_frames), total=num_frames):
        ret, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.resize(frame, None, fx=scale, fy=scale)

        mag, ang = flow_algo.calc(frame)
        if i < dist:
            continue
        flow_x, flow_y = cv2.polarToCart(mag,ang)

        flows.append((flow_x.mean(), flow_y.mean()))

    flows = np.asarray(flows)
    flows = np.hstack(cv2.cartToPolar(flows[:,0], flows[:,1]))

    cap.release()
    cv2.destroyAllWindows()

    if outpath is None:
        dirname = os.path.dirname(videopath)
        basename = os.path.basename(videopath)
        basename = basename[:basename.index('.')]

        outpath = os.path.join(dirname, f'{videopath}_framedist_{dist}.npy')

    np.save(outpath, flows)

    return flows