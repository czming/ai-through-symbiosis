import torch
import torch.hub
import mediapipe as mp
import numpy as np
import cv2

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

# Create the model
model = torch.hub.load(
    repo_or_dir='guglielmocamporese/hands-segmentation-pytorch', 
    model='hand_segmentor', 
    pretrained=True
)
model.eval()

def get_hand_boundary(img):
  results = hands.process(img)
  # print(results.multi_hand_landmarks)
  try:
    landmarks = results.multi_hand_landmarks[0]
    h, w, c = img.shape
    x_max = 0
    y_max = 0
    x_min = w
    y_min = h
    if len(landmarks.landmark)==21:
      for lm in landmarks.landmark:
        x, y = int(lm.x * w), int(lm.y * h)
        if x > x_max:
            x_max = x
        if x < x_min:
            x_min = x
        if y > y_max:
            y_max = y
        if y < y_min:
            y_min = y
      hand = img[y_min-10:y_max+10,x_min-10:x_max+10, :]
      return hand
  except:
    pass

def get_hand_segmentation(frame):
  # img_rnd = torch.randn(1, 3, 256, 256) # [B, C, H, W]
  frame = np.swapaxes(frame, 1, 2)
  frame = np.swapaxes(frame, 0, 1)
  frame = frame / 255.0
  img = torch.tensor(np.expand_dims(frame, axis=0)).float()
  print(img.shape)
  preds = model(img).argmax(1) # [B, H, W]
  print(preds.shape)
  cv2.imwrite('prediction.png', preds[0,:,:].numpy()*255.0)
  return preds[0,:,:].numpy()*255.0