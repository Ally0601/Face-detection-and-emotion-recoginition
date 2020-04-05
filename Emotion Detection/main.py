import cv2
from emotion_detection import predict_emotion
from PIL import Image as PImage

import numpy as np
import pandas as pd

import os
from fastai.vision import *
from fastai import *
import matplotlib.pyplot as plt
import seaborn as sns
from functools import partial
from tqdm.notebook import tqdm
import gc
from pylab import imread,subplot,imshow,show

def draw_boundary(img, classifier, scaleFactor, minNeighbors, color, text):

    ## converts color image to grayimage
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ## detects image using classifier
    features = classifier.detectMultiScale(gray_img, scaleFactor, minNeighbors)
    coords = []
    for (x, y, w, h) in features:
        cv2.rectangle(img, (x, y), (x +w, y+h), color, 2)
        cv2.putText(img, text, (x, y-4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
        coords = [x, y, w, h]

    return coords
#### ---------- Face and Eyes Dectection ---------------- ################
def detect(img, faceCascade, eyesCascade, text):
    color = {'blue':(255,0,0), 'red':(0,0,255), 'green':(0,255,0), 'white':(255,255,255)}
    coords = draw_boundary(img, faceCascade, 1.1, 10, color['white'], text=text)
    # print(coords)
    
    ## Face detection ###
    if len(coords) == 4:
        face_img = img[coords[1]:coords[1]+coords[3], coords[0]:coords[0]+coords[2]]
    #     coords = draw_boundary(roi_img, eyesCascade, 1.1, 14, color['red'], "Eyes")
        return img, face_img
    
    else:
        return img


## classifiers to detect face
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eyesCascade = cv2.CascadeClassifier('haarcascade_eye.xml')

## Uses 0: webcam for videocapture
## Uses -1: external drives for videocapture
video_capture = cv2.VideoCapture(0)
pred_class = "No emotion detected"
while True:

    ## reads data from webcam
    _, img = video_capture.read()
    
    ## crops face as images
    try:
        
        img, face_img = detect(img, faceCascade, eyesCascade, pred_class)
        ## converts np.array to image
        pil_im = PImage.fromarray(face_img)

        ## converts pilImage to tensor
        x = pil2tensor(pil_im ,np.float32).div_(255)
        fast_img = Image(x)

        ## predict emotion from the tensor image
        pred_class, pred_idx, outputs = predict_emotion(fast_img)

        ## type casting fastai.core.Category to str
        pred_class = str(pred_class)

        ## printing detected emotion
        print(pred_class)

    except ValueError:
        img = detect(img, faceCascade, eyesCascade, "Neutral")

    ## opens tab to show output from webcam
    cv2.imshow("Face Dectection", img)

    ## for closing the cam
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()

