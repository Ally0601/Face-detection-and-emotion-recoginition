
import os
from fastai.vision import *
from fastai import *
import matplotlib.pyplot as plt
import seaborn as sns
from functools import partial
from tqdm.notebook import tqdm
import gc
from pylab import imread,subplot,imshow,show

model_path = r'Enter path to your model'

def load_model(model_path):
    learn = load_learner(model_path)

    return learn

def predict_emotion(img):
    learn = load_model(model_path)
    pred_class, pred_idx, outputs = learn.predict(img)

    return pred_class, pred_idx, outputs