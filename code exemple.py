import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns 
import skimage.io as sk
import matplotlib.cm as cm
import time as times
import sklearn as skl
from sklearn.cluster import KMeans
from sklearn import datasets
import pandas as pd
import matplotlib.image as mpimg
import gradio as gr
from phasor2 import phasor as ph
x=ph('C:/blala/blabla/yourimage',lamdbasmin=410)
x.calcul_phasor()
img=x.hsv()
x.plot_phasor()
x.pha.display_img(hsv = True,alpha_v =0.01, alpha_s =0.5)
