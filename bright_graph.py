import pandas as pd
import seaborn as sns
import plotly.graph_objects as go
import cv2
import os
import fnmatch
import glob
from tqdm import tqdm

source_path = r'D:\Harry\002.Working\bright\mk-sd53r\2020-11-21\201'
source_path = source_path.replace('\\', '/')
img_list = glob.glob(f'{source_path}/*jpg')

avg_val = []
for img in tqdm(img_list):
    img = cv2.imread(img)
    img_dot = img
    lab= cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    y, x, z = img.shape
    l_blur = cv2.GaussianBlur(l, (11, 11), 5)
    maxval = []

    count_percent = 5
    count_percent = count_percent/100
    row_percent = int(count_percent*x)
    column_percent = int(count_percent*y)

    for i in range(1,x-1):
        if i%row_percent == 0:
            for j in range(1, y-1):
                if j%column_percent == 0:
                    pix_cord = (i,j)

                    img_segment = l_blur[i:i+3, j:j+3]
                    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(img_segment)
                    maxval.append(maxVal)

    avg_maxval = round(sum(maxval) / len(maxval))
    avg_val.append(avg_maxval)


time_list = fnmatch.filter(os.listdir(source_path), '*jpg')
time_stamp = [times.split("_")[1][8:14] for times in time_list]

fig = go.Figure()
fig.add_trace(go.Scatter(y = avg_val, x=time_stamp))
fig.show()