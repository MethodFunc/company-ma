from tensorflow.keras.models import Sequential, Model, load_model
from roi import setting_roi
import cv2
import numpy as np
import matplotlib.pyplot as plt


def read_img(path):
    path = path.replace('\\', '/')
    img = cv2.imread(path)

    return img


def roi_cut(img):
    roi_set = []
    for (i, j) in ROI:
        x, y = i * ROI_WIDTH, j * ROI_HEIGHT
        roi_img = img[y:y + ROI_HEIGHT, x:x + ROI_WIDTH]
        roi_set.append(roi_img)

    roi_set = np.array(roi_set).reshape(-1, ROI_HEIGHT, ROI_WIDTH, 3)

    return roi_set

ROI_WIDTH, ROI_HEIGHT = 150, 150
ROI = setting_roi('53R_202')

model_path = r'D:\Harry\000.DataAnalysis\004.Model\53R\53R_FOG_cp_20210106165128.tf'
model_path = model_path.replace('\\', '/')

# 정상
img_1 = r'C:\Users\user\Pictures\53R2021-01-12\new\OPTICAL202_20201108113955799623_0.jpg'
# 오분류
img_2 = r'C:\Users\user\Pictures\53R2021-01-12\OPTICAL202_20210112095409367793_3.jpg'
# 안개
img_3 = r'C:\Users\user\Pictures\53R2021-01-12\Fog\1\OPTICAL202_20201001071446040380_.jpg'
# 오분류 2
img_4 = r'C:\Users\user\Pictures\53R2021-01-12\OPTICAL202_20210112085605218671_3.jpg'

model = load_model(model_path)

load_img = read_img(img_2)
load_img = roi_cut(load_img)
load_img = load_img / 255.0

result = model.predict_classes(load_img)

preds = []
for pred in result:
    preds.append(pred)

prediction_list = []
for i in range(2):
    prediction_list.append(preds.count(i))


outputs = [model.layers[2].output]

feature_models = Model(model.inputs, outputs)
feature_maps = feature_models.predict(load_img)

square = 8
ix = 1
for _ in range(3):
    for _ in range(square):
        # specify subplot and turn of axis
        ax = plt.subplot(square, square, ix)
        ax.set_xticks([])
        ax.set_yticks([])
        # plot filter channel in grayscale
        plt.imshow(feature_maps[8, :, :, ix-1])
        ix += 1
# show the figure
plt.show()