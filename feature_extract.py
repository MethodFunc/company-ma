import os, fnmatch, cv2
import mahotas as mt
import numpy as np
import pandas as pd

from tqdm import tqdm
from skimage.feature import local_binary_pattern


SOURCE_PATH = r'D:\Harry\002.Working\images\aaa'
SOURCE_PATH = SOURCE_PATH.replace('\\', '/')

ROI = [(2, 0), (3, 0), (4, 0), (5, 0), (6, 0), (7, 0), (8, 0), (9, 0), (2, 1), (3, 1), (4, 1), (5, 1),  (6, 1),  (7, 1), (8, 1), (9, 1), (2, 2), (3, 2), (4, 2), (5, 2), (6, 2), (7, 2), (8, 2), (9, 2)]

width, height = 150, 150

files = os.listdir(SOURCE_PATH)

roi_image = []
h_feature = []
lbp_feature = []
avg_val = []
for file_list in tqdm(files):
    path = f'{str(SOURCE_PATH)}/{str(file_list)}'

    img = mt.imread(path)

    for i, j in ROI:
        x, y = i * width, j*height
        roi_img = img[y:y+height, x:x+width]

        lab = cv2.cvtColor(roi_img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        y, x, z = roi_img.shape
        l_blur = cv2.GaussianBlur(l, (11, 11), 5)
        maxval = []

        count_percent = 5
        count_percent = count_percent / 100
        row_percent = int(count_percent * x)
        column_percent = int(count_percent * y)

        for i in range(1, x - 1):
            if i % row_percent == 0:
                for j in range(1, y - 1):
                    if j % column_percent == 0:
                        pix_cord = (i, j)

                        # cv2.circle(img_dot, (int(i), int(j)), 5, (0, 255, 0), 2)
                        img_segment = l_blur[i:i + 3, j:j + 3]
                        (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(img_segment)
                        maxval.append(maxVal)

        avg_maxval = round(sum(maxval) / len(maxval))
        avg_val.append(avg_maxval)

        roi_img = mt.colors.rgb2gray(roi_img, dtype=np.uint8)
        haralick = mt.features.haralick(roi_img).mean(axis=0)
        lbp = local_binary_pattern(roi_img, 16, 2, 'uniform').mean(axis=0)
        # lbp2 = local_binary_pattern(roi_img, 16, 2, 'uniform')

        roi_image.append(roi_img)
        h_feature.append(haralick)
        lbp_feature.append(lbp)

h_df = pd.DataFrame(h_feature)
lbp_df = pd.DataFrame(lbp_feature)
b_df = pd.DataFrame(avg_val)
# concat_df = pd.concat([h_df, lbp_df, b_df], axis=1)
# concat_df.to_csv('python_feature_rain.csv', index=False)

# roi_image = np.array(roi_image)
# #
# xs, ys = [], []
#
# for patch in roi_image:
#     glcm = greycomatrix(patch, distances=[5], angles=[0], levels=256, symmetric=True, normed=True)
#     xs.append(greycoprops(glcm, 'dissimilarity')[0, 0])
#     ys.append(greycoprops(glcm, 'correlation')[0, 0])
#
