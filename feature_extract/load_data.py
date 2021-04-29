
import cv2
import os
import fnmatch
import mahotas as mt
import numpy as np
from skimage.feature import local_binary_pattern, greycomatrix, greycoprops


def feature_extractor(source_path, roi):
    source_path = source_path.replace('\\', '/')
    height, width = 150, 150
    eps = 1e-7
    file_list = fnmatch.filter(os.listdir(source_path), '*.jpg')

    haralick_list, lbp_list, bright_list = [], [], []

    for img in file_list:
        path = os.path.join(source_path, img)

        img = cv2.imread(path)

        for (i, j) in roi:
            x, y = i * width, j * height
            roi_img = img[y:y + width, x:x + height]
            bright = calc_bright(roi_img)
            roi_img = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
            haralick = mt.features.haralick(roi_img).mean(axis=0)
            lbp = local_binary_pattern(roi_img, 8, 3, method='uniform')
            lbp = np.array(lbp)
            (hist, _) = np.histogram(lbp.ravel(), density=True, bins=int(lbp.max() + 1),
                                     range=(0, int(lbp.max() + 1)))
            hist = hist.astype("float")
            hist /= (hist.sum() + eps)

            haralick_list.append(haralick)
            lbp_list.append(hist)
            bright_list.append(bright)

    output = np.concatenate((np.array(haralick_list), np.array(lbp_list), np.array(bright_list)), axis=1)

    return output


def calc_bright(img_file):
    avg_val = []
    lab = cv2.cvtColor(img_file, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    y, x, z = img_file.shape
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
                    img_segment = l_blur[i:i + 3, j:j + 3]
                    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(img_segment)
                    maxval.append(maxVal)

    avg_maxval = round(sum(maxval) / len(maxval))
    avg_val.append(avg_maxval)

    return avg_val


def feature_extractor_glcm(source_path, roi):
    source_path = source_path.replace('\\', '/')
    height, width = 150, 150
    file_list = fnmatch.filter(os.listdir(source_path), '*.jpg')

    glcm_list = [], [], []

    for img in file_list:
        path = os.path.join(source_path, img)

        img = cv2.imread(path)

        for (i, j) in roi:
            x, y = i * width, j * height
            roi_img = img[y:y + width, x:x + height]
            roi_img = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
            glcm_list

            bright_list.append(bright)

    output = np.concatenate((np.array(haralick_list), np.array(lbp_list), np.array(bright_list)), axis=1)

    return output


if __name__ == '__main__':
    source_path = r'D:\Harry\002.Working\images\fog'

    roi = [(5, 0), (6, 0), (7, 0), (5, 1), (6, 1), (7, 1)]

    test = feature_extractor(source_path, roi)
