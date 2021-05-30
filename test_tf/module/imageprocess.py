import cv2
import os
import fnmatch
import numpy as np


def images_list(source_path: str, extension="jpg"):
    path = [f"{source_path}/{img_list}" for img_list in fnmatch.filter(os.listdir(source_path), f"*.{extension}")]

    return path


def load_images(path: str, roi: list, height=150, width=150, depth=3):
    load_image = []
    img = cv2.imread(path)
    for (i, j) in roi:
        x, y = i * width, j * height
        roi_img = img[y:y + height, x:x + width]
        load_image.append(roi_img)

    load_image = np.array(load_image).reshape(-1, height, width, depth)
    load_image = load_image / 255.

    return load_image
