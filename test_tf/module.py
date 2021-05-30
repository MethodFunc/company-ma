import os
import cv2
import fnmatch
import numpy as np

from tensorflow.keras.models import load_model


class ImageProcess:
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


def predict_images(model_path: str, load_image: list, classes: int):
    model = load_model(model_path)
    predict = model.predict(load_image)

    predict_count = []

    for value in predict:
        predict_count.append(np.argmax(value))

    prediction_list = []

    for i in range(classes):
        prediction_list.append(predict_count.count(i))

    return prediction_list


def printresult(prediction_list: list, classes: int, categories):
    n = 0
    predict = np.argmax(prediction_list)

    if predict != classes:
        print(f"count: {prediction_list}, class: {categories[predict]}")
        n += 1

    return n


if __name__ == "__main__":
    source_path = "/Users/methodfunc/Pictures/wet&moist"

    print(ImageProcess.images_list(source_path=source_path))

