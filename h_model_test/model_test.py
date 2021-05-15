import cv2
import os
import fnmatch
import numpy as np

from roi import setting_roi
from tensorflow.keras.models import load_model


def classifier(path, model_path, categories, roi, width=150, height=150, depth=3):
    test_path = path
    model = load_model(model_path)
    test_img_list = fnmatch.filter(os.listdir(test_path), "*.jpg")

    for img_name in test_img_list:
        path = f"{test_folder_path}/{img_name}"
        img = cv2.imread(path)

        roi_set = [img[j * height: j * height + height, i * width: i * width + width] for (i, j) in roi]
        roi_set = np.array(roi_set).reshape(-1, width, height, depth) / 255.
        predict = model.predict(roi_set)

        predict_value = [np.argmax(pred) for pred in predict]
        predict_list = [predict_value(n) for n in range(len(categories))]
        class_count = np.argmax(predict_list)

        show_predict_categories(img_name=img_name, predict_list=predict_list, categories=categories, class_count=class_count)


def show_predict_categories(img_name, predict_list, categories, class_count):
    if class_count:
        print(f"Filename: {img_name}, count: {predict_list}, class: {categories[class_count]}")


if __name__ == "__main__":
    test_folder_path = "/Users/methodfunc/Pictures/wet&moist"
    categories = ["wet", "moist"]
    model_path = ""
    roi = setting_roi("53R_201")

    classifier(path= test_folder_path, model_path=model_path, categories=categories, roi=roi)
