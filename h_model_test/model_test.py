import cv2
import os
import shutil
import fnmatch
import numpy as np

from roi import setting_roi
from tensorflow.keras.models import load_model


def classifier(path: str, model_path: str, categories: list, roi: list, show=None, move=None, width=150, height=150,
               depth=3):
    test_path = path
    model = load_model(model_path)
    test_img_list = fnmatch.filter(os.listdir(test_path), "*.jpg")
    count = 0

    for i in range(len(categories)):
        globals()[f"count_{i}"] = 0

    for img_name in test_img_list:
        path = f"{test_folder_path}/{img_name}"
        img = cv2.imread(path)

        roi_set = [img[j * height: j * height + height, i * width: i * width + width] for (i, j) in roi]
        roi_set = np.array(roi_set).reshape(-1, height, width, depth) / 255.
        predict = model.predict(roi_set)

        predict_value = [np.argmax(pred) for pred in predict]
        predict_list = [predict_value.count(n) for n in range(len(categories))]

        if show:
            counted = show_predict_categories(img_name=img_name, predict_list=predict_list, categories=categories,
                                              show_number=show)
            count += counted

        if move:
            predict_num = np.argmax(predict_list)
            automatic_classifier(test_path=test_path, categories=categories, image_name=img_name, class_num=predict_num)

    if show:
        print(f"분류 정확도: {count} / {len(test_img_list) * 100:.2f}%")


def automatic_classifier(test_path: str, categories: list, image_name: str, class_num: np):
    if class_num:
        if not os.path.isdir(f"{test_path}/{categories[class_num]}"):
            os.mkdir(f"{test_path}/{categories[class_num]}")

        path = f"{test_path}/{categories[class_num]}"
        shutil.move(f"{test_path}/{image_name}", f"{path}/{image_name}")
        
        globals()[f"count_{class_num}"] += 1
        print(f"{globals()[f'count_{class_num}']}")


def show_predict_categories(img_name: str, predict_list: list, categories: list, show_number: int) -> int:
    count = 0
    if np.argmax(predict_list) != show_number:
        print(f"Filename: {img_name}, count: {predict_list}, class: {categories[np.argmax(predict_list)]}")
    else:
        count += 1

    return count


if __name__ == "__main__":
    test_folder_path = "/Users/methodfunc/Pictures/wet&moist"
    categories = ["wet", "moist"]
    model_path = ""
    roi = setting_roi("53R_201")
    show_categories = None

    classifier(path=test_folder_path, model_path=model_path, categories=categories, roi=roi, show=None, move=None)
