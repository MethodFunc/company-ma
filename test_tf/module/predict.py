import numpy as np

from tensorflow.keras.models import load_model


def predict_model(model_path: str, load_img: list):
    if "\\" in model_path:
        model_path = model_path.replace("\\", "/")

    model = load_model(model_path)

    predict = model.predict(load_img)

    predict_count = []

    for value in predict:
        predict_count.append(np.argmax(value))

    return predict_count


def predict_result(pred_list: list, categories: list, filename: str, show: int):
    value = np.argmax(pred_list)
    n = 0

    if value != show:
        print(f"{filename} - count: {pred_list}, class:{categories[value]}")
        n += 1

    return n
