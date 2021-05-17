from configparser import SectionProxy
from typing import Sequence
import cv2
import os
import fnmatch
import random
import numpy as np

from .roi import setting_roi
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, Dense, Dropout


class datamaker:
    def __init__(self, source_path, categories, roi, train_sample, height=150, width=150, val_ratio=0.1):
        self.source_path = source_path
        self.categories = categories
        self.roi = roi
        self.train_sample = train_sample

        self.validation_sample = val_ratio
        self.width, self.height = width, height

        self.train_image, self.train_label, self.val_image, self.val_label = [], [], [], []
        self.train_set, self.val_set = {}, {}

    def __call__(self, *args, **kwargs):
        self.load_set()

        self.train_val_set(self.train_set, self.train_image, self.train_label)
        self.train_val_set(self.val_set, self.val_image, self.val_label)
        self.preprocessing()

        return self.train_image, self.val_image, self.train_label, self.val_label

    def load_set(self):
        for cat in self.categories:
            path = f"{self.source_path}/{cat}"

            if not os.path.isdir(path):
                print(f"{cat} is not exists. please check your category or directory")
                break

            img_file_list = fnmatch.filter(os.listdir(path), "*.jpg")

            if len(img_file_list) < self.train_sample:
                print("Insufficient dataset.")
                break

            random.shuffle(img_file_list)

            img_file_list = img_file_list[:self.train_sample]

            val_number = int(self.train_sample * self.validation_sample)
            train_number = self.train_sample - val_number

            label = categories.index(cat)

            self.train_set.update({f"{path}/{img}": label for img in img_file_list[:train_number]})
            self.val_set.update({f"{path}/{img}": label for img in img_file_list[train_number:]})

    def train_val_set(self, dataset, img_list, label_list):
        for imgs, label in dataset:
            img = cv2.imread(imgs)
            for (i, j) in self.roi:
                x, y = i * self.width, j * self.height
                roi_img = img[y:y + self.height, x:x + self.width]

                img_list.append(roi_img)
                label_list.append(label)

    def preprocessing(self):
        self.train_image = np.array(self.train_image) / 255.
        self.val_image = np.array(self.train_image) / 255.
        self.train_label = to_categorical(self.train_image, len(self.categories))
        self.val_label = to_categorical(self.val_image, len(self.categories))


def transfer(model_path, categories):
    classes = len(categories)
    load_model = load_model(model_path)
    model_list = []
    for layer in load_model.layers:
        if "dense" in layer.name:
            break
        else:
            model_list.append(layer)

    base_model = Sequence(model_list)
    base_model.trainable = False
    
    base_model.add(Dense(64, activation="relu"))
    base_model.add(Dropout(0.3))
    base_model.add(Dense(32, activation="relu"))
    base_model.add(Dropout(0.1))
    base_model.add(Dense(16, activation="relu"))
    base_model.add(Dense(classes, activation="softmax"))

    base_model.compile(loss="categorial_crossentropy", optimizer = Adam(lr=0.001))

    return base_model

def trainable_true(model):
    for layer in model.layers:
        if "dense" in layer.name:
            break
        else:
            layer.trainable=True

    
    model.compile(loss="categorial_crossentropy", optimizer = Adam(lr=0.0005))

    return model

if __name__ == "__main__":
    source_path = "./201"
    model_path = "53R_201cp_20201227135418.tf"
    categories = ['dry_day', 'dry_night', 'wet_day', 'wet_night']
    roi = setting_roi('33A_201')
    train_sample = 500

    loaded = datamaker(source_path=source_path, categories=categories, roi=roi, train_sample=train_sample)
    (x_train, x_test, y_train, y_test) = loaded()


    model = transfer(model_path=model_path, categories=categories)
    history_frozen = model.fit(x_train, y_train, epochs=100, batch_size=128, validation_data=(x_test, y_test))

    model = trainable_true(model=model)
    history = model.fit(x_train, y_train, epochs=100, batch_size=128, validation_data=(x_test, y_test))

