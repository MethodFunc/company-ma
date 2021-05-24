from configparser import SectionProxy
import cv2
import os
import fnmatch
import random
import numpy as np

from tensorflow.keras.utils import to_categorical
from roi import setting_roi


class DataMaker:
    def __init__(self, source_path, categories, roi, train_sample, height=150, width=150, val_ratio=0.1, depth=3):
        self.source_path = source_path
        self.categories = categories
        self.roi = setting_roi(roi)
        self.train_sample = train_sample

        self.validation_sample = val_ratio
        self.width, self.height = width, height
        self.depth = depth

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

            label = self.categories.index(cat)

            self.train_set.update({f"{path}/{img}": label for img in img_file_list[:train_number]})
            self.val_set.update({f"{path}/{img}": label for img in img_file_list[train_number:]})

    def train_val_set(self, dataset, img_list, label_list):
        for imgs, label in dataset.items():
            img = cv2.imread(imgs)
            for (i, j) in self.roi:
                x, y = i * self.width, j * self.height
                roi_img = img[y:y + self.height, x:x + self.width]

                img_list.append(roi_img)
                label_list.append(label)

    def preprocessing(self):
        self.train_image = (np.array(self.train_image) / 255.).reshape(-1, self.height, self.width, self.depth)
        self.val_image = (np.array(self.val_image) / 255.).reshape(-1, self.height, self.width, self.depth)
        self.train_label = to_categorical(self.train_label, len(self.categories))
        self.val_label = to_categorical(self.val_label, len(self.categories))
