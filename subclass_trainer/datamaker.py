import cv2
import fnmatch
import random
import os
import numpy as np
from pprint import pprint

from tensorflow.keras.utils import to_categorical, plot_model


class DataMaker:
    def __init__(self, SOURCE_PATH, CATEGORIES, ROI, SAMPLE_IMAGE, validation_ratio=0.1, width=150, height=150,
                 depth=3):
        self.source_path = SOURCE_PATH
        self.categories = CATEGORIES
        self.sample_image = SAMPLE_IMAGE
        self.validation_ratio = validation_ratio
        self.roi = ROI
        self.width = width
        self.height = height
        self.depth = depth
        self.classnum = len(self.categories)

        self.train_set, self.test_set = [], []

    def __call__(self):
        self.split_image()

        train_images, train_labels = self.load_image_data(self.train_set)
        test_images, test_labels = self.load_image_data(self.test_set)

        train_images = train_images / 255.
        test_images = test_images / 255.
        train_labels = to_categorical(train_labels, self.classnum)
        test_labels = to_categorical(test_labels, self.classnum)

        return train_images, train_labels, test_images, test_labels

    def split_image(self):
        for cat in self.categories:
            cat_path = f"{self.source_path}/{cat}"

            if not os.path.isdir(cat_path):
                pprint(f"Category: [{cat}] is not exist. Check your category or folder name.")
                break

            img_file_list = fnmatch.filter(os.listdir(cat_path), "*.jpg")

            if len(img_file_list) == 0:
                pprint(f"{cat} image file does not exist.")
                pprint("Check your files")
                break

            if len(img_file_list) < self.sample_image:
                pprint(f"{cat} image file less then sample_image.")
                pprint("Check your files")
                break

            random.shuffle(img_file_list)

            test_num = int(self.sample_image * self.validation_ratio)
            train_num = self.sample_image - test_num

            label = self.categories.index(cat)

            for imgs in img_file_list[:train_num]:
                img_path = f"{cat_path}/{imgs}"
                self.train_set.append((img_path, label))

            for imgs in img_file_list[train_num:self.sample_image]:
                img_path = f"{cat_path}/{imgs}"
                self.test_set.append((img_path, label))

        random.shuffle(self.train_set)
        random.shuffle(self.test_set)

    def load_image_data(self, dataset):
        images, labels = [], []
        for imgs, label in dataset:
            img = cv2.imread(imgs)
            for (i, j) in self.roi:
                x, y = i * self.width, j * self.height
                roi_img = img[y:y + self.height, x:x + self.width]

                images.append(roi_img)
                labels.append(label)

        images = np.array(images).reshape(-1, self.width, self.height, self.depth)
        labels = np.array(labels)

        return images, labels
