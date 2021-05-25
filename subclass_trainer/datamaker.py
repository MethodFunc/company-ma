import cv2
import fnmatch
import random
import os
import numpy as np
import logging.config

from tensorflow.keras.utils import to_categorical


class DataMaker:
    def __init__(self, source_path, categories, roi, sample_image, validation_ratio=0.1, width=150, height=150,
                 depth=3):
        self.source_path = source_path
        self.categories = categories
        self.sample_image = sample_image
        self.validation_ratio = validation_ratio
        self.roi = roi
        self.width = width
        self.height = height
        self.depth = depth
        self.classes = len(self.categories)

        self.train_set, self.test_set = {}, {}
        logging.config.fileConfig("logger.conf")
        self.logger = logging.getLogger("datamaker")

    def __call__(self):
            self.logger.info("=== Data Maker Start ===")
            self.split_image()

            self.logger.info(f'Processing training dataset...')
            train_images, train_labels = self.load_image_data(self.train_set)
            self.logger.info(f'Processing validation dataset...')
            test_images, test_labels = self.load_image_data(self.test_set)

            train_images = train_images / 255.
            test_images = test_images / 255.
            train_labels = to_categorical(train_labels, self.classes)
            test_labels = to_categorical(test_labels, self.classes)

            self.logger.info(f'Dataset Processing Finished.')
            self.logger.info(f'train_image: {train_images.shape}, train_label: {train_labels.shape}')
            self.logger.info(f'test_image: {test_images.shape}, test_label: {test_labels.shape}')

            return train_images, train_labels, test_images, test_labels

    def split_image(self):
        for cat in self.categories:
            cat_path = f"{self.source_path}/{cat}"

            if not os.path.isdir(cat_path):
                self.logger.warning(f"Category: [{cat}] is not exist. Check your category or folder name.")
                break

            img_file_list = fnmatch.filter(os.listdir(cat_path), "*.jpg")

            if len(img_file_list) == 0:
                self.logger.warning(f"{cat} image file does not exist.")
                self.logger.warning("Check your files")
                break

            if len(img_file_list) < self.sample_image:
                self.logger.warning(f"{cat} image file less then sample_image.")
                self.logger.warning("Check your files")
                break

            random.shuffle(img_file_list)

            test_num = int(self.sample_image * self.validation_ratio)
            train_num = self.sample_image - test_num

            label = self.categories.index(cat)

            self.train_set.update({f"{cat_path}/{img}": label for img in img_file_list[:train_num]})
            self.test_set.update({f"{cat_path}/{img}": label for img in img_file_list[train_num:self.sample_image]})
        #
        # random.shuffle(self.train_set)
        # random.shuffle(self.test_set)

    def load_image_data(self, dataset):
        images, labels = [], []
        for imgs, label in dataset.items():
            img = cv2.imread(imgs)
            for (i, j) in self.roi:
                x, y = i * self.width, j * self.height
                roi_img = img[y:y + self.height, x:x + self.width]

                images.append(roi_img)
                labels.append(label)

        images = np.array(images).reshape(-1, self.width, self.height, self.depth)
        labels = np.array(labels)

        return images, labels
