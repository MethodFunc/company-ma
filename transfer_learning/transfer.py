import cv2
import os
import fnmatch
import random
import numpy as np

from roi import setting_roi
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping


class datamaker:
    def __init__(self, source_path, categories, roi, train_sample):
        self.source_path = source_path
        self.categories = categories
        self.roi = roi
        self.train_sample = train_sample
        self.train_image, self.train_label, self.val_image, self.val_label = [], [], [], []
        self.train_set, self.val_set = {}, {}

        self.validation_sample = 0.1
        self.width, self.height = 150, 150

    def __call__(self, *args, **kwargs):
        self.load_set()

        self.train_val_set(self.train_set, self.train_image, self.train_label)
        self.train_val_set(self.val_set, self.val_image, self.val_label)

        return (self.train_image, self.val_image, self.train_label, self.val_label)

    def load_set(self):
        for cat in self.categories:
            path = f"{self.source_path}/{cat}"

            if not os.path.isdir(path):
                print(f"{cat} is not exists. please check your category or directory")
                break

            img_file_list = fnmatch.filter(os.listdir(path), "*.jpg")
            random.shuffle(img_file_list)

            if len(img_file_list) < self.train_sample:
                print("Insufficient dataset.")
                break

            img_file_list = img_file_list[:self.train_sample]
            # print(img_file_list)

            val_number = int(self.train_sample * self.validation_sample)
            train_number = self.train_sample - val_number

            label = categories.index(cat)

            for img in img_file_list[:train_number]:
                img_path = f"{path}/{img}"
                self.train_set[img_path] = label

            for img in img_file_list[train_number:]:
                img_path = f"{path}/{img}"
                self.val_set[img_path] = label

        random.shuffle(self.train_set)
        random.shuffle(self.val_set)

    def train_val_set(self, dataset, img_list, label_list):
        for imgs, label in dataset:
            img = cv2.imread(imgs)
            for (i, j) in self.roi:
                x, y = i * self.width, j * self.height
                roi_img = img[y:y + self.height, x:x + self.width]

                img_list.append(roi_img)
                label_list.append(label)


if __name__ == "__main__":
    source_path = './201'
    categories = ['dry_day', 'dry_night', 'wet_day', 'wet_night']
    roi = setting_roi('33A_201')
    # roi = [(0, 5), (0, 7), (0, 8), (0, 9), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8), (2, 5), (2, 6), (2, 10), (3, 3), (3, 5), (3, 9), (4, 7)]
    train_sample = 500

    loaded = datamaker(source_path=source_path, categories=categories, roi=roi, train_sample=train_sample)
    (x_train, x_test, y_train, y_test) = loaded()

    x_train = np.array(x_train) / 255.0
    x_test = np.array(x_test) / 255.0
    y_train = to_categorical(y_train, len(categories))
    y_test = to_categorical(y_test, len(categories))

    model = load_model('53R_201cp_20201227135418.tf')
    model.compile(loss = "categorical_crossentropy", optimizer='adam', metrics=['accuracy'])


    history = model.fit(x_train, y_train, epochs=100, batch_size=128, validation_data=(x_test, y_test))
