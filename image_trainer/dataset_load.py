import cv2
import fnmatch
import os
import random

import numpy as np
import logging
import logging.config

logging.config.fileConfig("config.conf")
logger = logging.getLogger("dataset_load")


def load_data(source_path, categories, sample_number, validation_ratio=0.1):
    logger.info("======dataset load start======")
    source_path = source_path.replace("\\", "/")
    train_set, test_set = {}, {}
    for cat in categories:
        path = f"{source_path}/{cat}"

        if not os.path.isdir(path):
            logger.warning(f"{cat} folder's not exist. check your folder or category")
            exit()

        img_list = fnmatch.filter(os.listdir(path), "*.jpg")

        if min(len(img_list), sample_number) == 0:
            logger.error(f"Check {cat} category images")
            exit()
        elif len(img_list) < sample_number:
            logger.warning(f'Number of {cat} image ({len(img_list)}) is smaller'
                           f' than SAMPLE NUMBER({sample_number}).')
            exit()

        random.shuffle(img_list)
        label = categories.index(cat)
        val_count = int(sample_number * validation_ratio)
        train_count = sample_number - val_count

        for img in img_list[:train_count]:
            img_path = f"{path}/{img}"
            train_set[img_path] = label

        for img in img_list[train_count:sample_number]:
            img_path = f"{path}/{img}"
            test_set[img_path] = labels

    logger.info("======dataset load end======")
    return train_set, test_set


# rgb height, width , depth(channel)
def load_image(name, dataset, roi=None, crop=None, height=150, width=150, depth=3):
    logger.info(f"====== {name} set load start======")
    random.shuffle(dataset)
    images, labels = [], []
    for imgs, label in dataset:
        img = cv2.imread(imgs)

        if crop:
            images.append(img)
            labels.append(label)

        if crop is None:
            try:
                img = image_preprocessing(img)
                for (i, j) in roi:
                    x, y = i * width, j * height
                    roi_img = img[y:y + height, x:x + width]
                    images.append(roi_img)
                    labels.append(label)
            except:
                logger.error(f'image processing failed at {str(imgs)}')
                pass

    images = np.array(images).reshape(-1, height, width, depth)
    labels = np.array(labels)

    logger.info(f"====== {name} set load End======")

    return images, labels


def image_preprocessing(image):
    img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
    img_rgb = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

    return img_rgb

if __name__ == "__main__":
    source_path = r"D:\weather\lbp_result"
    categories = ['normal', 'normal_night', 'fog', 'fog_night']
    sample_number = 100

    train_data, test_data = load_data(source_path, categories, sample_number)
    roi = [(2, 0), (3, 0), (4, 0), (5, 0), (6, 0), (7, 0), (8, 0), (9, 0), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1), (7, 1), (8, 1), (9, 1), (2, 2), (3, 2), (4, 2), (5, 2), (6, 2), (7, 2), (8, 2), (9, 2)]

    x_train, y_train = load_image(train_data, roi=roi)
    x_test, y_test = load_image(test_data, roi=roi)
