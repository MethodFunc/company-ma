import os
import logging.config
import configparser
import numpy as np

from tensorflow.keras.callbacks import ModelCheckpoint, LambdaCallback
from tensorflow.keras.utils import to_categorical
from datetime import datetime
from datamaker import split_image, load_data
from load_model import base_model, custom_model
from roi import setting_roi, custom_roi


def __log():
    logging.config.fileConfig("logger.conf")
    return logging.getLogger("trainer")


def __config():
    config = configparser.ConfigParser()
    config.read("train.ini")

    return config


def run(source_path, categories, roi, sample_image, validation_ratio, EPOCHS, BATCH_SIZE, height, width, depth):
    logger = __log()
    set_model = config["model"]
    classes = len(categories)

    train_set, test_set = split_image(source_path=source_path, categories=categories, sample_image=sample_image,
                                      validation_ratio=validation_ratio)

    x_train, y_train = load_data(dataset=train_set, roi=roi, height=height, width=width, depth=depth)
    x_test, y_test = load_data(dataset=test_set, roi=roi, height=height, width=width, depth=depth)

    logger.info(f'Dataset Processing Finished.')
    logger.info(f'train_image: {x_train.shape}, train_label: {y_train.shape}')
    logger.info(f'test_image: {x_test.shape}, test_label: {y_test.shape}')

    x_train = x_train / 255.
    x_test = x_test / 255.
    y_train = to_categorical(y_train, classes)
    y_test = to_categorical(y_test, classes)

    model = None

    if set_model["base_model"] == "1":
        model = base_model(x_train, classes=classes)
        logger.info(f"Model setting: Base Model")

    if set_model["custom_model"] == "1":
        model = custom_model(x_train, classes=classes)
        logger.info(f"Model setting: Custom Model")

    if set_model["base_model"] == set_model["custom_model"]:
        logger.error(f"check train.ini - model - base_model & custom_model")
        logger.error(f"Ex. base_model = 1 -> custom_model = 0")
        logger.error(f"Ex. base_model = 0 -> custom_model = 1")
        exit()

    if not model:
        logger.error(f"check train.ini - model - base_model & custom_model")
        logger.error(f"Must One of the two should be 1")
        exit()

    os.chdir(source_path)
    start_time = datetime.now()
    cp = ModelCheckpoint(f"cp_{start_time.strftime('%Y%m%d_%H%M%S')}.tf", save_best_only=True, verbose=0)

    write_log = LambdaCallback(on_epoch_end=lambda epoch, logs: logger.info(f"EPOCH {epoch + 1}/{EPOCHS} - "
                                                                            f"loss: {logs['loss']:.4f} - "
                                                                            f"acc: {logs['acc']:.4f} - "
                                                                            f"val_loss: {logs['val_loss']:.4f} - "
                                                                            f"val_acc: {logs['val_acc']:.4f}"))

    history = model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(x_test, y_test),
                        callbacks=(cp, write_log), verbose=0)

    finish_time = datetime.now()
    run_time = finish_time - start_time

    optimal_loss = np.min(history.history['val_loss'])
    optimal_epoch = history.history['val_loss'].index(optimal_loss) + 1
    optimal_acc = (history.history['val_acc'][history.history['val_loss'].index(optimal_loss)]) * 100

    logger.info(f" >> Runtime     : {run_time}")
    logger.info(f' >> Model name: cp_{start_time.strftime("%Y%m%d_%H%M%S")}.tf')
    logger.info(f' >> Optimal_Epoch : {optimal_epoch}')
    logger.info(f' >> val_loss : {optimal_loss:.4f}({optimal_epoch} epoch)')
    logger.info(f' >> val_acc : {optimal_acc:.4f}({optimal_epoch} epoch)')


if __name__ == "__main__":
    logger = __log()
    config = __config()
    source_path = config["trainer"]["source_path"]
    if "\\" in source_path:
        source_path.replace("\\", "/")

    categories = [cat.strip() for cat in config["trainer"]["categories"].split(",")]

    roi = None

    if config["roi"]["setting_roi"] == "1":
        roi = setting_roi(config["roi"]["set_roi"])
        logger.info(f"Roi_setting: {config['roi']['set_roi']}")

    if config["roi"]["custom_roi"] == "1":
        roi = custom_roi(config["roi"]["cus_roi"])
        logger.info(f"Roi_setting: custom_roi")

    if config["roi"]["setting_roi"] == config["roi"]["custom_roi"]:
        logger.error(f"check train.ini - roi - setting_roi & custom_roi")
        logger.error(f"Ex. setting_roi = 1 -> custom_roi = 0")
        logger.error(f"Ex. setting_roi = 0 -> custom_roi = 1")
        exit()

    if not roi:
        logger.error(f"check train.ini - roi - setting_roi & custom_roi")
        logger.error(f"Must One of the two should be 1")
        exit()

    validation_ratio = float(config["trainer"]["validation_ratio"])
    sample_image = int(config["trainer"]["sample_image"])
    epoch = int(config["trainer"]["epoch"])
    batch_size = int(config["trainer"]["batch_size"])

    height = int(config["image_size"]["height"])
    width = int(config["image_size"]["width"])
    depth = int(config["image_size"]["depth"])

    run(source_path=source_path, categories=categories, roi=roi, sample_image=sample_image,
        validation_ratio=validation_ratio,
        EPOCHS=epoch, BATCH_SIZE=batch_size, height=height, width=width, depth=depth)
