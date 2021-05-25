import os
import configparser
import logging.config
import numpy as np

from datamaker import DataMaker
from load_model import create_model
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, LambdaCallback
from roi import setting_roi
from datetime import datetime

if __name__ == "__main__":
    logging.config.fileConfig("logger.conf")
    logger = logging.getLogger("trainer")
    config = configparser.ConfigParser()
    config.read("trainer.ini")

    source_path = config["train"]["source_path"]
    if "\\" in source_path:
        source_path = source_path.replace("\\", "/")

    categories = [cat.strip() for cat in config["train"]["categories"].split(",")]
    classes = len(categories)

    if "/" not in config["train"]["roi"]:
        roi = setting_roi(config["train"]["roi"])
    else:
        temp = config["train"]["roi"].split(",")
        roi = [(int(shape.split("/")[0]), int(shape.split("/")[1])) for shape in temp]

    sample_image = int(config["train"]["sample_image"])
    EPOCHS = int(config["train"]["epoch"])
    BATCH_SIZE = int(config["train"]["batch_size"])

    img_load = DataMaker(source_path=source_path, categories=categories, roi=roi, sample_image=sample_image)

    (train_image, train_label, test_image, test_label) = img_load()

    model = create_model(train_image=train_image, classnum=classes)

    # Setting Callback
    start = datetime.now()
    start = start.strftime("%Y%m%d_%H%M%S")
    log_dir = f"{source_path}/log/{start}"

    if not os.path.isdir(f"{source_path}/log/"):
        os.mkdir(f"{source_path}/log/")

    if not os.path.isdir(log_dir):
        os.mkdir(log_dir)

    tensorboard = TensorBoard(log_dir=log_dir, write_images=True, profile_batch=100000000)
    os.chdir(source_path)
    ck = ModelCheckpoint(f"cp_{start}_{classes}.tf", save_best_only=True)

    write_log = LambdaCallback(on_epoch_end=lambda epoch, logs: logger.info(f"EPOCH {epoch + 1}/{EPOCHS} - "
                                                                            f"loss: {logs['loss']:.4f} - "
                                                                            f"acc: {logs['acc']:.4f} - "
                                                                            f"val_loss: {logs['val_loss']:.4f} - "
                                                                            f"val_acc: {logs['val_acc']:.4f}"))

    # Model Fit
    history = model.fit(train_image, train_label, epochs=EPOCHS, batch_size=BATCH_SIZE,
                        validation_data=(test_image, test_label),
                        callbacks=[tensorboard, ck, write_log], verbose=0)

    optimal_loss = np.min(history.history['val_loss'])
    optimal_epoch = history.history['val_loss'].index(optimal_loss) + 1
    optimal_acc = (history.history['val_acc'][history.history['val_loss'].index(optimal_loss)]) * 100

    logger.info(f' >> Optimal_Epoch : {optimal_epoch}')
    logger.info(f' >> val_loss : {optimal_loss:.4f}({optimal_epoch} epoch)')
    logger.info(f' >> val_acc : {optimal_acc:.4f}({optimal_epoch} epoch)')