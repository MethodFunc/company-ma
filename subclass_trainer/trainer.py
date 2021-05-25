import os
import configparser

from datamaker import DataMaker
from load_model import create_model
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from roi import setting_roi
from datetime import datetime

if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read("trainer.ini")

    source_path = config["train"]["source_path"]
    if "\\" in source_path:
        source_path = source_path.replace("\\", "/")

    categories = config["train"]["categories"].split(",")
    classes = len(categories)

    if "/" not in config["train"]["roi"]:
        roi = setting_roi(config["train"]["roi"])
    else:
        temp = config["train"]["roi"].split(",")
        roi = [(int(shape.split("/")[0]), int(shape.split("/")[1])) for shape in temp]

    sample_image = int(config["train"]["sample_image"])

    img_load = DataMaker(source_path=source_path, categories=categories, roi=roi, sample_image=sample_image)

    (train_image, train_label, test_image, test_label) = img_load()

    model = create_model(train_image=train_image, classnum=classes)

    # Setting Callback
    start = datetime.now()
    start = start.strftime("%Y%m%d_%H%M%S")
    log_dir = f"{source_path}/log/{start}"

    if not os.path.isdir(log_dir):
        os.mkdir(log_dir)

    tensorboard = TensorBoard(log_dir=log_dir, write_images=True, profile_batch=100000000)
    os.chdir(source_path)
    ck = ModelCheckpoint(f"fog0_{start}_{classes}.tf", save_best_only=True)

    # Model Fit
    history = model.fit(train_image, train_label, epochs=300, batch_size=128, validation_data=(test_image, test_label),
                        callbacks=[tensorboard, ck], verbose=2)
