import os

from datamaker import DataMaker
from load_model import create_model
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from datetime import datetime

if __name__ == "__main__":
    source_path = 'D:/weather/categorical'
    categories = ['normal_day', 'normal_night', 'fog_day', 'fog_night']
    classnum = len(categories)
    roi0 = [(5, 0), (6, 0), (4, 0), (8, 0)]
    roi1 = [(5, 1), (6, 1), (7, 1), (8, 1), (9, 1)]
    sample_image = 400

    img_load = DataMaker(SOURCE_PATH=source_path, CATEGORIES=categories, ROI=roi0, SAMPLE_IMAGE=sample_image)

    (train_image, train_label, test_image, test_label) = img_load()

    model = create_model(train_image=train_image, classnum=classnum)

    # Setting Callback
    start = datetime.now()
    start = start.strftime("%Y%m%d_%H%M%S")
    log_dir = f"{source_path}/log/{start}"

    if not os.path.isdir(log_dir):
        os.mkdir(log_dir)

    tensorboard = TensorBoard(log_dir=log_dir, write_images=True, profile_batch=100000000)
    os.chdir(source_path)
    ck = ModelCheckpoint(f"fog0_{start}_{classnum}.tf", save_best_only=True)

    # Model Fit
    history = model.fit(train_image, train_label, epochs=300, batch_size=128, validation_data=(test_image, test_label),
                        callbacks=[tensorboard, ck], verbose=2)
