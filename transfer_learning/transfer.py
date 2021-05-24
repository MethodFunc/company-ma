import os
import tensorflow as tf

from datetime import datetime
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from datamaker import DataMaker
from transfer_module import transfer, trainable_true

if __name__ == "__main__":

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    source_path = "./201"
    model_path = "53R_201cp_20201227135418.tf"
    categories = ['dry_day', 'dry_night', 'wet_day', 'wet_night']
    roi = '33A_201'
    train_sample = 500

    loaded = DataMaker(source_path=source_path, categories=categories, roi=roi, train_sample=train_sample)
    (x_train, x_test, y_train, y_test) = loaded()

    print("load model & freezing layers")
    model = transfer(model_path=model_path, categories=categories)
    history_frozen = model.fit(x_train, y_train, epochs=100, batch_size=128, validation_data=(x_test, y_test))
    print("Done... Unfreezing layers tuning models")

    os.chdir(source_path)
    start_time = datetime.now()
    cp_path = f'cp_{start_time.strftime("%Y%m%d%H%M%S")}_transfer.tf'
    cp = ModelCheckpoint(filepath=cp_path, monitor="val_loss", save_best_only=True)
    log_path = "./logs"
    if not os.path.isdir(log_path):
        os.mkdir(log_path)

    if not os.path.isdir(f"{log_path}/{start_time.strftime('%Y%m%d_%H%M%S')}"):
        os.mkdir(f"{log_path}/{start_time.strftime('%Y%m%d_%H%M%S')}")

    tb = TensorBoard(log_dir=f"{log_path}/{start_time.strftime('%Y%m%d_%H%M%S')}", profile_batch=99999)

    model = trainable_true(model=model)
    history = model.fit(x_train, y_train, epochs=100, batch_size=128, validation_data=(x_test, y_test),
                        callbacks=[cp, tb])
