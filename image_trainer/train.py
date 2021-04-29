from dataset_load import load_data, load_image
from data_train import data_processing, create_model
from roi import setting_roi
from datetime import datetime
from tensorflow.keras.callbacks import ModelCheckpoint

if __name__ == "__main__":
    source_path = r'D:\weather\lbp_result'
    categories = ['normal', 'normal_night', 'fog', 'fog_night']
    sample_num = 400
    roi = setting_roi('53R_202')
    CLASSNUM = len(categories)

    train_set, test_set = load_data(source_path=source_path, categories=categories, sample_number=sample_num)
    x_train, y_train = load_image("train_set", train_set, roi=roi)
    x_test,  y_test = load_image("test_set", test_set, roi=roi)

    x_train, x_test, y_train, y_test = data_processing(x_train, x_test, y_train, y_test, CLASSNUM=CLASSNUM)

    model = create_model(CLASSNUM, x_train[0].shape)

    start_time = datetime.now()
    cp_path = f'cp_{start_time.strftime("%Y%m%d%H%M%S")}.tf'
    checkpoint = ModelCheckpoint(filepath=cp_path, monitor='val_loss', verbose=3,
                                 save_best_only=True)
    history = model.fit(x_train, y_train, epochs=100, batch_size=128, validation_data=(x_test, y_test),
              callbacks=[checkpoint], verbose=2)

