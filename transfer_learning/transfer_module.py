import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Dropout


def transfer(model_path, categories):
    classes = len(categories)
    model_load = load_model(model_path)
    model_list = []

    for layer in model_load.layers:
        if "dense" in layer.name:
            break
        else:
            model_list.append(layer)

    base_model = Sequential(model_list)
    base_model.trainable = False

    base_model.add(Dense(64, activation="relu"))
    base_model.add(Dropout(0.3))
    base_model.add(Dense(32, activation="relu"))
    base_model.add(Dropout(0.1))
    base_model.add(Dense(16, activation="relu"))
    base_model.add(Dense(classes, activation="softmax"))

    base_model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001), metrics=["accuracy"])

    return base_model


def trainable_true(model):
    for layer in model.layers:
        if "dense" in layer.name:
            break
        else:
            layer.trainable = True

    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0005), metrics=["accuracy"])

    return model


def view_graph(history):
    plt.figure(figsize=(32, 18))
    fig, loss_ax = plt.subplots()
    acc_ax = loss_ax.twinx()
    loss_ax.plot(history.history["loss"], "y", label="Train loss")
    loss_ax.plot(history.history["val_loss"], "r", label="Val loss")
    loss_ax.set_xlabel("Epochs")
    loss_ax.set_ylabel("Loss")
    loss_ax.legend(loc="upper right")

    acc_ax.plot(history.history["accuracy"], "b", label="Train acc")
    acc_ax.plot(history.history["val_accuracy"], "g", label="Val acc")
    acc_ax.set_ylabel("Accuracy")
    acc_ax.legend(loc="upper left")

    plt.show()
