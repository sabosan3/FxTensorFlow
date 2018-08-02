import datetime

import keras
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping

from data_manager import DataManager


def main():
    dm = DataManager()
    train_x, train_y, val_x, val_y, test_x, test_y = dm.get_inputs(dim=5)

    # create model
    n_input_dimension = train_x.shape[1]
    n_labels = train_y.shape[1]

    model = Sequential()
    model.add(Dense(50, input_shape=(n_input_dimension,)))
    model.add(Dense(50))
    model.add(Dense(n_labels, activation='softmax'))
    optimizer = Adam(lr=0.0001)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # callbacks
    log_name = "log_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    tb_path = "./logs/tensorboard/{}".format(log_name)
    tb = keras.callbacks.TensorBoard(log_dir=tb_path)
    es = EarlyStopping(monitor='val_loss', patience=10, verbose=1)

    # fit
    epochs = 10000
    batch_size = 100
    model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size,
              validation_data=(val_x, val_y), callbacks=[tb, es])


if __name__ == '__main__':
    main()