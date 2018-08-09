from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Conv1D, Dropout, MaxPool1D
from keras.utils import to_categorical
from keras.optimizers import SGD
from classify_bubble import get_candidate_signals, is_bubble
import os
from cv2 import imread
import numpy as np


def get_bubble_train_signal(img,
                            signal_len,
                            classifier,
                            threshold_abs=50,
                            min_distance=2):

    signals_x, signals_y, signals_z = get_candidate_signals(img,
                                                            signal_len,
                                                            threshold_abs=threshold_abs,
                                                            min_distance=min_distance,
                                                            smooth_img=False)

    signals_labels = [False] * len(signals_z)
    for idx, (sig_x, sig_y, sig_z) in enumerate(zip(signals_x, signals_y, signals_z)):
        if is_bubble(sig_z, classifier):
            signals_labels[idx] = True

    assert len(signals_labels) == len(signals_z)

    return signals_z, signals_labels


def create_bubbleNet():

    nb_classes = 2
    nb_features = 20
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=4, activation='relu', use_bias=True, input_shape=(nb_features,1)))
    model.add(MaxPool1D(pool_size=2))
    model.add(Conv1D(filters=32, kernel_size=2, activation='relu'))
    model.add(Flatten())
    model.add(Dense(32))
    model.add(Activation('relu'))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    return model


def main():

    NUM_CLASSES = 2

    # get data
    train_images = []
    mess_dir = "data/training/"
    for filename in os.listdir(mess_dir):
        if filename.endswith(".png"):
            train_images.append(imread(os.path.join(mess_dir, filename), 0))

    X_data = y_data = []
    for img in train_images:
        signal, label = get_bubble_train_signal(img,
                                                signal_len=10,
                                                classifier="logistic_regression",
                                                threshold_abs=50,
                                                min_distance=5)
        X_data.append(signal)
        y_data.append(label)

    X_data = np.asarray(X_data, dtype='float32')
    y_data = np.asarray(y_data, dtype='float32')
    X_data /= 255
    y_data /= 255
    y_data = to_categorical(y_data, NUM_CLASSES)

    nb_train = int(0.7 * len(X_data))
    X_train = X_data[0:nb_train]
    y_train = y_data[0:nb_train]
    X_valid = X_data[nb_train:]
    y_valid = y_data[nb_train:]

    # get and train model
    model = create_bubbleNet()

    sgd = SGD(lr=0.01, nesterov=True, decay=1e-6, momentum=0.9)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    nb_epoch = 15
    model.fit(X_train, y_train, nb_epoch=nb_epoch, validation_data=(X_valid, y_valid), batch_size=16)


if __name__ == '__main__':
    main()
