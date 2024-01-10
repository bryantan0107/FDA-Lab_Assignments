# imports here
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models


# define additional functions here
def preprocess(X):
    # Normalize pixel values
    X = X / 255.
    # Reshape to 44x48x3 images
    X = X.values.reshape(-1, 44, 48, 3)
    return X


def create_model(input_shape, num_classes):
    model = models.Sequential([
        layers.Conv2D(64, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(256, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adamax(lr=0.001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


# define train_predict function
def train_predict(X_train, y_train, X_test):
    # check that the input has the correct shape
    assert X_train.shape == (len(X_train), 6336)
    assert y_train.shape == (len(y_train), 1)
    assert X_test.shape == (len(X_test), 6336)

    # Preprocess data
    X_train = preprocess(X_train)
    X_test = preprocess(X_test)

    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42)

    # Define model
    input_shape = X_train[0].shape
    num_classes = len(np.unique(y_train))
    model = create_model(input_shape, num_classes)

    # Train model
    model.fit(X_train, y_train, epochs=30, batch_size=64,
              validation_data=(X_val, y_val))

    # Evaluate model on validation set
    val_loss, val_acc = model.evaluate(X_val, y_val)
    print("Validation accuracy:", val_acc)

    # Make predictions on test set
    y_pred = model.predict(X_test)
    y_pred = np.argmax(y_pred, axis=1)

    # test that the returned prediction has correct shape
    assert y_pred.shape == (len(X_test),) or y_pred.shape == (len(X_test), 1)

    return y_pred


if __name__ == "__main__":
    # load data
    X_train = pd.read_csv("X_train.csv")
    y_train = pd.read_csv("y_train.csv")

    # execute train_predict function
    y_pred = train_predict(X_train, y_train)

    # save predictions to file (change path if necessary)
    np.savetxt("y_pred.csv", y_pred, delimiter=",")