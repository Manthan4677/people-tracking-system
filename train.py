import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import random
import cv2
import os
import glob
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import (
    BatchNormalization,
    Conv2D,
    MaxPooling2D,
    Activation,
    Flatten,
    Dropout,
    Dense,
    Input,
)

dataset_dir = "./gender_dataset_face/"
img_dims = (96, 96, 3)
epochs = 100
lr = 1e-3
batch_size = 64

data = []
labels = []

# Load images from the dataset
image_files = [
    f for f in glob.glob(dataset_dir + "/**/*", recursive=True) if not os.path.isdir(f)
]
print(len(image_files))
random.shuffle(image_files)

# Label encoding and data preprocessing
for img in image_files:
    image = cv2.imread(img)
    image = cv2.resize(image, (img_dims[0], img_dims[1]))
    image = img_to_array(image)
    data.append(image)

    label = img.split(os.path.sep)[-2]  # Modify label encoding as needed
    if label == "woman":
        label = 1
    else:
        label = 0
    labels.append(label)

# Data preprocessing
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

# Split dataset for training and validation if the dataset is not empty
if len(data) > 0:
    (trainX, testX, trainY, testY) = train_test_split(
        data, labels, test_size=0.2, random_state=42
    )
    trainY = to_categorical(trainY, num_classes=2)
    testY = to_categorical(testY, num_classes=2)

    # Data augmentation
    aug = ImageDataGenerator(
        rotation_range=25,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest",
    )

    # Model definition
    def build_model(width, height, depth, classes):
        model = Sequential()
        inputShape = (height, width, depth)
        chanDim = -1

        model.add(Input(shape=inputShape))
        model.add(Conv2D(32, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(3, 3)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(1024))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        model.add(Dense(classes))
        model.add(Activation("sigmoid"))

        return model

    model = build_model(
        width=img_dims[0], height=img_dims[1], depth=img_dims[2], classes=2
    )

    # Compile the model
    opt = Adam(learning_rate=lr)
    model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

    # Train the model
    H = model.fit(
        aug.flow(trainX, trainY, batch_size=batch_size),
        validation_data=(testX, testY),
        steps_per_epoch=len(trainX) // batch_size,
        epochs=epochs,
        verbose=True,
    )

    # Save the model
    model.save("model/gender_detection.keras")

    # Plot training/validation loss and accuracy
    plt.style.use("ggplot")
    plt.figure()
    N = epochs
    plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")

    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="upper right")

    # Save plot to disk
    plt.savefig("training_plot.png")
else:
    print("Error: The dataset is empty. Please verify your dataset.")
