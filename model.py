import tensorflow as tf
import numpy as np
import os
import random
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import preprocess_input

tfk = tf.keras
tfkl = tfk.layers
# Random seed for reproducibility
seed = 42

random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
tf.compat.v1.set_random_seed(seed)
batch_size = 32
training_dir = './training_data_final/'

labels = ['Species1',  # 0
          'Species2',  # 1
          'Species3',  # 2
          'Species4',  # 3
          'Species5',  # 4
          'Species6',  # 5
          'Species7',  # 6
          'Species8']


def get_generators(batch_size):
    train_datagen = ImageDataGenerator(rescale=1. / 255,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True,
                                       validation_split=0.2)  # set validation split

    train_generator = train_datagen.flow_from_directory(
        training_dir,
        target_size=(96, 96),
        batch_size=batch_size,
        class_mode='categorical',
        subset='training')  # set as training data

    validation_generator = train_datagen.flow_from_directory(
        training_dir,  # same directory as training data
        target_size=(96, 96),
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation')  # set as validation data
    return train_generator, validation_generator


def get_next_batch(generator):
    batch = next(generator)

    image = batch[0]
    target = batch[1]

    print("(Input) image shape:", image.shape)
    print("Target shape:", target.shape)

    # Visualize only the first sample
    image = image[0]
    target = target[0]
    target_idx = np.argmax(target)
    print()
    print("Categorical label:", target)
    print("Label:", target_idx)
    print("Class name:", labels[target_idx])
    fig = plt.figure(figsize=(6, 4))
    plt.imshow(np.uint8(image))

    return batch


# train_ds = tfk.preprocessing.image_dataset_from_directory(
#     directory='./training_data_final/',
#     labels='inferred',
#     color_mode="rgb",
#     label_mode='categorical',
#     batch_size=32,
#     image_size=(96, 96))

# Obtain a data generator with the 'ImageDataGenerator.flow_from_directory' method

def construct_model():
    supernet = tfk.applications.VGG16(
        include_top=False,
        weights="imagenet",
        input_shape=(96, 96, 3)
    )
    supernet.trainable = True

    inputs = tfk.Input(shape=(96, 96, 3))
    x = supernet(inputs)
    x = tfkl.Flatten(name='Flattening')(x)
    x = tfkl.Dropout(0.3, seed=seed)(x)
    x = tfkl.Dense(
        4608,
        activation='relu',
        kernel_initializer=tfk.initializers.HeUniform(seed))(x)
    x = tfkl.Dropout(0.3, seed=seed)(x)
    outputs = tfkl.Dense(
        8,
        activation='softmax',
        kernel_initializer=tfk.initializers.GlorotUniform(seed))(x)
    tl_model = tfk.Model(inputs=inputs, outputs=outputs, name='model')
    tl_model.compile(loss=tfk.losses.CategoricalCrossentropy(), optimizer=tfk.optimizers.Adam(learning_rate=0.0001),
                     metrics='accuracy')
    tl_model.summary()
    return tl_model


def train_model(model, train_gen, valid_gen):
    # Train the model
    tl_history = model.fit_generator(
        generator=train_gen,
        epochs=10,
        validation_data=valid_gen,
        callbacks=[
            tfk.callbacks.EarlyStopping(monitor='val_accuracy', mode='max', patience=10, restore_best_weights=True)]
    ).history
    # Plot the training
    model.save("./weights")
    plt.figure(figsize=(15, 5))
    plt.plot(tl_history['loss'], alpha=.3, color='#4D61E2', linestyle='--')
    plt.plot(tl_history['val_loss'], label='Transfer Learning', alpha=.8, color='#4D61E2')
    plt.legend(loc='upper left')
    plt.title('Categorical Crossentropy')
    plt.grid(alpha=.3)

    plt.figure(figsize=(15, 5))
    plt.plot(tl_history['accuracy'], alpha=.3, color='#4D61E2', linestyle='--')
    plt.plot(tl_history['val_accuracy'], label='Transfer Learning', alpha=.8, color='#4D61E2')
    plt.legend(loc='upper left')
    plt.title('Accuracy')
    plt.grid(alpha=.3)

    plt.show()


class Model:
    def __init__(self, path):
        pass

    def predict(self):
        pass


if __name__ == '__main__':
    train_gen, valid_gen = get_generators(batch_size=batch_size)
    # batch = get_next_batch(train_gen)
    # model = construct_model()
    model = tfk.models.load_model("./weights")
    train_model(model, train_gen, valid_gen)
