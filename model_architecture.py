import os

import tensorflow as tf
from numpy import exp
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.models import Sequential


def create_model(
    total_words, embedding_size, hidden_size, max_sequence_length, optimizer
):
    model = Sequential()
    model.add(
        Embedding(total_words, embedding_size, input_length=max_sequence_length - 1)
    )
    model.add(LSTM(hidden_size))
    model.add(Dense(total_words, activation="softmax"))
    model.compile(
        loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"]
    )
    return model


# Function to decay the learning rate
def lr_decay_function(epoch, lr):
    min_lr = float(os.getenv("MINIMUM_LEARNING_RATE"))  # Set the minimum learning rate
    if epoch < int(os.getenv("NUMBER_EPOCHS_BEFORE_DECAYING")):
        return lr
    else:
        new_lr = lr * exp(
            -0.1
        )  # Decays the learning rate by 1% every epoch after the 10th
        return max(new_lr, min_lr)


def get_all_callbacks():
    # Define the learning rate scheduler for learning rate decay
    lr_scheduler = LearningRateScheduler(lr_decay_function)

    checkpoint_path = "model/model.keras"
    callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path, verbose=1, save_best_only=True
    )

    return [lr_scheduler, callback]
