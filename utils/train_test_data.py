import os

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences


def get_n_gram_phrases(data, tokenizer):
    # Create a list called `input_sequences` which generates the n_gram model.
    # use `texts_to_sequences` method to convert each phrase into sequence of numbers
    # For example: "I like apple" becomes [6, 49, 1996]
    number_of_words_needed_for_prediction = int(
        os.getenv("NUMBER_OF_WORDS_NEEDED_FOR_PREDICTION")
    )
    input_sequences = []
    max_sequence_length = 0
    for each_phrase in data:
        token_list = tokenizer.texts_to_sequences([each_phrase])[0]
        token_list_len = len(token_list)
        if token_list_len > max_sequence_length:
            max_sequence_length = token_list_len

        if token_list_len <= number_of_words_needed_for_prediction:
            # if current phrase has less than K words, directly append it to input_sequences
            input_sequences.append(token_list)
            continue

        for i in range(number_of_words_needed_for_prediction, token_list_len):
            # provide at least K words to predict the next word
            n_gram = token_list[: i + 1]
            input_sequences.append(n_gram)

    return input_sequences, max_sequence_length


def get_pad_sequences(input_sequences, max_sequence_length):
    # Pad the sequences to make them of equal length
    input_sequences = np.array(
        pad_sequences(input_sequences, maxlen=max_sequence_length, padding="pre")
    )
    return input_sequences


def get_train_validation_test_data(input_sequences):
    # Split dataset into training, validation, and test sets
    # First, split the whole dataset into 80% training + 20% testing
    train_data, test_data = train_test_split(
        input_sequences, test_size=0.2, random_state=42
    )

    # Next, split the 80% testing set into 50% validation + 50% testing
    val_data, test_data = train_test_split(test_data, test_size=0.5, random_state=42)

    return train_data, val_data, test_data


def create_feature_and_label(input_sequences, total_words):
    # Features are obtained by removing the last element from each sequence
    # Labels are the last element from each sequence
    features, labels = input_sequences[:, :-1], input_sequences[:, -1]
    one_hot_labels = tf.keras.utils.to_categorical(labels, num_classes=total_words)
    return features, one_hot_labels
