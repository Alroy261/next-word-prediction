import logging
import os

from dotenv import load_dotenv
from tensorflow.keras.optimizers import Adam

from model_architecture import create_model, get_all_callbacks
from utils.data_preprocess import preprocess
from utils.tokenization import tokenize
from utils.train_test_data import (
    create_feature_and_label,
    get_n_gram_phrases,
    get_pad_sequences,
    get_train_validation_test_data,
)

dotenv_path = "config.env"

# Load environment variables from the specified file
load_dotenv(dotenv_path=dotenv_path)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(filename)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def main():
    # 1. Get the preprocessed data, with an extra column called `cleaned`
    data_path = "phrases_data.txt"
    data = preprocess(data_path)
    logger.info("Data preprocessing completed.")

    # 2. Tokenize all phrases
    tokenizer, total_words = tokenize(data)
    logger.info("Tokenization completed.")

    # 3. Prepare for all input sequences by extracting n-gram phrases
    input_sequences, max_sequence_length = get_n_gram_phrases(
        data["cleaned"], tokenizer
    )
    logger.info(
        f"N-gram phrases extracted with a total of {len(input_sequences)} sequences."
    )

    # 4. Pre-padding the sequences to make them of equal length
    input_sequences = get_pad_sequences(input_sequences, max_sequence_length)
    logger.info("Sequence padding completed.")

    # 5. Split the dataset into training, validation, and test sets
    train_data, val_data, test_data = get_train_validation_test_data(input_sequences)

    # 6. Create features and labels (for training set, validation set, and test set)
    train_data, train_labels = create_feature_and_label(train_data, total_words)
    val_data, val_labels = create_feature_and_label(val_data, total_words)
    test_data, test_labels = create_feature_and_label(test_data, total_words)
    logger.info("Training, validation, test data and label creation completed.")

    # Set embedding_size and hidden_size from the environment variables
    embedding_size, hidden_size = int(os.getenv("EMBEDDING_SIZE")), int(
        os.getenv("HIDDEN_SIZE")
    )

    # Define model structure
    logger.info("Initializing model...")
    initial_learning_rate = float(os.getenv("INITIAL_LEARNING_RATE"))
    adam_optimizer = Adam(learning_rate=initial_learning_rate)

    model = create_model(
        total_words,
        embedding_size=embedding_size,
        hidden_size=hidden_size,
        max_sequence_length=max_sequence_length,
        optimizer=adam_optimizer,
    )

    all_callbacks = get_all_callbacks()

    logger.info("Model initialization completed. Training in progress...")

    epochs = int(os.getenv("EPOCHS"))
    batch_size = int(os.getenv("BATCH_SIZE"))
    model.fit(
        train_data,
        train_labels,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(val_data, val_labels),
        callbacks=all_callbacks,
        verbose=1,
    )

    # Finally, evaluate the model on the test set
    test_loss, test_accuracy = model.evaluate(test_data, test_labels)
    logger.info(f"Test loss: {test_loss}")
    logger.info(f"Test accuracy: {test_accuracy}")


if __name__ == "__main__":
    main()
