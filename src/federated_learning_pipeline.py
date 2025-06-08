import tensorflow as tf
import tensorflow_federated as tff
from tensorflow.keras.applications import MobileNetV2
from tensorflow_federated.python.learning.models import keras_utils
from tensorflow_federated.python.learning.optimizers import build_sgdm
from sklearn.model_selection import train_test_split
import pandas as pd
import sys 
import os
from sklearn.model_selection import KFold

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# add project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.federated_learning_testing_utils import (
    test_data_load,
    test_one_batch
)

def load_evaluation_data(evaluation_data_path):
    """
    Load and preprocess the evaluation dataset.
    Args:
        evaluation_data_path: Path to the evaluation dataset CSV file.
    Returns:
        A TensorFlow dataset.
    """
    df = pd.read_csv(evaluation_data_path)

    # extract features and labels
    image_paths = df['Path'].values
    labels = df[['Cardiomegaly', 'Pneumonia', 'Lung Opacity', 'Edema', 'Consolidation']].values
    df['Path'] = df['Path'].str.replace('CheXpert-v1.0/valid', 'chexlocalize/CheXpert/val', regex=False)

    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))

    # resize the images to adjust to model
    dataset = dataset.map(lambda img_path, label: (
        tf.image.resize(
            tf.image.decode_jpeg(tf.io.read_file(img_path), channels=3), [224, 224]
        ), tf.cast(label, tf.float32)
    ))

    dataset = dataset.batch(32).shuffle(buffer_size=len(df))

    return dataset

def load_client_data_kfold(client_data_paths, num_folds=5):
    client_folds = {}

    for idx, path in enumerate(client_data_paths):
        all_csvs = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(".csv")]
        combined_df = pd.concat((pd.read_csv(f) for f in all_csvs), ignore_index=True)

        # ['Cardiomegaly', 'Pneumonia', 'Lung Opacity', 'Edema', 'Consolidation']
        labels = combined_df[['Pneumonia']].values.astype('float32')
        image_paths = combined_df['Augmented_Path'].values

        kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
        client_fold_data = []

        for train_idx, test_idx in kf.split(image_paths):
            img_train, img_test = image_paths[train_idx], image_paths[test_idx]
            y_train, y_test = labels[train_idx], labels[test_idx]

            def make_dataset(images, labels):
                ds = tf.data.Dataset.from_tensor_slices((images, labels))
                ds = ds.map(lambda img_path, features: (
                    tf.image.resize(
                        tf.image.decode_jpeg(tf.io.read_file(img_path), channels=3), [224, 224]
                    ), features
                ))
                return ds.batch(32)

            client_fold_data.append((
                make_dataset(img_train, y_train),
                make_dataset(img_test, y_test)
            ))

        client_folds[f"Client_{idx + 1}"] = client_fold_data

    return client_folds

def run_kfold_federated_training(client_folds, num_folds=5, rounds=3):
    for fold in range(num_folds):
        print(f"\n========== Fold {fold + 1}/{num_folds} ==========\n")

        # Gather data for this fold
        client_train_data = {
            client_id: folds[fold][0] for client_id, folds in client_folds.items()
        }
        client_test_data = {
            client_id: folds[fold][1] for client_id, folds in client_folds.items()
        }

        # Initialize process and train
        iterative_process = initialize_federated_process()
        final_state = train_federated_model(iterative_process, client_train_data, rounds=rounds)

        # Evaluate
        evaluate_on_client_test_sets(final_state, client_test_data, iterative_process)

def model_fn():
    keras_model = tf.keras.Sequential([
        MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3)),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return keras_utils.from_keras_model(
        keras_model=keras_model,
        # input_spec=(
        #     tf.TensorSpec(shape=(None, 224, 224, 3), dtype=tf.float32),
        #     tf.TensorSpec(shape=(None, 5), dtype=tf.float32),
        # ),
        input_spec={
            'x': tf.TensorSpec(shape=(None, 224, 224, 3), dtype=tf.float32),
            'y': tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
        },
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
        metrics=[tf.keras.metrics.BinaryAccuracy()]
    )


def initialize_federated_process():
    """
    Create a federated learning process using TFF.
    """
    iterative_process = tff.learning.algorithms.build_weighted_fed_avg(
        model_fn=model_fn,
        client_optimizer_fn=build_sgdm(learning_rate=0.02),
        server_optimizer_fn=build_sgdm(learning_rate=1.0)
    )
    return iterative_process

def train_federated_model(iterative_process, client_datasets, rounds=3):
    """
    Train the federated model across multiple rounds.
    Args:
        iterative_process: The TFF iterative process.
        client_datasets: Dictionary of client datasets.
        rounds: Number of training rounds.
    Returns:
        Final server state.
    """
    state = iterative_process.initialize()

    for round_num in range(1, rounds + 1):
        client_data = [client_datasets[client_id] for client_id in client_datasets]
        state, metrics = iterative_process.next(state, client_data)
        binary_accuracy = metrics.get('client_work', {}).get('train', {}).get('binary_accuracy', None)
        print(f'Round {round_num}, Accuracy/Conversion: {binary_accuracy}')

    return state

def evaluate_on_client_test_sets(final_state, client_test_data, iterative_process):
    """
    Evaluate the global model on each client's test dataset.
    """
    keras_model = model_fn()._keras_model

    # Load trained weights into the model
    model_weights = iterative_process.get_model_weights(final_state)
    model_weights.assign_weights_to(keras_model)

    # Compile the model
    keras_model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
        metrics=[tf.keras.metrics.BinaryAccuracy()]
    )

    print("\n--- Client Test Set Evaluation ---")
    for client_id, test_dataset in client_test_data.items():
        print(f"Evaluating on {client_id}:")
        results = keras_model.evaluate(test_dataset, verbose=1)
        print(f"Loss: {results[0]}, Accuracy: {results[1]}\n")

# load evaluation data
evaluation_data_path = "chexlocalize/CheXpert/val_labels.csv"
evaluation_dataset = load_evaluation_data(evaluation_data_path)

# load client data
client_data_paths = [
    "output/clients/Client_1_data.csv",
    "output/clients/Client_2_data.csv",
    "output/clients/Client_3_data.csv",
    "output/clients/Client_4_data.csv",
]

num_folds = 5

client_folds = load_client_data_kfold(client_data_paths, num_folds=num_folds)

run_kfold_federated_training(client_folds, num_folds=num_folds, rounds=3)


