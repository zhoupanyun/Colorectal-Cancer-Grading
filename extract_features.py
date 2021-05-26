import os
import sys
import glob
import shutil
import h5py
import numpy as np
import tensorflow as tf



def read_data(file_path):
    with h5py.File(file_path, "r") as f:
        data = f['data'][()]
    return data


def Extractor(weights_path, input_shape):
    base_model = tf.keras.applications.DenseNet121(include_top=False,
                                                   weights=weights_path,
                                                   input_shape=input_shape,
                                                   pooling='avg')
    inputs = base_model.input
    x = base_model.layers[-3].output
    outputs = tf.keras.layers.GlobalAveragePooling2D()(x)
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    return model


def extract_features(model, file_path, batch_size, save_dir):

    file_name = file_path.split('/')[-1][:-3]

    # Read data
    try:
        data = read_data(file_path)
    except:
        print(f'ERROR: could not read {file_name}', flush=True)
        return

    # Extract features
    try:
        data = tf.keras.applications.densenet.preprocess_input(data)
        features = extractor.predict(data, batch_size=batch_size, verbose=0)
    except:
        print(f'ERROR: features not extracted {file_name}', flush=True)
        return

    # Save
    try:
        save_path = f'{save_dir}/{file_name}.h5'
        with h5py.File(save_path, "w") as f:
            _ = f.create_dataset(name='data', data=features, compression="gzip")
        print(f'saved: features extracted from {file_name}', flush=True)
    except:
        print(f'ERROR: could not write {file_name}', flush=True)


def main(config):

    files = sorted(glob.glob(f'{config["data_dir"]}/*.h5'))
    print(f'Extracting features from {len(files)} files...', flush=True)

    extractor = Extractor(config["model_path"], (config["patch_size"],config["patch_size"],3))
    print(f'\nModel loaded!', flush=True)

    print('\nExtracting features...\n', flush=True)
    for c, file_path in enumerate(files):
        print(f'{c:03} \t', end="", flush=True)
        extract_features(extractor, file_path, config["batch_size"], config["save_dir"])

    print('\nComplete!')



if __name__ == "__main__":

    config = {
        "data_dir": '/storage/patches',
        "save_dir": '/storage/features',
        "model_path": '/storage/assets/kimianet.h5',
        "patch_size": 512,
        "strides": 256,
        "batch_size": 16
    }

    # Create the directories
    if not os.path.exists(config["save_dir"]):
        os.makedirs(config["save_dir"])

    main(config)
