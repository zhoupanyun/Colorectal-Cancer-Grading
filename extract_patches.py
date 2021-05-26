import os
import glob
import cv2
import h5py
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt



def extract_patches(image, patch_size, strides, offset=0):

    x, y = image.shape[:-1]

    grid_x = list(range(offset, x-patch_size, strides)) + [x-patch_size]
    grid_y = list(range(offset, y-patch_size, strides)) + [y-patch_size]

    indices = np.meshgrid(grid_x, grid_y, indexing='ij')
    indices = np.transpose(indices, (1,2,0)).reshape(-1, 2)

    patches = np.array([image[x:x+patch_size, y:y+patch_size] for (x,y) in indices])

    return patches, indices


def main(config):

    # List the image files and labels
    images = sorted(glob.glob(f'{config["data_dir"]}/*.png'))
    labels = [int(x.split('/')[-1].split('_')[0][-1]) for x in images]

    for file_path in images:

        file_name  = file_path.split('/')[-1].split('.')[0]

        # Read image
        image = np.array(Image.open(file_path))

        # Extract patches
        patches, indices = extract_patches(image, config["patch_size"], config["strides"], config["offset"])

        # Save patches
        save_path = f'{config["save_dir"]}/{file_name}.h5'
        with h5py.File(save_path, "w") as f:
            _ = f.create_dataset(name='data', data=patches, compression="gzip")

        print(f'saved: {len(patches):05} patches extracted from {file_name}', flush=True)



if __name__ == "__main__":

    config = {
        "data_dir": '/storage/images',
        "save_dir": '/storage/patches',
        "patch_size": 512,
        "strides": 256,
        "offset": 0
    }

    # Create the directories
    if not os.path.exists(config["save_dir"]):
        os.makedirs(config["save_dir"])

    main(config)
