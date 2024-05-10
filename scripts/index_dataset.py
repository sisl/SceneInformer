import multiprocessing
import os

import numpy as np
from tqdm import tqdm

num_cores = multiprocessing.cpu_count()
import argparse


#Load a pickle file with the dataset summary
def process(path):
    num_positive_samples = 0
    num_negative_samples = 0

    files = os.listdir(os.path.join(path, 'dataset_summary'))
    mode = path.split('/')[-1]
    for filename in tqdm(files):

        if filename.split('_')[-1] != f'{mode}.npy' or filename.split('_')[-2] != 'info':
            continue

        data = np.load(os.path.join(path, 'dataset_summary', filename))
        scene, sample = filename.split('_')[0:2]
        sample = int(sample)
        scene = int(scene)

        num_positive_samples += data[0]
        num_negative_samples += data[1]

    print(f'Detected {num_positive_samples} positive samples and {num_negative_samples} negative samples.')

    total = num_positive_samples + num_negative_samples

    idx_to_bin_positive = np.empty((num_positive_samples,4), dtype = np.int32)
    idx_to_bin_negative = np.empty((num_negative_samples,4), dtype = np.int32)
    num_positive_samples_ = 0
    num_negative_samples_ = 0

    bin_idx = 0
    for filename in tqdm(files):
        if filename.split('_')[-1] != f'{mode}.npy' or filename.split('_')[-2] != 'info':
            continue

        data = np.load(os.path.join(path, 'dataset_summary', filename))
        scene, sample = filename.split('_')[0:2]
        sample = int(sample)
        scene = int(scene)

        idx_to_bin_positive[num_positive_samples_:num_positive_samples_+data[0],:] = np.array([bin_idx, num_positive_samples_, scene, sample])
        idx_to_bin_negative[num_negative_samples_:num_negative_samples_+data[1],:] = np.array([bin_idx, num_negative_samples_, scene, sample])
        num_positive_samples_ += data[0]
        num_negative_samples_ += data[1]
        bin_idx += 1

    np.save(os.path.join(path, 'dataset_summary', f'idx_to_bin_positive_{mode}.npy'), idx_to_bin_positive)
    np.save(os.path.join(path, 'dataset_summary', f'idx_to_bin_negative_{mode}.npy'), idx_to_bin_negative)


def main(): 
    parser = argparse.ArgumentParser(description='Index the dataset.')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the dataset.')
    args = parser.parse_args()

    path = args.data_path

    for mode in ['training', 'validation']:
        process(os.path.join(path,mode))

if __name__ == "__main__":
    main()