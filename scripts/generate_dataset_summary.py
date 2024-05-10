import multiprocessing
import os
import pathlib
import random
import sys

import h5py
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

sys.path.append(str(pathlib.Path(__file__).parent.parent.absolute()))
num_cores = multiprocessing.cpu_count()
random.seed(0)

T_INIT = 10
T_PRED = 40
T_FINAL = 91 - T_PRED
OCC_DIS_MAX = 30 
summary_name = 'dataset_summary'

def process_file(save_path, path, filename, mode):

    if not filename.endswith('.h5'):
        return
    
    with h5py.File(os.path.join(path, filename), 'r') as f:
        trajectories_tensor = f['/objects/trajectories'][:]
        perspective_ids_tensor = f['/occlusions/object_id_perspective'][:]
        occluding_object_ids_tensor = f['/occlusions/occluding_object_id'][:]
        occluded_object_ids_tensor = f['/occlusions/occluded_object_ids'][:]

    positive_samples = []
    negative_samples = []
    scene = int(filename.split('-')[1])
    sample = int((filename.split('_')[-1]).split('.')[0])

    for t in range(T_INIT, T_FINAL):
        n_valid_perspectives =  perspective_ids_tensor[t].shape[0]

        for perspective_idx in range(n_valid_perspectives):
            obj_idx = perspective_ids_tensor[t, perspective_idx]
            if obj_idx == -1:
                break
            n_occluding_objects = occluding_object_ids_tensor[t, perspective_idx].shape[0]

            for occlusion_idx in range(n_occluding_objects):
                occluding_object_id = int(occluding_object_ids_tensor[t, perspective_idx, occlusion_idx]) #why 0.0?

                if occluding_object_id == -1:
                    break

                pos_ego = trajectories_tensor[obj_idx, t, :2]
                pos_occluding_object = trajectories_tensor[occluding_object_id, t, :2]

                if np.linalg.norm(pos_ego - pos_occluding_object) > OCC_DIS_MAX:
                    continue
                else:
                    pass

                objects_to_be_inferred = occluded_object_ids_tensor[t, perspective_idx, occlusion_idx]
                objects_to_be_inferred = objects_to_be_inferred[objects_to_be_inferred != -1]

                info = np.array([t, obj_idx, occlusion_idx])

                if objects_to_be_inferred.shape[0] > 0:
                    positive_samples.append(info)
                else:
                    negative_samples.append(info)

    info = np.array([len(positive_samples), len(negative_samples)])
    # print(f'Scene: {scene}, sample: {sample}, positive samples: {len(positive_samples)}, negative samples: {len(negative_samples)}')

    np.save(os.path.join(save_path, f'{summary_name}/{scene}_{sample}_dataset_summary_info_{mode}.npy'), info)
    np.save(os.path.join(save_path, f'{summary_name}/{scene}_{sample}_dataset_summary_positive_samples_{mode}.npy'), positive_samples)
    np.save(os.path.join(save_path, f'{summary_name}/{scene}_{sample}_dataset_summary_negative_samples_{mode}.npy'), negative_samples)

import argparse

def main(): 
    parser = argparse.ArgumentParser(description='Generate dataset summary.')
    parser.add_argument('--data_path', type=str, required=True, help='Dataset path.')
    args = parser.parse_args()

    dataset_path = args.data_path

    for mode in ['training', 'validation']:
        path = os.path.join(dataset_path, mode)
        summary_path = os.path.join(path, summary_name)
        if not os.path.exists(summary_path):
            os.makedirs(summary_path)
        for scene in tqdm(os.listdir(path)):
            scene_path = os.path.join(path, scene)
            all_files = os.listdir(scene_path)
     
            processed_list = Parallel(n_jobs = num_cores - 1)(delayed(process_file)(path, scene_path, i, mode) for i in all_files)

if __name__ == "__main__":
    main()


