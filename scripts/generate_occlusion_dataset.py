import pathlib
import sys

sys.path.append(str(pathlib.Path(__file__).parent.parent.absolute()))

import os

import h5py
from joblib import Parallel, delayed
from sceneinformer.utils.dataset_processing import (
    flattened_data_structure, load_pickle,
    recursively_save_dict_contents_to_group)
from sceneinformer.utils.scene_processing import (get_objects, get_occlusions,
                                                  get_road)
from tqdm import tqdm


def process_scenario(name, data_dir, output_dir):
    path = os.path.join(data_dir, name)
    scenario_list = load_pickle(path)
    for s_idx, single_scenario in enumerate(scenario_list):
        if single_scenario == None: 
            return
        
        print(f'Processing {name}_{s_idx}')
        
        roads = single_scenario['roads']
        objects = single_scenario['objects']
        traffic_lights = single_scenario['tl_states']

        # We assume we can predict occlusion & trajectories from the perspectives of "tracks_to_predict" + "sdc".
        # It could be further extended to more agents if more data is needed.
        valid_perspectives = single_scenario["tracks_to_predict"]
        valid_perspectives.append(single_scenario[ "sdc_track_index"])
        
        occlusions_per_timestep = []

        objects_to_be_saved = get_objects(objects)
        maps_to_be_saved = get_road(roads)
        
        for t in range(0,91,1):    
            occlusions = get_occlusions(objects_to_be_saved, t, valid_perspectives)
            occlusions_per_timestep.append(occlusions)

        data = {
            'objects': objects_to_be_saved,
            'maps': maps_to_be_saved,
            'valid_perspectives': valid_perspectives,
            # 'traffic_lights': traffic_lights_to_be_saved, #Traffic lights label are mixed quality so we ignore them for now. 
            'occlusions': occlusions_per_timestep,
        }

        scene_name = name.split('-')[1]
        filename = f'{name}_{s_idx}.h5'

        data = flattened_data_structure(data)
        output_file_dir = os.path.join(output_dir, scene_name)
        if not os.path.exists(output_file_dir):
            os.makedirs(output_file_dir)
        output_file_name = os.path.join(output_file_dir, filename)

    
        with h5py.File(output_file_name, 'w') as hf:
            recursively_save_dict_contents_to_group(hf,'', data)

        ### For debuging/testing:
        # if s_idx > 2:
        #     break
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate vectorized dataset.')
    parser.add_argument('--data_dir', type=str, required=True, help='Data directory.')
    parser.add_argument('--out_dir', type=str, required=True, help='Output directory.')
    parser.add_argument('--n_cores', type=int, default=10, help='Number of cores to use for parallel processing.')

    args = parser.parse_args()

    DATA_DIR = args.data_dir
    OUT_DIR = args.out_dir
    n_cores = args.n_cores

    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)

    for mode in ['training', 'validation']: 
        data_dir = os.path.join(DATA_DIR, mode)
        save_dir = os.path.join(OUT_DIR, mode)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        files = os.listdir(data_dir)
        results = Parallel(n_jobs=n_cores)(delayed(process_scenario)(name, data_dir, save_dir) for name in tqdm(files))