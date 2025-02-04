import pathlib
import sys

sys.path.append(str(pathlib.Path(__file__).parent.parent.absolute()))
import multiprocessing
import os
import pickle

from joblib import Parallel, delayed
from sceneinformer.utils.waymo_utils import return_scenario_list

num_cores = multiprocessing.cpu_count()

def collect_data(input_path, output_path):
    scenario_list = return_scenario_list(input_path)
    scenarios_data = []

    for s_idx, single_scenario in enumerate(scenario_list):
        if single_scenario == None: 
            continue
        scenarios_data.append(single_scenario)

    with open(output_path, 'wb') as file:
        pickle.dump(scenarios_data, file) 
    

def main():
    import argparse


    parser = argparse.ArgumentParser(description='Collect raw data.')
    parser.add_argument('--src_path', type=str, required=True, help='Source path for the data.')
    parser.add_argument('--out_path', type=str, required=True, help='Output path for the data.')
    parser.add_argument('--n_cores', type=int, default=5, help='Number of cores to use for parallel processing.')
    args = parser.parse_args()

    SRC_PATH = args.src_path
    OUT_PATH = args.out_path
    n_cores = args.n_cores

    for mode in ['training', 'validation']: 

        DATA_PATH = os.path.join(SRC_PATH, mode)
        OUTPUT_PATH = os.path.join(OUT_PATH, mode)

        if not os.path.exists(OUTPUT_PATH):
            os.makedirs(OUTPUT_PATH)

        files = os.listdir(DATA_PATH)
        inputs = []
        from tqdm import tqdm

        for file in tqdm(files):
            input_path = os.path.join(DATA_PATH, file)
            output_path = os.path.join(OUTPUT_PATH, file)  
            inputs.append((input_path, output_path))

        processed_list = Parallel(n_jobs = n_cores)(delayed(collect_data)(i[0], i[1]) for i in tqdm(inputs)) # 5


if __name__ == '__main__':
    main()
