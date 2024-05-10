#!/bin/bash

# Define paths (they are all very large so plan accordingly!)
# Raw tfrecords from Waymo Motion
RAW_DATASET_PATH="/home/benksy/Projects/OcclusionInference/small_subset_waymo_dataset" #"/path/to/raw/dataset" 
# Temporary files
TEMP_RAW_DATASET_PATH="/home/benksy/Projects/SceneInformer/small_subset_waymo_dataset_temp" #"/path/to/temp/raw/dataset"
# Final occlusion dataset
OCCLUSION_DATASET_PATH="/home/benksy/Projects/SceneInformer/occlusion_dataset"

# Convert files from tfrecords and h5
python scripts/collect_raw_meas.py --src_path $RAW_DATASET_PATH --out_path $TEMP_RAW_DATASET_PATH
# Process occlusions (slow!, main computation is done here)
python scripts/generate_occlusion_dataset.py --data_dir $TEMP_RAW_DATASET_PATH --out_dir $OCCLUSION_DATASET_PATH
# Create summary and index files for the dataset iterators
python scripts/generate_dataset_summary.py --data_path $OCCLUSION_DATASET_PATH
python scripts/index_dataset.py --data_path $OCCLUSION_DATASET_PATH

# Delete temporary directory
rm -rf $TEMP_RAW_DATASET_PATH