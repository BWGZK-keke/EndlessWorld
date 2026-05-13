#!/bin/bash

# Define the model list
model="ours"

# Define the dimension list
dimensions=("subject_consistency" "background_consistency" "aesthetic_quality" "imaging_quality" "object_class" "multiple_objects" "color" "spatial_relationship" "scene" "temporal_style" "overall_consistency" "human_action" "temporal_flickering" "motion_smoothness" "dynamic_degree" "appearance_style")

# Corresponding folder names
folders=("subject_consistency" "scene" "overall_consistency" "overall_consistency" "object_class" "multiple_objects" "color" "spatial_relationship" "scene" "temporal_style" "overall_consistency" "human_action" "temporal_flickering" "subject_consistency" "subject_consistency" "appearance_style")

# Base path for videos (folder containing the generated VBench clips)
base_path='outputs/endlessworld_vbench/' # TODO: change to local path

# Loop over each model
# Loop over each dimension
for i in "${!dimensions[@]}"; do
    # Get the dimension and corresponding folder
    dimension=${dimensions[i]}
    folder=${folders[i]}

    # Construct the video path
    videos_path="${base_path}"
    echo "$dimension $videos_path"

    # Run the evaluation script
CUDA_VISIBLE_DEVICES=2 python evaluate.py --videos_path $videos_path --dimension $dimension
done
