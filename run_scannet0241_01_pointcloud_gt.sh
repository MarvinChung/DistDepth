#!/bin/bash

# Prefix for the scene
PREFIX="scene0241_01"

# Base directory for the dataset
BASE_DIR="/home/marvin/Desktop/dataset/scannet0241"

# Paths using the prefix
IMAGE_FOLDER="$BASE_DIR/$PREFIX/exported/input"
DEPTH_FOLDER="./results_$PREFIX"
POSE_FOLDER="$BASE_DIR/$PREFIX/exported/pose"
GROUND_TRUTH_DEPTH_FOLDER="$BASE_DIR/$PREFIX/exported/depth"
OUTPUT_FOLDER="output_gt"
OUTPUT_FILE_PREFIX="${PREFIX}_gt"

# Ground truth depth scale
GROUND_TRUTH_DEPTH_SCALE=1000.0

# Run the Python script with the dynamic paths
python3 create_point_cloud.py \
  --image_folder "$IMAGE_FOLDER" \
  --depth_folder "$DEPTH_FOLDER" \
  --pose_folder "$POSE_FOLDER" \
  --output_folder "$OUTPUT_FOLDER" \
  --output_file_prefix "$OUTPUT_FILE_PREFIX" \
  --ground_truth_depth_folder "$GROUND_TRUTH_DEPTH_FOLDER" \
  --ground_truth_depth_scale "$GROUND_TRUTH_DEPTH_SCALE" \
  --use_ground_truth_depth \
  --use_rerun