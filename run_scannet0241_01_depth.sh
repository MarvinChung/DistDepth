#!/bin/bash

# Folder path passed as argument
INPUT_FOLDER="./scannet0241/scene0241_01/exported/color"

# Output folder path (can be modified as needed)
OUTPUT_FOLDER="./results_scene0241_01"

CKPT_FOLDER="./ckpts"

# Run the Python script
python3 scannet_demo.py --input_folder "$INPUT_FOLDER" --output_folder "$OUTPUT_FOLDER" --ckpt_folder "$CKPT_FOLDER"
