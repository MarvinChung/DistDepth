#!/bin/bash

# Folder path passed as argument
INPUT_FOLDER="./scannet0241/scene0241_00/exported/color"

# Output folder path (can be modified as needed)
OUTPUT_FOLDER="./results_scene0241_00"

# Run the Python script
python3 scannet_demo.py --input_folder "$INPUT_FOLDER" --output_folder "$OUTPUT_FOLDER"
