# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import argparse
import cv2
import matplotlib as mpl
import matplotlib.cm as cm
import numpy as np
import os
import torch
from networks.depth_decoder import DepthDecoder
from networks.resnet_encoder import ResnetEncoder
from utils import output_to_depth
from natsort import natsorted

torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Run depth prediction on a folder of images.")
    parser.add_argument("--input_folder", required=True, help="Path to the folder containing input images.")
    parser.add_argument("--output_folder", default="./results", help="Path to save the output images.")
    parser.add_argument("--ckpt_folder", default="./ckpts", help="Path to the folder containing checkpoints.")
    args = parser.parse_args()

    input_folder = args.input_folder
    output_folder = args.output_folder
    ckpt_folder = args.ckpt_folder

    # Check if the input folder exists
    if not os.path.exists(input_folder):
        raise ValueError(f"Input folder {input_folder} does not exist.")

    # Get list of image files in the input folder
    files = sorted([os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith(".jpg")])

    if not files:
        raise ValueError(f"No image files found in the folder {input_folder}.")

    # Ensure output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    with torch.no_grad():
        print("Loading the pretrained network")
        # Load encoder
        encoder = ResnetEncoder(152, False)
        loaded_dict_enc = torch.load(
            os.path.join(ckpt_folder, "encoder.pth"),
            map_location=device,
        )

        filtered_dict_enc = {
            k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()
        }
        encoder.load_state_dict(filtered_dict_enc)
        encoder.to(device)
        encoder.eval()

        # Load depth decoder
        depth_decoder = DepthDecoder(num_ch_enc=encoder.num_ch_enc, scales=range(4))
        loaded_dict = torch.load(
            os.path.join(ckpt_folder, "depth.pth"),
            map_location=device,
        )
        depth_decoder.load_state_dict(loaded_dict)
        depth_decoder.to(device)
        depth_decoder.eval()

        # Process each image
        for idx, file in enumerate(natsorted(files)):
            raw_img = np.transpose(
                cv2.imread(file, -1)[:, :, :3], (2, 0, 1)
            )
            input_image = torch.from_numpy(raw_img).float().to(device)
            input_image = (input_image / 255.0).unsqueeze(0)

            # Resize to input size
            input_image = torch.nn.functional.interpolate(
                input_image, (256, 256), mode="bilinear", align_corners=False
            )
            features = encoder(input_image)
            outputs = depth_decoder(features)

            out = outputs[("out", 0)]
            out_resized = torch.nn.functional.interpolate(
                out, (raw_img.shape[1], raw_img.shape[2]), mode="bilinear", align_corners=False
            )

            # Convert disparity to depth
            depth = output_to_depth(out_resized, 0.1, 10)
            metric_depth = depth.cpu().numpy().squeeze()

            # Visualization
            normalizer = mpl.colors.Normalize(vmin=0.1, vmax=10.0)
            mapper = cm.ScalarMappable(norm=normalizer, cmap="turbo")
            colormapped_im = (mapper.to_rgba(metric_depth)[:, :, :3] * 255).astype(np.uint8)

            # Save the output image
            output_file = os.path.join(output_folder, f"{idx:04d}.png")
            cv2.imwrite(output_file, colormapped_im[:, :, [2, 1, 0]])

            np_output_file = os.path.join(output_folder, f"{idx:04d}.npy")
            np.save(np_output_file, metric_depth)

            print(f"Processed {file} -> {output_file} (for visualization) and {np_output_file} (metric depth)")
