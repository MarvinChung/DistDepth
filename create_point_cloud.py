import argparse
import numpy as np
import os
import re
from PIL import Image

# ScanNet camera intrinsic parameters
fx = 1169.621094
fy = 1167.105103
centerX = 646.295044
centerY = 489.927032

def generate_pointcloud(rgb_file, depth_file, pose_file):
    """
    Generate points from a color image, a depth array, and a pose matrix,
    applying the extrinsic pose matrix.

    Inputs:
    rgb_file -- filename of the color image
    depth_file -- filename of the depth array (NumPy .npy file)
    pose_file -- filename of the pose (extrinsic) matrix

    Returns:
    points -- list of point strings to be written to the PLY file
    """
    # Load RGB image
    rgb = Image.open(rgb_file).convert('RGB')
    rgb_pixels = rgb.load()

    # Load depth data from NumPy array
    depth = np.load(depth_file)

    # If necessary, flip or transpose depth to align with the RGB image
    # Adjust these operations based on your data alignment
    # For example:
    # depth = depth.T
    # depth = np.flip(depth, axis=0)

    # Load pose matrix
    pose = np.loadtxt(pose_file)

    # Prepare point cloud data
    points = []

    width, height = rgb.size
    depth_height, depth_width = depth.shape
    # Ensure the depth and RGB images have the same dimensions
    if (width, height) != (depth_width, depth_height):
        print(f"Dimension mismatch between RGB image and depth map for {rgb_file}")
        # Optionally resize depth to match RGB dimensions
        # depth = cv2.resize(depth, (width, height), interpolation=cv2.INTER_NEAREST)
        # Or adjust code accordingly
        return []

    for v in range(height):
        for u in range(width):
            Z = depth[v, u]
            if Z <= 0 or np.isinf(Z) or np.isnan(Z):
                continue
            X = (u - centerX) * Z / fx
            Y = (v - centerY) * Z / fy
            point_cam = np.array([X, Y, Z, 1.0])

            # Apply extrinsic matrix to transform point to world coordinates
            point_world = pose @ point_cam

            color = rgb_pixels[u, v]
            points.append("%f %f %f %d %d %d 0\n" % (
                point_world[0], point_world[1], point_world[2],
                color[0], color[1], color[2]))

    return points

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate a combined point cloud from images, depth maps, and poses.")
    parser.add_argument('--image_folder', required=True, help='Path to the folder containing RGB images.')
    parser.add_argument('--depth_folder', required=True, help='Path to the folder containing depth .npy files.')
    parser.add_argument('--pose_folder', required=True, help='Path to the folder containing pose files.')
    parser.add_argument('--output_file', required=True, help='Path to save the combined point cloud PLY file.')
    parser.add_argument('--skip_idx', default=5, type=int)

    args = parser.parse_args()

    image_folder = args.image_folder
    depth_folder = args.depth_folder
    pose_folder = args.pose_folder
    output_file = args.output_file

    # Function to extract numeric index from filename
    def extract_index(filename):
        # Use regular expression to find numeric part
        match = re.search(r'\d+', filename)
        if match:
            return int(match.group())
        else:
            return None

    # Build a mapping from index to file paths for images, depths, and poses
    image_files = {}
    for f in os.listdir(image_folder):
        if f.endswith('.jpg') or f.endswith('.png'):
            idx = extract_index(f)
            if idx is not None:
                image_files[idx] = os.path.join(image_folder, f)

    depth_files = {}
    for f in os.listdir(depth_folder):
        if f.endswith('.npy'):
            idx = extract_index(f)
            if idx is not None:
                depth_files[idx] = os.path.join(depth_folder, f)

    pose_files = {}
    for f in os.listdir(pose_folder):
        if f.endswith('.txt'):
            idx = extract_index(f)
            if idx is not None:
                pose_files[idx] = os.path.join(pose_folder, f)

    # Find common indices
    common_indices = set(image_files.keys()) & set(depth_files.keys()) & set(pose_files.keys())
    if not common_indices:
        print("No matching files found across image, depth, and pose folders.")
        exit(1)

    all_points = []
    total_points = 0

    # Process files with matching indices
    for idx in sorted(common_indices)[::args.skip_idx]:
        rgb_file = image_files[idx]
        depth_file = depth_files[idx]
        pose_file = pose_files[idx]

        print(f"Processing index {idx}...")

        points = generate_pointcloud(rgb_file, depth_file, pose_file)
        all_points.extend(points)
        total_points += len(points)

    # Write all points to a single PLY file
    with open(output_file, "w") as file:
        file.write('''ply
format ascii 1.0
element vertex %d
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
property uchar alpha
end_header
%s
''' % (total_points, "".join(all_points)))

    print(f"Combined point cloud saved to {output_file}")
