import argparse
import numpy as np
import os
import re
from PIL import Image
import rerun as rr  # Import the Rerun library
from scipy.spatial.transform import Rotation
from tqdm import tqdm  # Import tqdm for progress display

# Initialize Rerun
rr.init("PointCloudVisualization", spawn=True)

# Original ScanNet camera intrinsic parameters
fx = 1169.621094
fy = 1167.105103
centerX = 646.295044
centerY = 489.927032

def generate_pointcloud(rgb, depth_data, pose_file, idx, adjusted_fx, adjusted_fy, adjusted_centerX, adjusted_centerY, use_rerun):
    """
    Generate points from an RGB image, a depth array, and a pose matrix,
    applying the extrinsic pose matrix.

    Inputs:
    rgb -- numpy array of RGB image (already resized and cropped)
    depth_data -- numpy array of depth data (already resized and cropped)
    pose_file -- filename of the pose (extrinsic) matrix
    idx -- index of the current frame (needed for logging)
    adjusted_fx, adjusted_fy -- adjusted focal lengths after resizing
    adjusted_centerX, adjusted_centerY -- adjusted principal point after resizing and cropping

    Returns:
    points -- NumPy array of point coordinates
    colors -- NumPy array of RGB colors
    pose -- the pose matrix
    """
    # Load pose matrix
    pose = np.loadtxt(pose_file)

    # Prepare point cloud data
    points = []
    colors = []

    height, width = depth_data.shape

    # Generate point cloud
    for v in range(height):
        for u in range(width):
            Z = depth_data[v, u]
            if Z <= 0 or np.isinf(Z) or np.isnan(Z):
                continue
            X = (u - adjusted_centerX) * Z / adjusted_fx
            Y = (v - adjusted_centerY) * Z / adjusted_fy
            point_cam = np.array([X, Y, Z, 1.0])

            # Apply extrinsic matrix to transform point to world coordinates
            point_world = pose @ point_cam

            color = rgb[v, u, :]  # Access RGB values
            points.append(point_world[:3])  # Only XYZ, discard homogeneous coordinate
            colors.append(color)

    points = np.array(points)
    colors = np.array(colors)

    if use_rerun:
        # Log the camera position
        keyframe_entry = f"frame_{idx}/camera_position"
        rr.log(
            keyframe_entry,
            rr.Transform3D(
                translation=pose[:3, 3],
                rotation=rr.Quaternion(xyzw=Rotation.from_matrix(pose[:3, :3]).as_quat()))
        )

        # Define adjusted camera intrinsic matrix
        cam_intrinsic_adjusted = np.array(
            [
                [adjusted_fx, 0, adjusted_centerX],
                [0, adjusted_fy, adjusted_centerY],
                [0, 0,  1],
            ]
        )
        rr.log(
            keyframe_entry,
            rr.Pinhole(
                resolution=[width, height],
                image_from_camera=cam_intrinsic_adjusted,
                camera_xyz=rr.ViewCoordinates.RDF,
            )
        )

    return points, colors, pose

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Visualize camera positions and point clouds incrementally using Rerun.")
    parser.add_argument('--image_folder', required=True, help='Path to the folder containing RGB images.')
    parser.add_argument('--depth_folder', required=True, help='Path to the folder containing predicted depth .npy files.')
    parser.add_argument('--pose_folder', required=True, help='Path to the folder containing pose files.')
    parser.add_argument('--ground_truth_depth_folder', required=True, help='Path to the folder containing ground truth depth .png files.')
    parser.add_argument('--ground_truth_depth_scale', type=float, required=True, help='The depth scale of the ground truth depth .png files.')
    parser.add_argument('--output_file', required=True, help='Path to save the combined point cloud PLY file.')
    parser.add_argument('--stride', type=int, default=5, help='Stride for selecting frames.')
    parser.add_argument('--use_ground_truth_depth', action='store_true', help='Use ground truth depth to generate point cloud.')
    parser.add_argument('--use_rerun', action='store_true', help='Use rerun to visualize point cloud.')

    args = parser.parse_args()

    image_folder = args.image_folder
    depth_folder = args.depth_folder
    pose_folder = args.pose_folder
    gt_depth_folder = args.ground_truth_depth_folder
    use_rerun = args.use_rerun

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

    gt_depth_files = {}
    for f in os.listdir(gt_depth_folder):
        if f.endswith('.png'):
            idx = extract_index(f)
            if idx is not None:
                gt_depth_files[idx] = os.path.join(gt_depth_folder, f)

    # Find common indices
    common_indices = set(image_files.keys()) & set(depth_files.keys()) & set(pose_files.keys()) & set(gt_depth_files.keys())
    if not common_indices:
        print("No matching files found across image, depth, pose, and ground truth depth folders.")
        exit(1)

    # Get original image dimensions (assuming all images have the same size)
    sample_image_path = next(iter(image_files.values()))
    with Image.open(sample_image_path) as img:
        original_width, original_height = img.size

    # Adjust camera intrinsic parameters due to resizing
    scale_x = 640 / original_width
    scale_y = 480 / original_height

    # Adjust fx and fy
    adjusted_fx = fx * scale_x
    adjusted_fy = fy * scale_y

    # Adjust centerX and centerY due to scaling
    adjusted_centerX = centerX * scale_x
    adjusted_centerY = centerY * scale_y

    # Define cropping parameters (central crop to 624x468)
    crop_width = 624
    crop_height = 468
    left = (640 - crop_width) // 2
    upper = (480 - crop_height) // 2
    right = left + crop_width
    lower = upper + crop_height

    # Adjust centerX and centerY due to cropping
    adjusted_centerX -= left
    adjusted_centerY -= upper

    all_points = []
    all_colors = []
    total_points = 0

    l1_losses = []

    # Use tqdm to show progress
    t = tqdm(sorted(common_indices)[::args.stride], desc='Processing', unit='frame')

    # Process files with matching indices
    for idx in t:
        rgb_file = image_files[idx]
        depth_file = depth_files[idx]
        pose_file = pose_files[idx]
        gt_depth_file = gt_depth_files[idx]

        # Load and process RGB image
        rgb = Image.open(rgb_file).convert('RGB')
        rgb = rgb.resize((640, 480), Image.BILINEAR)
        rgb = np.array(rgb)
        rgb = rgb[upper:lower, left:right, :]

        # Load and process predicted depth
        predicted_depth = np.load(depth_file)
        predicted_depth_image = Image.fromarray(predicted_depth)
        predicted_depth_image = predicted_depth_image.resize((640, 480), Image.NEAREST)
        predicted_depth = np.array(predicted_depth_image)
        predicted_depth = predicted_depth[upper:lower, left:right]

        # Load and process ground truth depth
        gt_depth = np.array(Image.open(gt_depth_file)).astype(np.float32) / args.ground_truth_depth_scale  # Convert to meters
        gt_depth_image = Image.fromarray(gt_depth)
        gt_depth_image = gt_depth_image.resize((640, 480), Image.NEAREST)
        gt_depth = np.array(gt_depth_image)
        gt_depth = gt_depth[upper:lower, left:right]

        # Decide which depth data to use for point cloud generation
        if args.use_ground_truth_depth:
            depth_data = gt_depth
        else:
            depth_data = predicted_depth

        # Generate point cloud
        points, colors, pose = generate_pointcloud(
            rgb, depth_data, pose_file, idx,
            adjusted_fx, adjusted_fy, adjusted_centerX, adjusted_centerY, use_rerun
        )

        # Compute L1 loss between predicted depth and ground truth depth
        valid_mask = (gt_depth > 0) & (predicted_depth > 0) & \
                     (~np.isnan(gt_depth)) & (~np.isnan(predicted_depth)) & \
                     (~np.isinf(gt_depth)) & (~np.isinf(predicted_depth))

        if np.sum(valid_mask) > 0:
            l1_loss = np.abs(predicted_depth[valid_mask] - gt_depth[valid_mask]).mean()
        else:
            l1_loss = np.nan

        l1_losses.append(l1_loss)

        # Update tqdm with per-image L1 loss
        t.set_postfix(l1_loss=l1_loss)

        # Log the point cloud to Rerun
        rr.log(f"points_{idx}", rr.Points3D(points, colors=colors, radii=0.02))

        all_points.extend(points)
        all_colors.extend(colors)
        total_points += len(points)

    # Write all points to a single PLY file
    with open(args.output_file, "w") as file:
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
''' % total_points)

        for point, color in zip(all_points, all_colors):
            file.write("%f %f %f %d %d %d 0\n" % (
                point[0], point[1], point[2],
                color[0], color[1], color[2]
            ))

    print(f"Combined point cloud saved to {args.output_file}")

    # Compute average L1 loss
    average_l1_loss = np.nanmean(l1_losses)
    print(f"Average L1 Depth Loss: {average_l1_loss:.4f} meters")

    print("Visualization complete.")
