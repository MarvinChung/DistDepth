import argparse
import numpy as np
import os
import re
from PIL import Image
from tqdm import tqdm  # Import tqdm for progress display
from skimage.measure import marching_cubes
import plyfile
from scipy.spatial.transform import Rotation
import trimesh

# Original ScanNet camera intrinsic parameters
fx = 1169.621094
fy = 1167.105103
centerX = 646.295044
centerY = 489.927032

def save_ply(filename, points, colors):
    """
    Save point cloud data to a PLY file.

    Args:
        filename (str): Path to the output PLY file.
        points (np.ndarray): Nx3 array of point coordinates.
        colors (np.ndarray): Nx3 array of RGB colors.
    """
    with open(filename, 'w') as f:
        f.write('ply\n')
        f.write('format ascii 1.0\n')
        f.write(f'element vertex {len(points)}\n')
        f.write('property float x\n')
        f.write('property float y\n')
        f.write('property float z\n')
        f.write('property uchar red\n')
        f.write('property uchar green\n')
        f.write('property uchar blue\n')
        f.write('end_header\n')
        for p, c in zip(points, colors):
            f.write(f"{p[0]} {p[1]} {p[2]} {int(c[0])} {int(c[1])} {int(c[2])}\n")

def create_volume(points, grid_size=256):
    """
    Create a volumetric occupancy grid from point cloud data.

    Args:
        points (np.ndarray): Nx3 array of point coordinates.
        grid_size (int): Resolution of the volumetric grid along each axis.

    Returns:
        volume (np.ndarray): 3D occupancy grid.
        min_coords (np.ndarray): Minimum coordinates along each axis.
        max_coords (np.ndarray): Maximum coordinates along each axis.
    """
    # Remove invalid points (NaN, Inf)
    valid_mask = np.all(np.isfinite(points), axis=1)
    points = points[valid_mask]

    # Check if there are valid points left
    if points.shape[0] == 0:
        print("No valid points to create volume.")
        return None, None, None

    # Compute min and max coordinates
    min_coords = points.min(axis=0)
    max_coords = points.max(axis=0)

    # Compute grid spacing
    grid_spacing = (max_coords - min_coords) / grid_size

    # Avoid division by zero in case of zero grid spacing
    grid_spacing[grid_spacing == 0] = 1e-6

    # Initialize volume
    volume = np.zeros((grid_size, grid_size, grid_size), dtype=np.uint8)

    # Map points to grid indices
    indices = ((points - min_coords) / grid_spacing).astype(int)

    # Remove any indices with invalid values after processing
    valid_indices_mask = (indices >= 0) & (indices < grid_size)
    valid_indices_mask = np.all(valid_indices_mask, axis=1)
    indices = indices[valid_indices_mask]

    if indices.shape[0] == 0:
        print("No valid indices after mapping to grid.")
        return None, None, None

    indices_x = indices[:, 0]
    indices_y = indices[:, 1]
    indices_z = indices[:, 2]

    # Set occupied voxels
    volume[indices_x, indices_y, indices_z] = 1

    return volume, min_coords, max_coords

def create_default_material_file(material_file_path):
    """
    Creates a default material file if it does not exist.

    Args:
        material_file_path (str): Path to the .mtl file to create.
    """
    if not os.path.exists(material_file_path):
        default_material_content = """# default.mtl
newmtl default
Ka 1.000 1.000 1.000
Kd 0.800 0.800 0.800
Ks 0.000 0.000 0.000
d 1.0
illum 2
"""
        with open(material_file_path, 'w') as f:
            f.write(default_material_content)
        print(f"Material file created: {material_file_path}")
    else:
        print(f"Material file already exists: {material_file_path}")


# def convert_volume_to_mesh_obj_with_material(volume, min_coords, max_coords, output_filename, material_name="default", level=0.5):
#     """
#     Convert a volumetric occupancy grid to a mesh and save it as an OBJ file with a material.

#     Args:
#         volume (np.ndarray): 3D occupancy grid.
#         min_coords (np.ndarray): Minimum coordinates along each axis.
#         max_coords (np.ndarray): Maximum coordinates along each axis.
#         output_filename (str): Path to save the mesh OBJ file.
#         material_name (str): Name of the material to include in the OBJ file.
#         level (float): The value of the iso-surface to extract.
#     """
#     # Compute voxel size
#     voxel_size = (max_coords - min_coords) / np.array(volume.shape)

#     # Apply Marching Cubes
#     verts, faces, normals, _ = marching_cubes(
#         volume, level=level, spacing=voxel_size, allow_degenerate=True
#     )

#     # Map vertices back to world coordinates
#     verts += min_coords

#     # Create a Trimesh object
#     mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)

#     # Add material information
#     obj_data = trimesh.exchange.obj.export_obj(mesh, include_color=False)

#     # Add material references to the OBJ file
#     obj_data = f"mtllib {material_name}.mtl\nusemtl {material_name}\n" + obj_data

#     # Save the OBJ file
#     with open(output_filename, "w") as f:
#         f.write(obj_data)

#     print(f"Mesh saved to {output_filename} with material '{material_name}'")

def convert_volume_to_mesh_obj_with_uv(
    volume, min_coords, max_coords, output_filename, material_name="default", level=0.5
):
    """
    Convert a volumetric occupancy grid to a mesh, apply transformations, and save it as an OBJ file with UV coordinates.

    Args:
        volume (np.ndarray): 3D occupancy grid.
        min_coords (np.ndarray): Minimum coordinates along each axis.
        max_coords (np.ndarray): Maximum coordinates along each axis.
        output_filename (str): Path to save the mesh OBJ file.
        material_name (str): Name of the material to use.
        level (float): The value of the iso-surface to extract.
    """
    # Compute voxel size
    voxel_size = (max_coords - min_coords) / np.array(volume.shape)

    # Apply Marching Cubes
    verts, faces, normals, _ = marching_cubes(
        volume, level=level, spacing=voxel_size, allow_degenerate=True
    )

    # 90-degree clockwise rotation about the z-axis
    rotation_z = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])  
    verts = verts @ rotation_z.T
    normals = normals @ rotation_z.T

    # Map back to world coordinates
    verts += min_coords

    # Generate UV coordinates (simple planar projection)
    uv_coords = np.zeros((verts.shape[0], 2))
    uv_coords[:, 0] = (verts[:, 0] - min_coords[0]) / (max_coords[0] - min_coords[0])  # U
    uv_coords[:, 1] = (verts[:, 1] - min_coords[1]) / (max_coords[1] - min_coords[1])  # V

    # Write OBJ file
    with open(output_filename, 'w') as f:
        f.write(f"mtllib {material_name}.mtl\n")
        f.write(f"usemtl {material_name}\n")

        # Write vertices
        for vert in verts:
            f.write(f"v {vert[0]:.6f} {vert[1]:.6f} {vert[2]:.6f}\n")

        # Write normals
        for normal in normals:
            f.write(f"vn {normal[0]:.6f} {normal[1]:.6f} {normal[2]:.6f}\n")

        # Write texture coordinates
        for uv in uv_coords:
            f.write(f"vt {uv[0]:.6f} {uv[1]:.6f}\n")

        # Write faces
        for face in faces:
            v1, v2, v3 = face + 1  # OBJ format indices are 1-based
            f.write(f"f {v1}/{v1}/{v1} {v2}/{v2}/{v2} {v3}/{v3}/{v3}\n")

    print(f"Transformed OBJ file with UV coordinates saved to {output_filename}")


import numpy as np

def generate_pointcloud(
    rgb, depth_data, pose_file, idx, adjusted_fx, adjusted_fy,
    adjusted_centerX, adjusted_centerY, use_rerun, downsample_factor=10
):
    """
    Generate points from an RGB image, a depth array, and a pose matrix,
    applying the extrinsic pose matrix. Includes point cloud downsampling.

    Inputs:
        rgb (np.ndarray): RGB image array (already resized and cropped).
        depth_data (np.ndarray): Depth data array (already resized and cropped).
        pose_file (str): Filename of the pose (extrinsic) matrix.
        idx (int): Index of the current frame (needed for logging).
        adjusted_fx (float): Adjusted focal length fx after resizing.
        adjusted_fy (float): Adjusted focal length fy after resizing.
        adjusted_centerX (float): Adjusted principal point X after resizing and cropping.
        adjusted_centerY (float): Adjusted principal point Y after resizing and cropping.
        use_rerun (bool): Flag to indicate whether to use Rerun for visualization.
        downsample_factor (int): Factor by which to downsample the point cloud.

    Returns:
        points (np.ndarray): Nx3 array of point coordinates (downsampled).
        colors (np.ndarray): Nx3 array of RGB colors (downsampled).
        pose (np.ndarray): 4x4 pose matrix.
    """
    # Load pose matrix (c2w)
    pose = np.loadtxt(pose_file)

    # Prepare point cloud data
    points = []
    colors = []

    height, width = depth_data.shape

    # Generate point cloud
    for v in range(0, height, downsample_factor):  # Downsample by skipping rows
        for u in range(0, width, downsample_factor):  # Downsample by skipping columns
            Z = depth_data[v, u]
            if Z <= 0 or np.isinf(Z) or np.isnan(Z):
                continue
            X = (u - adjusted_centerX) * Z / adjusted_fx
            Y = (v - adjusted_centerY) * Z / adjusted_fy
            point_cam = np.array([X, Y, Z, 1.0])

            # Apply c2w to transform point to world coordinates
            point_world = pose @ point_cam

            color = rgb[v, u, :]  # Access RGB values
            points.append(point_world[:3])  # Only XYZ, discard homogeneous coordinate
            colors.append(color)

    points = np.array(points)
    colors = np.array(colors)

    if use_rerun and idx % 50 == 0:
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

        # Log the point cloud to Rerun
        rr.log(f"points_{idx}", rr.Points3D(points, colors=colors, radii=0.02))

    return points, colors, pose


def generate_mesh(all_points, grid_size, mesh_output_filename, material_name):
    all_points_np = np.array(all_points)

    # Create volumetric grid
    volume, min_coords, max_coords = create_volume(all_points_np, grid_size=grid_size)

    # Convert volume to mesh and save using plyfile
    convert_volume_to_mesh_obj_with_uv(volume, min_coords, max_coords, mesh_output_filename, material_name, level=0.5)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Visualize camera positions and point clouds incrementally.")
    parser.add_argument('--image_folder', required=True, help='Path to the folder containing RGB images.')
    parser.add_argument('--depth_folder', required=True, help='Path to the folder containing predicted depth .npy files.')
    parser.add_argument('--pose_folder', required=True, help='Path to the folder containing pose files.')
    parser.add_argument('--ground_truth_depth_folder', required=True, help='Path to the folder containing ground truth depth .png files.')
    parser.add_argument('--ground_truth_depth_scale', type=float, required=True, help='The depth scale of the ground truth depth .png files.')
    parser.add_argument('--output_folder', type=str, default="output", help='folder saved all output files')
    parser.add_argument('--output_file_prefix', required=True, help='Prefix for output files (point clouds and mesh).')
    parser.add_argument('--stride', type=int, default=5, help='Stride for selecting frames.')
    parser.add_argument('--grid_size', type=int, default=256, help='Resolution for each axis of volume for marching cubes.')
    parser.add_argument('--use_ground_truth_depth', action='store_true', help='Use ground truth depth to generate point cloud.')
    parser.add_argument('--use_rerun', action='store_true', help='Use Rerun to visualize point cloud.')
    parser.add_argument('--material_name', type=str, default="default", help='Create a fake material for mesh')

    args = parser.parse_args()

    image_folder = args.image_folder
    depth_folder = args.depth_folder
    pose_folder = args.pose_folder
    gt_depth_folder = args.ground_truth_depth_folder
    use_rerun = args.use_rerun

    os.makedirs(args.output_folder, exist_ok=True)
    output_prefix_path = os.path.join(args.output_folder, args.output_file_prefix)

    material_save_path = os.path.join(args.output_folder, args.material_name+".mtl")
    create_default_material_file(material_save_path)

    if use_rerun:
        import rerun as rr  # Import Rerun library
        # Initialize Rerun
        rr.init("PointCloudVisualization", spawn=True)

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

    # Define cropping parameters (central crop to 624x468)
    crop_width = 624
    crop_height = 468
    left = (640 - crop_width) // 2
    upper = (480 - crop_height) // 2
    right = left + crop_width
    lower = upper + crop_height

    # Adjust centerX and centerY due to scaling
    adjusted_centerX = centerX * scale_x
    adjusted_centerY = centerY * scale_y

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
    for ct, idx in enumerate(t):
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

        all_points.extend(points)
        all_colors.extend(colors)
        total_points += len(points)

        # Save point cloud for the first image
        if ct % 100 == 0:
            first_ply_filename = output_prefix_path + f"_{idx}_point_cloud.ply"
            save_ply(first_ply_filename, points, colors)
            print(f"Point cloud for the {ct} frame saved to {first_ply_filename}")
            mesh_output_filename = output_prefix_path + f"_{idx}_mesh.obj"
            generate_mesh(all_points, args.grid_size, mesh_output_filename, args.material_name)


    # Compute average L1 loss
    average_l1_loss = np.nanmean(l1_losses)
    print(f"Average L1 Depth Loss: {average_l1_loss:.4f} meters")

    # Generate and save mesh using Marching Cubes
    print("Generating mesh using Marching Cubes...")

    final_ply_filename = output_prefix_path + f"_final_point_cloud.ply"
    save_ply(final_ply_filename, all_points, all_colors)
    mesh_output_filename =  output_prefix_path + "_mesh.obj"
    generate_mesh(all_points, args.grid_size, mesh_output_filename, args.material_name)

    print("Visualization complete.")
