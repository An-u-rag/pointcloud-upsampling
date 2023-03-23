import subprocess
import os

image_dataset = "trottier"
max_image_dim = "2000"

# Path to Image Dataset directory
data_dir = os.path.join(os.getcwd(), "data\\", image_dataset)

# Feature Extraction using SIFT
command = [
    "colmap", "feature_extractor",
    "--database_path", data_dir+"\database.db",
    "--image_path", data_dir,
    "--SiftExtraction.use_gpu", "1"
]
subprocess.check_call(command)

# Feature matching
command = [
    "colmap", "exhaustive_matcher",
    "--database_path", data_dir+"\database.db",
    "--SiftMatching.use_gpu", "1"
]
subprocess.check_call(command)

out_sparse_dir = data_dir + "\sparse"
if not os.path.exists(out_sparse_dir):
    os.makedirs(out_sparse_dir)

# Sparse LR Point Cloud Reconstruction
command = [
    "colmap", "mapper",
    "--database_path", data_dir+"\database.db",
    "--image_path", data_dir,
    "--output_path", out_sparse_dir
]
subprocess.check_call(command)

out_sparse_dir_txt = os.path.join(out_sparse_dir, "txt")
if not os.path.exists(out_sparse_dir_txt):
    os.makedirs(out_sparse_dir_txt)

# Convert the generated sparse point cloud to Text File
command = [
    "colmap", "model_converter",
    "--input_path", out_sparse_dir+"\\0",
    "--output_path", out_sparse_dir_txt,
    "--output_type", "TXT"
]
subprocess.check_call(command)

out_dense_dir = data_dir + "\dense"
if not os.path.exists(out_dense_dir):
    os.makedirs(out_dense_dir)
print(out_dense_dir)
# Generate a dense point cloud from the sparse point cloud using MVS
command = [
    "colmap", "image_undistorter",
    "--image_path", data_dir,
    "--input_path", out_sparse_dir+"\\0",
    "--output_path", out_dense_dir,
    "--output_type", "COLMAP",
    "--max_image_size", max_image_dim,
]
subprocess.check_call(command)

command = [
    "colmap", "patch_match_stereo",
    "--workspace_path", out_dense_dir,
    "--workspace_format", "COLMAP",
    "--PatchMatchStereo.geom_consistency", "true"
]
subprocess.check_call(command)

command = [
    "colmap", "stereo_fusion",
    "--workspace_path", out_dense_dir,
    "--workspace_format", "COLMAP",
    "--input_type", "geometric",
    "--output_path", out_dense_dir + "\\fused.ply",
    "--StereoFusion.use_cache", "1"
]
subprocess.check_call(command)
