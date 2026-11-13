import os
import glob
import trimesh
import argparse
import numpy as np
from joblib import Parallel, delayed

def get_point_cloud(mesh):
    sampled_points, face_indices = trimesh.sample.sample_surface(mesh, 100000)
    normals = mesh.face_normals[face_indices]

    sampled_points = np.concatenate([sampled_points, normals], axis=-1)

    return sampled_points

def join_scene(scene):
    geometries = list(scene.geometry.values())
    combined_mesh = trimesh.util.concatenate(geometries)

    return combined_mesh

def merge_and_normalize_obj_files(item, output_directory):

    try:
        point_cloud_file = os.path.join(output_directory, item.split('/')[-1].replace('.glb','.npy'))
        scene = trimesh.load(item)
        mesh = join_scene(scene)

        # Sample 100,000 points from the mesh surface
        sampled_points, face_indices = trimesh.sample.sample_surface(mesh, 100000)
        normals = mesh.face_normals[face_indices]
        
        sampled_points = np.concatenate([sampled_points, normals], axis=-1)
        # Save the sampled point cloud to an NPY file
        np.save(point_cloud_file, np.array(sampled_points))
    except Exception as e:
        print(e)

def main(args):
    data_path  = args.data_path

    cate_lists = os.listdir(data_path)
    cate_lists = [cate for cate in cate_lists if os.path.isdir(os.path.join(data_path, cate))]
    
    
    input_file_lists = []
    output_file_lists = []
    
    for each_cate in cate_lists:
        cate_path = os.path.join(data_path, each_cate)
        
        obj_file_lists = os.listdir(cate_path)
        obj_file_lists = [obj_file for obj_file in obj_file_lists if os.path.isdir(os.path.join(cate_path, obj_file))]
        
        for each_obj in obj_file_lists:
            obj_path = os.path.join(cate_path, each_obj)
            os.listdir(obj_path)
            if os.path.exists(os.path.join(obj_path, "Preprocessed0111","scan.npy")):
                continue
            elif os.path.exists(os.path.join(obj_path, "Preprocessed0111","scan.glb")):
                input_file_lists.append(os.path.join(obj_path, "Preprocessed0111","scan.glb"))
                output_file_lists.append(os.path.join(obj_path, "Preprocessed0111"))
            else:
                out_path = os.path.join(obj_path, "Preprocessed0111","scan.glb")
                print(f"Skipping {out_path} as no scan.glb found.")
    
    print("There are total ", len(input_file_lists), " objects to process.")
    
    Parallel(n_jobs=8, backend='multiprocessing', verbose=10)(
        delayed(merge_and_normalize_obj_files)(item, output_path) for item, output_path in zip(input_file_lists, output_file_lists))
    
    # items = sorted(glob.glob(f"{data_path}/*"))
    # print(items)
    
    # Parallel(n_jobs=8, backend='multiprocessing', verbose=10)(
    #     delayed(merge_and_normalize_obj_files)(item, output_path) for item in items)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to the object file",
    )

    # parser.add_argument(
    #     "--output_path",
    #     type=str,
    #     required=True,
    #     help="Path to save the object file",
    # )

    args = parser.parse_args()
    main(args)
