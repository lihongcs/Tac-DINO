import bpy # type: ignore
import numpy as np
import argparse
import sys
import os
import json
import bmesh
from typing import *
from joblib import Parallel, delayed


def scene_root_objects():
    for obj in bpy.context.scene.objects.values():
        if not obj.parent:
            yield obj

def scene_meshes():
    for obj in bpy.context.scene.objects.values():
        if isinstance(obj.data, (bpy.types.Mesh)):
            yield obj

def scene_non_meshes():
    for obj in bpy.context.scene.objects.values():
        if not (obj.data and isinstance(obj.data, bpy.types.Mesh)):
            yield obj


def clear_parents():
    bpy.context.view_layer.objects.active = None
    for obj in scene_root_objects():
        
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)

        bpy.ops.object.parent_clear(type='CLEAR_KEEP_TRANSFORM')

def clear_nonmesh():
    bpy.ops.object.select_all(action='DESELECT')
    for obj in scene_non_meshes():
        bpy.data.objects.remove(obj)

def join_mesh():
    bpy.ops.object.select_all(action='DESELECT')
    objects = list(scene_meshes())
    
    if len(objects) == 0:
        print("Error: No meshes found to join!")
        return
    
    if len(objects) == 1:
        print("Info: Only one mesh found, skipping join.")
        bpy.context.view_layer.objects.active = objects[0]
        objects[0].select_set(True)
        return

    
    for obj in objects:
        obj.select_set(True)

    bpy.context.view_layer.objects.active = objects[0]
    bpy.ops.object.join()

def save_normalized_data_with_markers(obj, points_dict, output_dir, file_name, marker_radius=0.02):
    """
    保存归一化后的 obj 模型和对应的 json 坐标数据。
    同时，在保存 OBJ 前，会在场景中创建红色的球体来可视化 points_dict 中的点，
    并将这些点与原物体一同合并导出到最终的 OBJ 文件中。
    
    :param obj: Blender 对象 (主物体)
    :param points_dict: 归一化后的坐标字典 {'label': [x, y, z]}
    :param output_dir: 输出文件夹路径
    :param file_name: 文件名前缀 (不带后缀)
    :param marker_radius: 可视化标记球的半径
    """
    
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # ==================================================
    # 阶段 A: 创建临时可视化标记 (红色球体)
    # ==================================================
    
    # 1. 创建红色材质
    mat_name = "Temp_Marker_Red"
    # 检查是否已存在同名材质，如果存在先删除（确保干净的状态）
    if mat_name in bpy.data.materials:
        bpy.data.materials.remove(bpy.data.materials[mat_name])
        
    red_mat = bpy.data.materials.new(name=mat_name)
    red_mat.use_nodes = True
    bsdf = red_mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        # 设置基础色为红色 (RGBA)
        bsdf.inputs['Base Color'].default_value = (1.0, 0.0, 0.0, 1.0)
        # 可选：增加一点自发光让它更显眼
        bsdf.inputs['Emission Color'].default_value = (1.0, 0.0, 0.0, 1.0)
        bsdf.inputs['Emission Strength'].default_value = 1.0

    # 2. 创建基础球体网格数据 (所有标记共用一份数据，节省内存)
    bm = bmesh.new()
    # 创建一个 UV 球体，精度适中即可 (seg=16, ring=8)
    bmesh.ops.create_uvsphere(bm, u_segments=16, v_segments=8, radius=marker_radius)
    marker_mesh_data = bpy.data.meshes.new("Temp_Marker_Mesh")
    bm.to_mesh(marker_mesh_data)
    bm.free()
    # 给网格数据赋予红色材质
    marker_mesh_data.materials.append(red_mat)

    # 3. 根据字典创建临时物体对象
    temp_markers = []
    # 使用当前场景的集合
    collection = bpy.context.scene.collection
    
    for name, coords in points_dict.items():
        # 创建新物体，使用上面创建的球体网格
        marker_obj = bpy.data.objects.new(name=f"Temp_Marker_{name}", object_data=marker_mesh_data)
        marker_obj.location = coords
        # 将物体链接到当前场景集合中
        collection.objects.link(marker_obj)
        temp_markers.append(marker_obj)

    # ==================================================
    # 阶段 B: 保存 OBJ 模型 (包含主物体和标记)
    # ==================================================
    obj_path = os.path.join(output_dir, f"{file_name}.obj")
    
    # 1. 准备选择状态
    # 取消所有选择
    bpy.ops.object.select_all(action='DESELECT')
    
    # 选中主物体，并设为活动物体
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    
    # 选中所有新创建的临时标记物体
    for marker in temp_markers:
        marker.select_set(True)
    
    # 2. 执行导出 OBJ
    # 注意：use_selection=True 确保导出了主物体和红点
    if hasattr(bpy.ops.wm, "obj_export"):
        # Blender 3.2+ 新版导出器
        bpy.ops.wm.obj_export(
            filepath=obj_path,
            export_selected_objects=True,
            export_uv=True,
            export_normals=True,
            export_materials=True, # 确保导出材质信息(MTL文件)以显示红色
            apply_modifiers=True
        )
    else:
        # 旧版导出器
        bpy.ops.export_scene.obj(
            filepath=obj_path,
            use_selection=True,
            use_uvs=True,
            use_normals=True,
            use_materials=True, # 确保导出材质
            use_mesh_modifiers=True
        )
    
    print(f"[Success] OBJ (with visualized points) saved to: {obj_path}")

    # ==================================================
    # 阶段 C: 清理现场 (重要!)
    # ==================================================
    # 导出完成后，删除临时的标记物体、网格数据和材质，保持场景整洁
    
    # 删除物体对象
    for marker in temp_markers:
        bpy.data.objects.remove(marker, do_unlink=True)
        
    # 删除共用的网格数据
    if marker_mesh_data:
        bpy.data.meshes.remove(marker_mesh_data)
        
    # 删除材质
    if red_mat:
        bpy.data.materials.remove(red_mat)


    # ==================================================
    # 阶段 D: 保存 Points Dict (JSON) - (保持不变)
    # ==================================================
    json_path = os.path.join(output_dir, f"{file_name}.json")
    
    serializable_dict = {}
    for key, val in points_dict.items():
        if isinstance(val, np.ndarray):
            serializable_dict[key] = val.tolist()
        else:
            serializable_dict[key] = val
            
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_dict, f, indent=4)
        
    print(f"[Success] Points JSON saved to: {json_path}")


def normalize_object_and_dict(obj, points_dict):
    """
    obj: Blender 对象
    points_dict: 包含坐标点的字典, 格式如 {'point_name': [x, y, z], ...}
    return: 处理后的 points_dict (新的副本)
    """
    
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    
    bpy.ops.object.make_single_user(type='SELECTED_OBJECTS', object=True, obdata=True)
    bpy.ops.object.transform_apply(scale=True, location=True, rotation=True)

    # ==========================
    # 步骤 1：同步居中 (Centering)
    # ==========================
    
    # 获取所有顶点坐标
    vertices = np.array([v.co for v in obj.data.vertices])
    
    # 计算几何中心 (Bounding Box Center)
    # 对应 Blender 的 origin_set(center="BOUNDS")
    min_coords = np.min(vertices, axis=0)
    max_coords = np.max(vertices, axis=0)
    center_offset = (min_coords + max_coords) / 2
    
    # A. 处理 Dict：将所有点减去中心偏移量 (平移到原点)
    # 建议创建副本以防修改原始数据
    new_points_dict = {} 
    for key, coord in points_dict.items():
        # 转换为 numpy 数组方便计算
        new_points_dict[key] = np.array(coord["center_position"]) - center_offset

    # B. 处理 Object：执行原有的 Blender 归一化逻辑
    # 将原点设为几何中心
    bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="BOUNDS")
    # 移动物体到世界原点
    obj.location = (0, 0, 0)
    # 应用变换（将位移烘焙进顶点）
    bpy.ops.object.transform_apply(scale=True, location=True, rotation=True)

    # ==========================
    # 步骤 2：同步缩放 (Scaling)
    # ==========================

    # 重新获取居中后的顶点（此时物体中心已在 0,0,0）
    vertices = np.array([v.co for v in obj.data.vertices])
    
    # 计算最大维度距离
    max_dim = np.max(np.abs(vertices))
    
    if max_dim > 0:
        scale_factor = 0.5 / max_dim
        
        # A. 处理 Dict：应用缩放因子
        for key in new_points_dict:
            new_points_dict[key] *= scale_factor
            # 如果需要转回 list 格式：
            # new_points_dict[key] = new_points_dict[key].tolist()

        # B. 处理 Object：应用缩放因子
        obj.scale = obj.scale * scale_factor
        bpy.ops.object.transform_apply(scale=True, location=True, rotation=True)
        
    return new_points_dict


def save_normalized_data(obj, points_dict, output_dir, file_name):
    """
    保存归一化后的 obj 模型和对应的 json 坐标数据
    
    :param obj: Blender 对象
    :param points_dict: 归一化后的坐标字典
    :param output_dir: 输出文件夹路径
    :param file_name: 文件名前缀 (不带后缀)
    """
    
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # ==========================
    # 1. 保存 OBJ 模型
    # ==========================
    obj_path = os.path.join(output_dir, f"{file_name}.obj")
    
    # 确保只选中了目标物体
    bpy.ops.object.select_all(action='DESELECT')
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    
    # 导出 OBJ
    # 注意：这里使用 'use_selection=True' 确保只导出当前物体
    # 适配 Blender 不同版本的导出命令
    if hasattr(bpy.ops.wm, "obj_export"):
        # Blender 3.2+ 新版导出器 (速度更快)
        bpy.ops.wm.obj_export(
            filepath=obj_path,
            export_selected_objects=True,
            export_uv=True,
            export_normals=True,
            apply_modifiers=True
        )
    else:
        # 旧版导出器
        bpy.ops.export_scene.obj(
            filepath=obj_path,
            use_selection=True,
            use_uvs=True,
            use_normals=True,
            use_mesh_modifiers=True
        )
    
    print(f"[Success] OBJ saved to: {obj_path}")

    # ==========================
    # 2. 保存 Points Dict (JSON)
    # ==========================
    json_path = os.path.join(output_dir, f"{file_name}.json")
    
    # 预处理字典：将 numpy array 转换为 list，否则 json 无法序列化
    serializable_dict = {}
    for key, val in points_dict.items():
        if isinstance(val, np.ndarray):
            serializable_dict[key] = val.tolist()
        else:
            serializable_dict[key] = val
            
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_dict, f, indent=4)
        
    print(f"[Success] Points JSON saved to: {json_path}")

def normalize_object(obj):
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="BOUNDS")
    obj.location = (0, 0, 0)
    bpy.ops.object.transform_apply(scale=True, location=True, rotation=True)

    vertices = np.array([v.co for v in obj.data.vertices])
    max_dim = np.max(np.abs(vertices))
    if max_dim > 0:
        obj.scale = obj.scale * 0.5 / max_dim
        bpy.ops.object.transform_apply(scale=True, location=True, rotation=True)


IMPORT_FUNCTIONS: Dict[str, Callable] = {
    "obj": bpy.ops.import_scene.obj,
    "glb": bpy.ops.import_scene.gltf,
    "gltf": bpy.ops.import_scene.gltf,
    "usd": bpy.ops.import_scene.usd,
    "fbx": bpy.ops.import_scene.fbx,
    "stl": bpy.ops.import_mesh.stl,
    "usda": bpy.ops.import_scene.usda,
    "dae": bpy.ops.wm.collada_import,
    "ply": bpy.ops.import_mesh.ply,
    "abc": bpy.ops.wm.alembic_import,
    "blend": bpy.ops.wm.append,
}


def load_one_object(object_path: str) -> None:
    """Loads a model with a supported file extension into the scene.

    Args:
        object_path (str): Path to the model file.

    Raises:
        ValueError: If the file extension is not supported.

    Returns:
        None
    """
    file_extension = object_path.split(".")[-1].lower()
    if file_extension is None:
        raise ValueError(f"Unsupported file type: {object_path}")

    if file_extension == "usdz":
        # install usdz io package
        dirname = os.path.dirname(os.path.realpath(__file__))
        usdz_package = os.path.join(dirname, "io_scene_usdz.zip")
        bpy.ops.preferences.addon_install(filepath=usdz_package)
        # enable it
        addon_name = "io_scene_usdz"
        bpy.ops.preferences.addon_enable(module=addon_name)
        # import the usdz
        from io_scene_usdz.import_usdz import import_usdz

        import_usdz(context, filepath=object_path, materials=True, animations=True)
        return None

    # load from existing import functions
    import_function = IMPORT_FUNCTIONS[file_extension]

    print(f"Loading object from {object_path}")
    if file_extension == "blend":
        import_function(directory=object_path, link=False)
    elif file_extension in {"glb", "gltf"}:
        import_function(filepath=object_path, merge_vertices=True, import_shading='NORMALS')
    else:
        import_function(filepath=object_path)


# def load_object(path):
#     ext = os.path.splitext(path)[1].lower()
#     if ext == '.ply':
#         if hasattr(bpy.ops.wm, 'ply_import'):
#             bpy.ops.wm.ply_import(filepath=path)
#         elif hasattr(bpy.ops.import_mesh, 'ply'):
#             bpy.ops.import_mesh.ply(filepath=path)
#             # load_object(path)
#         else:
#             raise RuntimeError("Current Blender version does not support PLY import.")
#     elif ext in ['.glb', '.gltf']:
#         bpy.ops.import_scene.gltf(filepath=path, merge_vertices=True)
#     else:
#         raise ValueError(f"Unsupported file format: {ext}")

# 头部记得导入 traceback 以便调试
import traceback

def load_object(path):
    """
    尝试加载模型，针对 PLY 增加容错机制：
    优先尝试新版导入器(快)，失败则回退到旧版导入器(慢但兼容性好)。
    """
    ext = os.path.splitext(path)[1].lower()
    
    if ext == '.ply':
        # 方法 A: 尝试新版 C++ 导入器 (Blender 3.2+)
        try:
            if hasattr(bpy.ops.wm, 'ply_import'):
                print(f"Trying C++ Importer for: {path}")
                bpy.ops.wm.ply_import(filepath=path)
                return # 成功则返回
        except RuntimeError as e:
            print(f"Warning: C++ PLY import failed ({e}). Attempting legacy importer...")
        except Exception as e:
            print(f"Warning: Unknown error in C++ importer: {e}. Attempting legacy importer...")

        # 方法 B: 回退到旧版 Python 导入器
        # 注意：有时候旧版导入器并未开启，或者函数名不同
        try:
            if hasattr(bpy.ops.import_mesh, 'ply'):
                print(f"Trying Legacy Python Importer for: {path}")
                bpy.ops.import_mesh.ply(filepath=path)
                return # 成功则返回
            else:
                print("Error: Legacy PLY importer (import_mesh.ply) not found.")
        except Exception as e:
            raise RuntimeError(f"Legacy PLY import also failed: {e}")

        # 如果两个都失败（或者没找到对应的 operator），抛出异常
        raise RuntimeError(f"Failed to import PLY file: {path}")

    elif ext in ['.glb', '.gltf']:
        bpy.ops.import_scene.gltf(filepath=path, merge_vertices=True)
    else:
        # 其他格式尝试通用加载
        if hasattr(bpy.ops.import_scene, 'obj') and ext == '.obj':
             bpy.ops.import_scene.obj(filepath=path)
        else:
            raise ValueError(f"Unsupported or unhandled file format: {ext}")




def clear_parents_and_normalize(obj_path,anno_pc_location_path, output_path):
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    load_object(obj_path)
    clear_parents()
    clear_nonmesh()
    join_mesh()

    meshes = list(scene_meshes())
    if len(meshes) != 1:
        print(f"Warning: Expected 1 mesh after join, found {len(meshes)}. Using the first one.")

    if len(meshes) > 0:
        obj = meshes[0]
        
        # normalize_object(obj)
        # anno_pc_location_path = os.path.join(cate_file_path, each_obj, "Annotation", "tactile_pc_location.json")
        with open(anno_pc_location_path, "r") as f:
            anno_pc_location_data = json.load(f)
        normalize_pc_location = normalize_object_and_dict(obj,anno_pc_location_data)
        
        normalized_anno_pc_location_path = anno_pc_location_path.replace("pc_location.json", "normalized_pc_location.json")
        
        serializable_normalized_anno_pc_location_dict = {}
        for key, val in normalize_pc_location.items():
            if isinstance(val, np.ndarray):
                serializable_normalized_anno_pc_location_dict[key] = val.tolist()
            else:
                serializable_normalized_anno_pc_location_dict[key] = val
        
        with open(normalized_anno_pc_location_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_normalized_anno_pc_location_dict, f, indent=4)
        
        name = os.path.split(obj_path)[1].replace('.glb', '')
        name = os.path.splitext(os.path.basename(obj_path))[0]
        
        obj.name = name

        print(f"Exporting to: {output_path}")
        bpy.ops.export_scene.gltf(filepath=output_path)
    else:
        print("Error: No mesh data found to export!")


def main(args):

    for obj in bpy.data.objects:
        bpy.data.objects.remove(obj, do_unlink=True)

    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()


    # traverse the all objects

    cate_list = os.listdir(args.object_dir)
    cate_list = [f for f in cate_list if os.path.isdir(os.path.join(args.object_dir, f))]

    input_file_lists = []
    input_anno_pc_location_list = []
    output_file_lists = []
    
    for each_cate in cate_list:
        cate_file_path = os.path.join(args.object_dir, each_cate)
        
        obj_instance_list = os.listdir(cate_file_path)
        obj_instance_list = [obj for obj in obj_instance_list if os.path.isdir(os.path.join(cate_file_path, obj))]
        
        for each_obj in obj_instance_list:
            
            obj_path = os.path.join(cate_file_path, each_obj, "3D_Scanning", "scan.ply")
            each_pc_location_path = os.path.join(cate_file_path, each_obj, "Annotation", "tactile_pc_location.json")
            out_path = os.path.join(cate_file_path, each_obj, "Preprocessed0111")
            os.makedirs(out_path, exist_ok=True)
            
            name = os.path.split(obj_path)[1].replace('.glb', '')
            name = os.path.splitext(os.path.basename(obj_path))[0]
            
            if os.path.isdir(out_path) or not out_path.lower().endswith('.glb'):
                out_path = os.path.join(out_path, name + ".glb")
            normalized_anno_pc_location_path = each_pc_location_path.replace("pc_location.json", "normalized_pc_location.json")
            
            if os.path.exists(out_path) and os.path.exists(normalized_anno_pc_location_path):
                continue
            elif os.path.exists(obj_path):
                input_file_lists.append(obj_path)
                input_anno_pc_location_list.append(each_pc_location_path)
                output_file_lists.append(out_path)
            else:
                print(f"Warning: {obj_path} does not exist, skipping.")
                
    
    print("There are total ", len(input_file_lists), " objects to process.")
    # input()
    Parallel(n_jobs=1, backend='multiprocessing', verbose=10)(
        delayed(clear_parents_and_normalize)(item, anno_item, output_path) for item, anno_item, output_path in zip(input_file_lists, input_anno_pc_location_list, output_file_lists))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--object_dir",
        type=str,
        required=True,
        help="Path to the object file",
    )


    argv = sys.argv[sys.argv.index("--") + 1 :]
    args = parser.parse_args(argv)
    main(args)