import os
import numpy as np
import OpenEXR as exr
import Imath
import argparse
from joblib import Parallel, delayed

def readEXR(filename):
    """Read RGB + Depth data from EXR image file.
    Parameters
    ----------
    filename : str
        File path.
    Returns
    -------
    Z: Depth buffer in float3.
    """

    exrfile = exr.InputFile(filename)
    header = exrfile.header()
    dw = exrfile.header()['dataWindow']
    isize = (dw.max.y - dw.min.y + 1, dw.max.x - dw.min.x + 1)

    available_channels = header['channels'].keys()
    
    # 定义查找优先级：先找 Z，再找 Depth，最后找 R (通常灰度图存在 R 通道中)
    # 如果你的渲染器确实输出了 V，这里也可以加上 'V'
    possible_names = ['Z', 'Depth', 'R', 'Y', 'V']
    
    target_channel = None
    for name in possible_names:
        if name in available_channels:
            target_channel = name
            break
            
    # 如果没找到常用名，且只有一个通道，就直接用那个
    if target_channel is None and len(available_channels) == 1:
        target_channel = list(available_channels)[0]
        
    if target_channel is None:
        raise TypeError(f"在图像 {filename} 中未找到深度通道。可用通道: {list(available_channels)}")

    # 读取找到的通道
    Z = exrfile.channel(target_channel, Imath.PixelType(Imath.PixelType.FLOAT))
    Z = np.frombuffer(Z, dtype=np.float32)
    Z = np.reshape(Z, isize)
    # Z = exrfile.channel('V', Imath.PixelType(Imath.PixelType.FLOAT))
    # Z = np.frombuffer(Z, dtype=np.float32)
    # Z = np.reshape(Z, isize)
    
    return Z

def convert_exr_to_npy(input_directory, output_directory):
    # Ensure the output directory exists
    os.makedirs(output_directory, exist_ok=True)
    
    # Loop through all files in the input directory
    for filename in os.listdir(input_directory):
        if filename.endswith('.exr'):
            # Construct full file path
            file_path = os.path.join(input_directory, filename)
            
            depth_array = readEXR(file_path).astype(np.float16)
            
            # Define output filename and save the array
            output_filename = os.path.splitext(filename)[0].replace('_0001','') + '.npy'
            output_path = os.path.join(output_directory, output_filename)
            np.save(output_path, depth_array)
            #print(f"Converted {filename} to {output_filename}")

def main(args):
    
    
    cate_list = os.listdir(args.object_dir)
    cate_list = [f for f in cate_list if os.path.isdir(os.path.join(args.object_dir, f))]

    input_file_lists = []
    output_file_lists = []
    
    for each_cate in cate_list:
        cate_file_path = os.path.join(args.object_dir, each_cate)
        
        obj_instance_list = os.listdir(cate_file_path)
        obj_instance_list = [obj for obj in obj_instance_list if os.path.isdir(os.path.join(cate_file_path, obj))]
        
        for each_obj in obj_instance_list:
            input_exr_path = os.path.join(cate_file_path, each_obj, "Preprocessed0111", "render","depth_exr")
            output_npy_path = os.path.join(cate_file_path, each_obj, "Preprocessed0111", "render","depth_npy")
            
            os.listdir(input_exr_path)
            if os.path.exists(output_npy_path) and len(os.listdir(output_npy_path)) > 0:
                continue
            elif os.path.exists(input_exr_path):
                input_file_lists.append(input_exr_path)
                output_file_lists.append(output_npy_path)
            else:
                print(f"Skipping {each_obj} as no depth_exr found.")
            
    
    print(f"{len(output_npy_path)} to process")
    
    # items = sorted(os.listdir(render_dir))
    # input_dirs = []
    # output_dirs = []
    # for input_dir, output_dir in zip(input_file_lists, output_file_lists):
    #     # print(f"Processing {input_dir} to {output_dir}")
    #     input_dirs.append(input_dir)
    #     output_dirs.append(output_dir)
    
    # input_dirs = [os.path.join(render_dir, item, "depth_exr") for item in items]
    # output_dirs = [input_dir.replace('depth_exr', 'depth_npy') for input_dir in input_dirs]

    Parallel(n_jobs=1, backend='multiprocessing', verbose=10)(
        delayed(convert_exr_to_npy)(input_dir, output_dir) for input_dir, output_dir in zip(input_file_lists, output_file_lists))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--object_dir",
        type=str,
        required=True,
        help="Path to the object file",
    )


    args = parser.parse_args()
    main(args)
