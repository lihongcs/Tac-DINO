import numpy as np
import OpenEXR as exr
import Imath
import argparse
from pathlib import Path
from tempfile import NamedTemporaryFile

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
    input_directory = Path(input_directory)
    output_directory = Path(output_directory)
    depth_files = sorted(input_directory.glob("*.exr"))
    if not depth_files:
        raise FileNotFoundError(f"No rendered EXR depth files in {input_directory}")

    depth_by_view = {path.stem.removesuffix("_0001"): path for path in depth_files}
    for image in sorted((input_directory.parent / "rgb_images").glob("*.png")):
        if image.stem not in depth_by_view:
            raise FileNotFoundError(f"Missing rendered EXR depth for {image}")

    output_directory.mkdir(parents=True, exist_ok=True)
    for view, path in depth_by_view.items():
        output = output_directory / f"{view}.npy"
        if output.is_file():
            continue
        depth = readEXR(str(path)).astype(np.float16)
        with NamedTemporaryFile(dir=output_directory, suffix=".tmp", delete=False) as temporary:
            temporary_path = Path(temporary.name)
            try:
                np.save(temporary, depth)
                temporary.close()
                temporary_path.replace(output)
            finally:
                temporary_path.unlink(missing_ok=True)


def main(args):
    root = Path(args.object_dir)
    render_dirs = {path for path in root.glob("*/*/Preprocessed0111/render") if path.is_dir()}
    render_dirs.update(mesh.parent / "render" for mesh in root.glob("*/*/Preprocessed0111/scan.glb"))
    if not render_dirs:
        raise FileNotFoundError(f"No normalized scans or render directories found under {root}")

    print(f"Converting depth for {len(render_dirs)} scans")
    for render in sorted(render_dirs):
        convert_exr_to_npy(render / "depth_exr", render / "depth_npy")


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
