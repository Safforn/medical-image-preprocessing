import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np


def convert_3d_nii_to_png(nii_path, output_png_folder):
    """将3D的nii文件转换为身体横截方向(tra)的png序列.

    :param nii_path: 源nii文件路径.
    :param output_png_folder: 输出的png的文件夹路径.
    """
    nii_img = nib.load(nii_path)  # 加载NIfTI文件
    nii_data = nii_img.get_fdata()  # 获取NIfTI数据
    # 遍历数据中的每个切片
    for i in range(nii_data.shape[2]):
        # 获取当前切片的数据
        slice_data = nii_data[:, :, i]

        # 顺时针旋转90度
        rotated_slice = np.rot90(slice_data, k=1)

        # 上下镜像
        mirrored_slice = np.flipud(rotated_slice)

        # 创建一个matplotlib图像
        plt.imshow(mirrored_slice, cmap='gray')
        plt.axis('off')

        # 保存图像为PNG文件
        plt.savefig(f'{output_png_folder}/nii2png_{i:04d}.png', bbox_inches='tight', pad_inches=0, dpi=100)
        plt.close()


project_path = r'E:\Project\medical-image-preprocessing'
# NIfTI文件路径
nii_path = project_path + r'\data\3_classified_nii\XU YUAN BIN_M00135657\19-t1_vibe_fs_tra_post_two scans.nii'
# 保存PNG图像的文件夹路径
output_folder = project_path + r'\test\nii2png_output'

convert_3d_nii_to_png(nii_path, output_folder)
