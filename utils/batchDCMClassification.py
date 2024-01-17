"""
FileName: dcmClassification.py
Function: 批量将乱序的dcm分别存放在不同的Series_XX文件夹中
"""
import os
from utils.dcmClassification import organize_dicom_files

root_folder = r"C:\Users\18049\Desktop\nii\1_Unclassified_dcm\metastasis 2\metastasis 15-16"
output_folder = r"E:\Project\medical-image-preprocessing\data\2_classified_dcm"

# 遍历根文件夹下的所有子文件夹
for subfolder in os.listdir(root_folder):
    subfolder_path = os.path.join(root_folder, subfolder)

    # 检查是否是文件夹
    if os.path.isdir(subfolder_path):
        # 构造输入 DICOM 文件夹路径和输出 NIfTI 文件路径
        input_dicom_folder = subfolder_path
        print(input_dicom_folder)

        # 将 DICOM 文件夹中的多个文件合成一个 NIfTI 文件（.nii 格式）
        try:
            organize_dicom_files(input_dicom_folder, os.path.join(output_folder, subfolder))
        except Exception as e:
            # 处理异常，例如输出错误信息
            print(f"Error processing {input_dicom_folder}: {str(e)}")
            # 继续循环到下一个对象
            continue






