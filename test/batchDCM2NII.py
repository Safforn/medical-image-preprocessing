"""
FileName: batchDCM2NII.py
Function: 批量dcm转换为nii
"""
import os
from test.dcm2nii import dicom_series_to_nifti

classified_dcm_root_folder = r"E:\Project\medical-image-preprocessing\data\2_classified_dcm"
output_folder = r"E:\Project\medical-image-preprocessing\data\3_classified_nii"

# 遍历根文件夹下的所有子文件夹
for patient_folder in os.listdir(classified_dcm_root_folder):
    if '`' in patient_folder:
        continue  # 排除掉特殊标记的文件夹
    patient_folder_path = os.path.join(classified_dcm_root_folder, patient_folder)
    for series_folder in os.listdir(patient_folder_path):
        series_folder_path = os.path.join(patient_folder_path, series_folder)
        # 检查是否是文件夹
        if os.path.isdir(series_folder_path):
            # 将 DICOM 文件夹中的多个文件合成一个 NIfTI 文件（.nii 格式）
            output_nifti_path = os.path.join(output_folder, patient_folder)
            try:
                dicom_series_to_nifti(series_folder_path, output_nifti_path)
            except Exception as e:
                print(f"Error processing {series_folder_path}: {str(e)}")  # 输出错误信息
                continue  # 继续循环到下一个对象
