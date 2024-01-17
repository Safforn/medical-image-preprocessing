import os
import pydicom
import numpy as np


def process_dicom_file(file_path):
    # 读取DICOM文件
    dicom_file = pydicom.dcmread(file_path)

    # 获取像素数组
    pixel_data = dicom_file.pixel_array

    # 打印像素数组的最大值
    max_pixel_value = np.max(pixel_data)
    print(file_path, "最大值为: ", max_pixel_value)


def process_dicom_files_in_folder(folder_path):
    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        # 检查文件是否是DICOM文件
        if filename.endswith(".dcm"):
            file_path = os.path.join(folder_path, filename)
            process_dicom_file(file_path)


dicom_file_path = r"E:\Project\medical-image-preprocessing\data\2_classified_dcm\XU YUAN BIN_M00135657\Series_20"
process_dicom_files_in_folder(dicom_file_path)
