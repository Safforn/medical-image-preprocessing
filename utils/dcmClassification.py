"""
FileName: dcmClassification.py
Function: 将乱序的dcm分别存放在不同的Series_XX文件夹中
"""
import os
import pydicom
from collections import Counter
from shutil import copyfile


def get_series_number(dicom_file_path):
    try:
        # 读取DICOM文件
        dicom_dataset = pydicom.dcmread(dicom_file_path)

        # 获取Series Number
        series_number = dicom_dataset.SeriesNumber

        return series_number
    except Exception as e:
        print(f"Error: {e}")
        return None


def organize_dicom_files(input_folder, output_folder):
    # 遍历文件夹
    for root, dirs, files in os.walk(input_folder):
        for file_name in files:
            # 检查文件扩展名是否为.dcm
            if file_name.lower().endswith('.dcm'):
                dicom_file_path = os.path.join(root, file_name)

                # 获取Series Number
                series_number = get_series_number(dicom_file_path)

                if series_number is not None:
                    # 构建目标文件夹路径
                    target_folder = os.path.join(output_folder, f"Series_{series_number}")

                    # 如果目标文件夹不存在，则创建它
                    if not os.path.exists(target_folder):
                        os.makedirs(target_folder)

                    # 构建目标文件路径
                    target_file_path = os.path.join(target_folder, file_name)

                    # 复制DICOM文件到目标文件夹
                    copyfile(dicom_file_path, target_file_path)


def process_dicom_folder(folder_path):
    # 存储所有DICOM文件的Series Number
    series_numbers = []

    # 遍历文件夹
    for root, dirs, files in os.walk(folder_path):
        for file_name in files:
            # 检查文件扩展名是否为.dcm
            if file_name.lower().endswith('.dcm'):
                dicom_file_path = os.path.join(root, file_name)

                # 获取Series Number
                series_number = get_series_number(dicom_file_path)

                if series_number is not None:
                    # 将Series Number 添加到列表中
                    series_numbers.append(series_number)

    return series_numbers


# 指定DICOM文件夹路径
# dicom_folder_path = r"C:\Users\18049\Desktop\nii\1_Unclassified_dcm\metastasis 2\metastasis2017\XU YUAN BIN_M00135657"

# 处理DICOM文件夹
# all_series_numbers = process_dicom_folder(dicom_folder_path)
#
# # 统计Series Number
# series_number_counts = Counter(all_series_numbers)
#
# # 打印统计结果
# print("Series Number Counts:")
# for series_number, count in series_number_counts.items():
#     print(f"Series Number: {series_number}, Count: {count}")


# 指定DICOM文件夹路径和目标文件夹路径
input_dicom_folder = r"C:\Users\18049\Desktop\nii\1_Unclassified_dcm\metastasis 2\metastasis2017\XU YUAN BIN_M00135657"
output_root_folder = r"E:\Project\medical-image-preprocessing\data\3_Classified_dcm\\"

# 处理DICOM文件并按Series Number分类存储
organize_dicom_files(input_dicom_folder, output_root_folder)
