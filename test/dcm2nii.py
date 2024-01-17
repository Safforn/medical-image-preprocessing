"""
FileName: dcm2nii.py
Function: 将目录下的所有子目录中的dcm分别打包成nii文件
"""
import os
import pydicom
import SimpleITK as sitk


# 获取dcm文件的序列信息
def get_series_info_sitk(dicom_file_path):
    # 读取DICOM文件
    dicom_reader = sitk.ImageFileReader()
    dicom_reader.SetFileName(dicom_file_path)
    dicom_reader.ReadImageInformation()

    # 获取Series Description和Series Number
    series_description = dicom_reader.GetMetaData("0008|103e")  # Series Description
    series_number = dicom_reader.GetMetaData("0020|0011")  # Series Number

    return series_description.strip(), series_number


def get_series_info_py(dicom_file_path):
    try:
        # 读取DICOM文件
        dicom_dataset = pydicom.dcmread(dicom_file_path)

        # 获取Series Number和Series Description
        series_number = dicom_dataset.SeriesNumber
        series_description = dicom_dataset.SeriesDescription

        return series_description, series_number
    except Exception as e:
        print(f"Error: {e}")
        return None, None


# 获取input_folder文件夹下的dcm图片序列
def get_nii_name(dicom_file_path):
    dicom_files = [f for f in os.listdir(dicom_file_path) if f.endswith('.dcm')]
    if not dicom_files:
        print("文件夹中没有DICOM文件")
        return

    # 选择文件夹中的第一个DICOM文件, 获取Series Description和Series Number
    first_dcm_file = os.path.join(dicom_file_path, dicom_files[0])
    series_description, series_number = get_series_info_py(first_dcm_file)

    # 构建新的文件名 "SeriesNumber-SeriesDescription.nii"
    # series_number = str(series_number).zfill(2)
    nii_name = f"{series_number}-{series_description}.nii"
    return nii_name


# 将input_folder文件夹下的dcm图片序列合成为nii文件
def dicom_series_to_nifti(input_folder, output_path):
    # 获取文件夹中的所有 DICOM 文件
    dicom_reader = sitk.ImageSeriesReader()
    dicom_series = dicom_reader.GetGDCMSeriesFileNames(input_folder)
    dicom_reader.SetFileNames(dicom_series)

    image = dicom_reader.Execute()  # 读取 DICOM 文件
    nifti_image = sitk.Cast(image, sitk.sitkFloat32)  # DICOM 类型为NIfTI类型

    # Ensure output folder exists
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    nii_name = get_nii_name(input_folder)
    print(os.path.join(output_path, nii_name))
    sitk.WriteImage(nifti_image, os.path.join(output_path, nii_name))  # 保存为 NIfTI 文件


# 单次dcm转换nii
# input_dicom_folder = r"E:\Project\medical-image-preprocessing\data\2_classified_dcm\CAI GUO MIN_NM00002073\Series_15"
# output_nifti_path = r"E:\Project\medical-image-preprocessing\test\dcm2nii_output\CAI GUO MIN_NM00002073"
# dicom_series_to_nifti(input_dicom_folder, output_nifti_path)


root_folder = \
    r"C:\Users\18049\Desktop\nii\2022HCC\2_classified_dcm\ZHAO XIN JIE_M00217105\MR__20220113080428_1895391"
output_nifti_path = r"C:\Users\18049\Desktop\nii\2022HCC\3_classified_nii\ZHAO XIN JIE_M00217105"

# 遍历根文件夹下的所有子文件夹
for subfolder in os.listdir(root_folder):
    subfolder_path = os.path.join(root_folder, subfolder)

    # 检查是否是文件夹
    if os.path.isdir(subfolder_path):
        # 构造输入 DICOM 文件夹路径和输出 NIfTI 文件路径
        input_dicom_folder = subfolder_path

        # 将 DICOM 文件夹中的多个文件合成一个 NIfTI 文件（.nii 格式）
        try:
            dicom_series_to_nifti(input_dicom_folder, output_nifti_path)
        except Exception as e:
            # 处理异常，例如输出错误信息
            print(f"Error processing {input_dicom_folder}: {str(e)}")
            # 继续循环到下一个对象
            continue
