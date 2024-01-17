import os
import pydicom


def batch_processing(root_folder, handler, output_folder=""):

    # 遍历根文件夹下的所有子文件夹
    for subfolder in os.listdir(root_folder):
        subfolder_path = os.path.join(root_folder, subfolder)

        # 检查是否是文件夹
        if os.path.isdir(subfolder_path):
            # 构造输入 DICOM 文件夹路径和输出 NIfTI 文件路径
            input_dicom_folder = subfolder_path
            print(input_dicom_folder, subfolder)

            # 将 DICOM 文件夹中的多个文件合成一个 NIfTI 文件（.nii 格式）
            try:
                if output_folder == "":
                    handler(input_dicom_folder)
                else:
                    handler(input_dicom_folder, output_folder)
            except Exception as e:
                # 处理异常，例如输出错误信息
                print(f"Error processing {input_dicom_folder}: {str(e)}")
                # 继续循环到下一个对象
                continue


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

