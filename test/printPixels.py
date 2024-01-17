import pydicom

# 1. 获取像素数据
dicom_file_path = r"E:\Project\medical-image-preprocessing\data\2_classified_dcm\XU YUAN BIN_M00135657\Series_20"
file_name = r"761.dcm"
dicom_file = pydicom.dcmread(dicom_file_path+"\\" + file_name)
pixel_data = dicom_file.pixel_array

# # 2. 打印像素值
# for row in pixel_data:
#     for pixel in row:
#         print(pixel, end=" ")
#     print("")


# 3. 像素统计频数
# flattened_array = [item for sublist in pixel_data for item in sublist]
# from collections import Counter
# counter = Counter(flattened_array)
# sorted_keys = sorted(counter.keys())
# for key in sorted_keys:
#     print(f"{key}: \t{counter[key]}")


# 4. 打印最大值
import numpy as np
array_np = np.array(pixel_data)
max_value = np.max(array_np)
print(file_name, "最大值为: ", max_value)

