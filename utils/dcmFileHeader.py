import pydicom

# 数据路径
# file_path = r"E:\Project\medical-image-preprocessing\data\3_Classified_dcm\XU YUAN BINM00135657t1\t1_vibe_fs_tra_post_two scans-19\t1_vibe_fs_tra_post_two scans-19-1.dcm"
file_path = r"E:\Project\medical-image-preprocessing\test\seriesNumber\Series_"
# file_path += r"19\670.dcm"
# file_path += r"17\580.dcm"  # srs: 17
file_path += r"20\770.dcm"  # srs: 20

data0 = pydicom.read_file(file_path)
print(data0)

# dcmread
# data1 = pydicom.dcmread(file_path)
# Series Number

# 读取DICOM文件
# dicom_dataset = pydicom.dcmread(file_path)

# 获取Series Number
# series_number = dicom_dataset.SeriesNumber
# print(series_number)
