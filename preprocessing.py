from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
import sys
import os
import pydicom
import cv2
from tqdm import tqdm
import numpy as np
import SimpleITK as sitk
import nibabel as nib

PROJECT_PATH = r'E:\Project\medical-image-preprocessing'
DATA_PATH = PROJECT_PATH + r'\data'
unclassified_dcm_path = DATA_PATH + r'\1_Unclassified_dcm'
classified_nii_path = DATA_PATH + r'\2_Classified_nii'
classified_dcm_path = DATA_PATH + r'\3_Classified_dcm'
afterfolder_nii_path = DATA_PATH + r'\4_afterfolder_nii'
unmarked_mp4_path = DATA_PATH + r'\5_unmarked_mp4'
marked_mp4_path = DATA_PATH + r'\6_marked_mp4'
src_dcm_rath = DATA_PATH + r'\7_src_dcm\XU YUAN BINM00135657_t1_vibe_fs_tra_post_two scans-19'
print(classified_nii_path)


def filename(dcm_path):
    a = []
    e = []
    name = []
    foldername = []
    dirs = os.listdir(dcm_path)
    for file in dirs:
        path = dcm_path + '/' + file
        info = {}
        # 读取dicom文件
        dcm = pydicom.read_file(path)
        info["PatientID"] = dcm.PatientID  # 患者ID
        info["PatientName"] = dcm.PatientName  # 患者姓名
        # info['StudyID'] = dcm.StudyID  # 检查ID
        # info['StudyDate'] = dcm.StudyDate  # 检查日期
        # info['Instance Creation Time'] = dcm.InstanceCreationTime
        # info['AcquisitionTime'] = dcm.AcquisitionTime  # 采集时间不一样
        # info['ProtocolName'] = dcm.ProtocolName  # 协议名称
        info['Series Instance UID '] = dcm.SeriesInstanceUID  # UID
        info['Series Number'] = dcm.SeriesNumber  # 序列号
        info['Series Description'] = dcm.SeriesDescription  # 检查方式名称
        c = 1
        for b in a:
            if dcm.SeriesInstanceUID == b:
                c = 0
        if c == 1:
            a.append(dcm.SeriesInstanceUID)
            name.append(str(dcm.SeriesDescription) + '-' + str(dcm.SeriesNumber))
        c = 1
        for d in e:
            if str(dcm.PatientName) + str(dcm.PatientID) == d:
                c = 0
        if c == 1:
            foldername.append(str(dcm.PatientName) + str(dcm.PatientID))
    return name, foldername
dcm_path = unclassified_dcm_path
nii_path = classified_nii_path
name, foldername = filename(dcm_path)
f = foldername[0]
print("for f in foldername:", f)
if os.path.exists(nii_path + r'\\' + f + 'all') == False:
    os.mkdir(nii_path + r'\\' + f + 'all')
if os.path.exists(nii_path + r'\\' + f + 't1') == False:
    os.mkdir(nii_path + r'\\' + f + 't1')
reader = sitk.ImageSeriesReader()
reader.MetaDataDictionaryArrayUpdateOn()  # 这一步是加载公开的元信息
reader.LoadPrivateTagsOn()  # 这一步是加载私有的元信息
series_IDs = sitk.ImageSeriesReader.GetGDCMSeriesIDs(dcm_path)


def find_corresponding_standard_img(standard_files, num_standard_files, standard_dir, position):
    """
     寻找对应位置的标准图像，从标准组中找到位于特定位置的图片，position取值为[0, 1]
     :param standard_dir: 标准图像文件夹路径
     :param position: 所需位置
    """
    # standard_files = os.listdir(standard_dir)
    # num_standard_files = len(standard_files)

    # 计算指定位置对应的文件名在列表中的索引
    index = int(num_standard_files * position)
    # 返回指定位置对应的文件名
    return os.path.join(standard_dir, standard_files[index])

def aveBatch(nii_path, in_path, out_path, src_path):
    """
     直方图规定化，把in_path下的图像按照src_img图像进行规定化
     # 23/03/13修改: 以下路径均为文件夹
     :param in_path:  待处理图像输入路径
     :param out_path: 输出路径
     :param src_path: 源图像路径
    """
    ori_data = sitk.ReadImage(nii_path)  # 读取一个数据
    data1 = sitk.GetArrayFromImage(ori_data)  # 获取数据的array

    # 获取所有图片名称
    Index = out_path.rfind(r"\\")
    a = out_path[Index:]
    out_path2 = out_path.replace(a, "")
    print(out_path2)
    if not os.path.exists(out_path2):
        os.makedirs(out_path2)

    mark = 0
    c = []
    names = os.listdir(in_path)  # 路径

    print("name数组大小为：%s" % len(names))
    # 将文件夹中的文件名称与后边的 .dcm分开
    for name in names:
        index = name.rfind('.')
        name = name[:index]
        c.append(name)
    c.sort(key=lambda x: int(x.split('-')[-1]))

    # 遍历所有 .dcm 文件并处理
    standard_files = os.listdir(src_path)
    num_standard_files = len(standard_files)

    standard_files_name = []
    for file in standard_files:
        idx = file.rfind('.')
        name = file[:idx]
        standard_files_name.append(name)
    standard_files_name.sort(key=lambda x: int(x.split('-')[-1]))
    standard_files = []
    for file_name in standard_files_name:
        standard_files.append(file_name + ".dcm")
    # print(standard_files)
    i = 0

    # 获取文件夹中的文件列表
    file_list = os.listdir(in_path)
    file_list.sort()

    # 计算关键帧位置
    middle_index = len(file_list) // 2
    middle_frame_snr = 0  # 初始化关键帧的信噪比

    for index, files in enumerate(c):
        picture_in_path = in_path + "/" + files + ".dcm"
        # 计算当前正在处理的图片的位置
        position = index / len(c)
        # print("i:" + str(i))
        # print("position:" + str(position))
        i = i + 1
        # 处理图片并保存
        standard_img_path = find_corresponding_standard_img(standard_files, num_standard_files, src_path,
                                                                 position)

        # print(picture_in_path)
        ds1 = pydicom.read_file(standard_img_path)
        src_img = ds1.pixel_array
        # print(src_img.shape)
        ds2 = pydicom.read_file(picture_in_path)
        dst_img = ds2.pixel_array

        RANGE = dst_img.max() + 1

        # 直方图规定化（匹配）
        # src_img为标准图像
        result = hist_specify(src_img, dst_img, RANGE)

        # 若为关键帧，计算信噪比
        if index == middle_index:
            middle_frame_snr = snr(result)

        # print(data1.shape)
        data1[mark] = result
        # print(data1.shape)
        # print("第%s张图像处理完成" % mark)
        # print(data1.shape)
        mark += 1

    snr_threshold = 0.50
    if middle_frame_snr > snr_threshold:
        print("白平衡, 阈值=", snr_threshold)
        # for i in range(len(data1)):
        #     data1[i] = self.white_balance(data1[i])
            # RANGE = data1[i].max() + 1
            # 直方图规定化（匹配）
            # src_img为标准图像
            # data1[i] = self.hist_specify(src_img, data1[i], RANGE)
    else:
        print("边缘强化")
        for i in range(len(data1)):
            # edge_enhance(self, img, low_threshold, high_threshold)
            low_threshold, high_threshold = 60, 150

            data1[i] = edge_enhance(data1[i], low_threshold, high_threshold)
            # RANGE = data1[i].max() + 1
            # # 直方图规定化（匹配）
            # # src_img为标准图像
            # data1[i] = self.hist_specify(src_img, data1[i], RANGE)

    picture_out_path = out_path + ".nii"
    out = sitk.GetImageFromArray(data1)
    sitk.WriteImage(out, picture_out_path)
    # print('所有图像均值化完毕')

def edge_enhance(self, img, low_threshold, high_threshold):
    # 将16位图像转换为8位
    img_8bit = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
    # 高斯滤波
    img_8bit = cv2.GaussianBlur(img_8bit, (5, 5), 1.4)
    # 应用Canny边缘检测
    edges = cv2.Canny(img_8bit, low_threshold, high_threshold)
    # 创建掩膜，将边缘区域设为白色
    mask = np.zeros_like(img)
    mask[edges > 0] = 255
    src_img_weight = 1
    mask_weight = 0.5
    print("$测试$ 原图像权重=", src_img_weight, ", 掩膜权重=", mask_weight)

    result = cv2.addWeighted(img, src_img_weight, mask, mask_weight, 0)
    return result

# aveBatch(nii, Classified_dcm_path2, afterfolder_nii_path2, src_dcm_rath)