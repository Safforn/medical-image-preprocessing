from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
import sys
import os
import pydicom
import cv2
from cv2 import VideoWriter_fourcc
from tqdm import tqdm
import numpy as np
import SimpleITK as sitk
import nibabel as nib
# from app import APP_EISeg



class ThreadCRM(QThread):
    nii_path = pyqtSignal(list)
    process_max = pyqtSignal(int)  # 用于控制进度条
    process_num = pyqtSignal(int)

    # 构造函数
    def __init__(self):
        super(ThreadCRM, self).__init__()
        self.unclassified_dcm_path = ""
        self.isPause = False
        self.isCancel = False
        self.cond = QWaitCondition()
        self.mutex = QMutex()


    # 暂停
    def pause(self):
        print("线程暂停")
        self.isPause = True

    # 恢复
    def resume(self):
        print("线程恢复")
        self.isPause = False
        self.cond.wakeAll()

    # 取消
    def cancel(self):
        print("线程取消")
        self.isCancel = True

    def set_unclassified_dcm_path(self, unclassified_dcm_path=r'D:\Medical_image\tx\all\1_Unclassified_dcm'):
        if unclassified_dcm_path is None:
            return
        self.unclassified_dcm_path = unclassified_dcm_path

    def filename(self, dcm_path):
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
            c == 1
            for d in e:
                if str(dcm.PatientName) + str(dcm.PatientID) == d:
                    c = 0
            if c == 1:
                foldername.append(str(dcm.PatientName) + str(dcm.PatientID))
        return name, foldername

    def setResample(self, imagesitk, newspace=(1, 1, 1)):
        euler3d = sitk.Euler3DTransform()
        nh, nw, nq = imagesitk.GetSize()
        dh, dw, dq = imagesitk.GetSpacing()
        origin = imagesitk.GetOrigin()
        direction = imagesitk.GetDirection()
        size = (int(nh * dh / newspace[0]), int(nw * dw / newspace[1]), int(nq * dq / newspace[2]))
        imagesitk = sitk.Resample(imagesitk, size, euler3d, sitk.sitkLinear, origin, newspace, direction)

        return imagesitk

    # 运行(入口)
    def run(self):
        # TODO: 将未分类的dcm分类打包成nii文件，并选出t1序列
        Classified_nii_path = r'D:\Medical_image\tx\all\2_Classified_nii'

        dcm_path = self.unclassified_dcm_path
        nii_path = Classified_nii_path
        name, foldername = self.filename(dcm_path)

        f = foldername[0]
        # 分割目录
        print("for f in foldername:", f)
        if os.path.exists(nii_path + r'\\' + f + 'all') == False:
            os.mkdir(nii_path + r'\\' + f + 'all')
        if os.path.exists(nii_path + r'\\' + f + 't1') == False:
            os.mkdir(nii_path + r'\\' + f + 't1')
        reader = sitk.ImageSeriesReader()
        reader.MetaDataDictionaryArrayUpdateOn()  # 这一步是加载公开的元信息
        reader.LoadPrivateTagsOn()  # 这一步是加载私有的元信息
        series_IDs = sitk.ImageSeriesReader.GetGDCMSeriesIDs(dcm_path)
        i = 0
        nii_path3 = []

        self.process_max.emit(len(series_IDs))  # 进度条
        process_num = 0

        for series_ID in series_IDs:
            self.mutex.lock()  # 线程锁，用于控制暂停终止
            if self.isPause:
                self.cond.wait(self.mutex)
            if self.isCancel:
                # TODO: 线程终止后，是否要删除创建的文件
                self.process_num.emit(0)
                self.mutex.unlock()  # 解锁
                return
            try:
                dicom_names = reader.GetGDCMSeriesFileNames(dcm_path, series_ID)  # 选取其中一个序列ID,获得该序列的若干文件名
                reader.SetFileNames(dicom_names)  # 设置文件名
                image3D = reader.Execute()  # 读取dicom序列
                imagesitk = self.setResample(image3D, newspace=(1, 1, 1))
                j = name[i].replace('<', '')
                j = j.replace('>', '')
                i = i + 1
                nii_path2 = nii_path + r'\\' + f + 'all' + r'\\' + j + '.nii'
                print(j)
                image_array = sitk.GetArrayFromImage(imagesitk)  # z, y, x
                origin = imagesitk.GetOrigin()  # x, y, z
                spacing = imagesitk.GetSpacing()  # x, y, z
                direction = imagesitk.GetDirection()  # x, y, z
                # 3.将array转为img，并保存为.nii
                image3 = sitk.GetImageFromArray(image_array)
                image3.SetSpacing(spacing)
                image3.SetDirection(direction)
                image3.SetOrigin(origin)
                sitk.WriteImage(image3, nii_path2)
                j2 = j[:2]
                if j2 == 't1':
                    nii_path4 = nii_path + r'\\' + f + 't1' + r'\\' + j + '.nii'
                    sitk.WriteImage(image3, nii_path4)
                    nii_path3.append(nii_path4)
            except Exception as e:
                print('错误信息:', e)

            process_num += 1  # 进度条
            self.process_num.emit(process_num)
            self.mutex.unlock()  # 解锁

        self.nii_path.emit(nii_path3)
        # return nii_path3, f


class ThreadEqualization(QThread):
    targetDir = pyqtSignal(str)
    process_max = pyqtSignal(int)  # 用于控制进度条
    process_num = pyqtSignal(int)

    # 构造函数
    def __init__(self):
        super(ThreadEqualization, self).__init__()
        self.nii_path = None
        self.isPause = False
        self.isCancel = False
        self.cond = QWaitCondition()
        self.mutex = QMutex()

    # 暂停
    def pause(self):
        print("线程暂停")
        self.isPause = True

    # 恢复
    def resume(self):
        print("线程恢复")
        self.isPause = False
        self.cond.wakeAll()

    # 取消
    def cancel(self):
        print("线程取消")
        self.isCancel = True

    def set_nii_path(self, nii_path: list):
        if nii_path is None:
            return
        self.nii_path = nii_path

    def nii2dcm_single(self, nii_path, save_folder, IsData=True):

        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        ori_data = sitk.ReadImage(nii_path)  # 读取一个数据
        data1 = sitk.GetArrayFromImage(ori_data)  # 获取数据的array
        # if IsData:  # 过滤掉其他无关的组织，标签不需要这步骤
        # data1[data1 > 250] = 250
        # data1[data1 < -250] = -250

        img_name = os.path.split(nii_path)  # 分离文件名
        img_name = img_name[-1]
        img_name = img_name.split('.')
        img_name = img_name[0]
        i = data1.shape[0]
        if i < 50:  # TODO: 筛掉z轴太短的图像，持保留意见
            return False
        # 关键部分
        castFilter = sitk.CastImageFilter()
        castFilter.SetOutputPixelType(sitk.sitkInt16)
        for j in range(0, i):  # 将每一张切片都转为png
            if IsData:  # 数据
                slice_i = data1[j, :, :]
                data_img = sitk.GetImageFromArray(slice_i)
                # Convert floating type image (imgSmooth) to int type (imgFiltered)
                data_img = castFilter.Execute(data_img)
                sitk.WriteImage(data_img, "%s/%s-%d.dcm" % (save_folder, img_name, j))
            else:  # 标签
                slice_i = data1[j, :, :] * 122
                label_img = sitk.GetImageFromArray(slice_i)
                # Convert floating type image (imgSmooth) to int type (imgFiltered)
                label_img = castFilter.Execute(label_img)
                sitk.WriteImage(label_img, "%s/%s-%d.dcm" % (save_folder, img_name, j))
        return True

    # 根据映射表进行灰度映射
    def grayscaleMapping(self, img, table):
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                img[i][j] = table[img[i][j]]
        return img

    def get_acc_prob_hist(self, hist, RANGE):
        # 返回来一个给定形状和类型的用0填充的数组 RANGE行1列
        acc_hist = np.zeros([RANGE, 1], np.float32)
        pre_val = 0.
        for i in range(RANGE):
            acc_hist[i, 0] = pre_val + hist[i, 0]
            pre_val = acc_hist[i, 0]
        acc_hist /= pre_val
        return acc_hist

    def hist_specify(self, src_img, dst_img, RANGE):
        """
        直方图规定化，把dst_img按照src_img的图像进行规定化
        :param src_img: 源图像像素数据
        :param dst_img: 待处理图像像素数据
        :param RANGE: 灰度范围限制值
        :return:
        """
        """
        计算源图像和规定化之后图像的累计直方图
        void calcHist(const Mat * images, int images, const int * channels, InputArray mask,
        OutputArray hist, int dims, const int * histSize,
        const float ** ranges, bool uniform = true, bool accumulate = false );

        histSize：直方图中每个dims维度需要分成多少个区间（如果把直方图看作一个一个竖条的话，就是竖条的个数）；
        ranges：统计像素值的区间；
        """
        src_hist = cv2.calcHist([src_img.astype(np.uint16)], [0], None, [RANGE], [0.0, float(RANGE)])
        dst_hist = cv2.calcHist([dst_img.astype(np.uint16)], [0], None, [RANGE], [0.0, float(RANGE)])
        src_acc_prob_hist = self.get_acc_prob_hist(src_hist, RANGE)
        dst_acc_prob_hist = self.get_acc_prob_hist(dst_hist, RANGE)
        diff_acc_prob = abs(
            np.tile(src_acc_prob_hist.reshape(RANGE, 1), (1, RANGE)) - dst_acc_prob_hist.reshape(1, RANGE))
        # 求出各阶灰度对应的差值的最小值，该最小值对应的灰度阶即为映射之后的灰度阶 垂直方向最小值所在索引
        table = np.argmin(diff_acc_prob, axis=0)
        # 将源图像按照求出的映射关系做映射
        result = self.grayscaleMapping(dst_img, table)
        # 存储图像
        return result

    def find_corresponding_standard_img(self, standard_files, num_standard_files, standard_dir, position):
        """
         寻找对应位置的标准图像，从标准组中找到位于特定位置的图片，position取值为[0, 1]
         :param standard_dir: 标准图像文件夹路径
         :param position: 所需位置
         :return:
        """
        # standard_files = os.listdir(standard_dir)
        # num_standard_files = len(standard_files)

        # 计算指定位置对应的文件名在列表中的索引
        index = int(num_standard_files * position)
        # print("index:" + str(index))
        # print("standard_files[index]:" + standard_files[index])
        # print("")

        # 返回指定位置对应的文件名
        return os.path.join(standard_dir, standard_files[index])

    def aveBatch(self, nii_path, in_path, out_path, src_path):
        """
         直方图规定化，把in_path下的图像按照src_img图像进行规定化
         # 23/03/13修改: 以下路径均为文件夹
         :param in_path:  待处理图像输入路径
         :param out_path: 输出路径
         :param src_path: 源图像路径
         :return:
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
        """
        在代码中，enumerate(c)
        函数可以同时遍历列表元素和它们的下标。每处理一张图片时，都可以通过
        index / len(c)
        计算出当前图片在所有图片中的位置，位置的值为一个[0, 1]
        范围内的浮点数。
        """
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
            standard_img_path = self.find_corresponding_standard_img(standard_files, num_standard_files, src_path,
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
            result = self.hist_specify(src_img, dst_img, RANGE)

            # 若为关键帧，计算信噪比
            if index == middle_index:
                middle_frame_snr = self.snr(result)

            # print(data1.shape)
            data1[mark] = result
            # print(data1.shape)
            # print("第%s张图像处理完成" % mark)
            # print(data1.shape)
            mark += 1

        print("$测试$ 程序走到这里了嘛 thread.py:404")
        # 根据关键帧信噪比，进行白平衡或边缘强化
        # TODO: zzl补充：刘铠嘉实现在前端界面上设置0.50（就是下面这个） 10:29会崩溃掉
        # app_eiseg = APP_EISeg()  # 引入主界面的控件，便于获取前端数据
        # snr_threshold = (app_eiseg.snr_setted()) / 100.0
        snr_threshold = 0.50
        # print("$测试$ 白平衡-信噪比SNR-阈值=", snr_threshold)
        if middle_frame_snr > snr_threshold:
            print("白平衡, 阈值=", snr_threshold)
            # for i in range(len(data1)):
            #     data1[i] = self.white_balance(data1[i])
                # RANGE = data1[i].max() + 1
                # # 直方图规定化（匹配）
                # # src_img为标准图像
                # data1[i] = self.hist_specify(src_img, data1[i], RANGE)
        else:
            print("边缘强化")
            for i in range(len(data1)):
                # edge_enhance(self, img, low_threshold, high_threshold)
                # low_threshold：低阈值，用于过滤较弱的边缘像素。
                # 低于该阈值的边缘像素会被认为是噪声或不重要的边缘，会被抑制掉，不被检测出来。
                # high_threshold：高阈值，用于标记较强的边缘像素。
                # 高于该阈值的边缘像素会被认为是明显的边缘，会被保留下来作为最终的边缘结果。

                # TODO: 刘铠嘉实现在前端界面上设置low_threshold和high_threshold
                # low_threshold = app_eiseg.elt_setted()  # 获取前端设置的边缘强化低阈值
                # high_threshold = app_eiseg.eht_setted()
                low_threshold, high_threshold = 60, 150

                data1[i] = self.edge_enhance(data1[i], low_threshold, high_threshold)
                # RANGE = data1[i].max() + 1
                # # 直方图规定化（匹配）
                # # src_img为标准图像
                # data1[i] = self.hist_specify(src_img, data1[i], RANGE)

        picture_out_path = out_path + ".nii"
        out = sitk.GetImageFromArray(data1)
        sitk.WriteImage(out, picture_out_path)
        # print('所有图像均值化完毕')

    def setResample(self, imagesitk, newspace=(1, 1, 1)):
        euler3d = sitk.Euler3DTransform()
        nh, nw, nq = imagesitk.GetSize()
        dh, dw, dq = imagesitk.GetSpacing()
        origin = imagesitk.GetOrigin()
        direction = imagesitk.GetDirection()
        size = (int(nh * dh / newspace[0]), int(nw * dw / newspace[1]), int(nq * dq / newspace[2]))
        imagesitk = sitk.Resample(imagesitk, size, euler3d, sitk.sitkLinear, origin, newspace, direction)
        return imagesitk

    def load_medical_data(self, f):
        """
        load data of different format into numpy array, return data is in xyz
        f: the complete path to the file that you want to load
        """
        f_nps = []
        itkimage = f
        if itkimage.GetDimension() == 4:
            slicer = sitk.ExtractImageFilter()
            s = list(itkimage.GetSize())
            s[-1] = 0
            slicer.SetSize(s)
            images = []
            for slice_idx in range(itkimage.GetSize()[-1]):
                slicer.SetIndex([0, 0, 0, slice_idx])
                sitk_volume = slicer.Execute(itkimage)
                images.append(sitk_volume)
                images = [sitk.DICOMOrient(img, "SLP") for img in images]
                f_nps = [sitk.GetArrayFromImage(img) for img in images]
        else:
            image = sitk.DICOMOrient(itkimage, "SLP")
            f_np = sitk.GetArrayFromImage(image)
            f_nps = [f_np]

        return f_nps

    # TODO: zzl补充：刘铠嘉实现在前端界面上窗宽窗位
    def normalize(self, frame, ww=300, wc=300):
        # ————————————————————————————————————————
        # 如果运行中 ww和wc出现问题请注释这一部分"从前端拖动条获取窗位窗宽数据"的代码
        # from app import APP_EISeg
        # app_eiseg = APP_EISeg()
        # ww = app_eiseg.ww()
        # wc = app_eiseg.wc()
        print("$测试$ 窗位wc=", wc, ", 窗宽ww=", ww)
        # ————————————————————————————————————————
        wl = wc - ww / 2
        wh = wc + ww / 2
        frame = frame.astype("float16")
        np.clip(frame, wl, wh, out=frame)
        frame = (frame - wl) / ww * 255
        frame = frame.astype(np.uint8)
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        return frame

    def array_to_video(self, array_data, video_path, fps=15):
        Index = video_path.rfind(r"\\")
        a = video_path[Index:]
        video_path2 = video_path.replace(a, "")
        print(video_path2)
        if not os.path.exists(video_path2):
            os.makedirs(video_path2)
        h, w, s = array_data.shape
        fourcc = VideoWriter_fourcc(*"mp4v")
        videoWriter = cv2.VideoWriter(video_path, fourcc, fps, (w, h))

        for idx in tqdm(range(s)):
            frame = array_data[:, :, idx]
            frame = self.normalize(frame)
            videoWriter.write(frame)

        videoWriter.release()
        cv2.destroyAllWindows()
        print(video_path)
        print("转视频结束！")

    # 计算信噪比
    def snr(self, img):
        # 将图像数据转换为浮点型，方便计算
        img = img.astype(np.float64)
        # 计算图像的平均值，即信号
        signal = np.mean(img)
        # 计算图像的标准差，即噪声
        noise = np.std(img)
        # 计算信噪比
        return signal / noise

    # 计算图像的灰度直方图,用于白平衡
    def get_histogram(self, img):
        # 获取图像的最大和最小值
        min_value = np.min(img)
        max_value = np.max(img)
        # 创建一个数组，表示灰度直方图的每个bin的频数
        histogram = np.zeros(max_value - min_value + 1, dtype=np.int64)
        # 遍历图像的每个像素，统计灰度直方图的频数
        for pixel in img.flat:
            histogram[pixel - min_value] += 1
        # 返回灰度直方图和最小值
        return histogram, min_value

    # 定义一个函数来对图像进行白平衡处理
    def white_balance(self, image):
        # 获取图像的灰度直方图和最小值
        histogram, min_value = self.get_histogram(image)
        # 计算图像的总像素数
        total = image.size
        # 计算累积分布函数（CDF）
        cdf = np.cumsum(histogram) / total
        # 找到累积分布函数中最接近0.01和0.99的索引，作为白平衡的下限和上限
        lower = np.argmin(np.abs(cdf - 0.01)) + min_value
        upper = np.argmin(np.abs(cdf - 0.99)) + min_value
        # 计算原始图像的平均灰度值
        mean_intensity = np.mean(image)
        # 计算白平衡后的图像的平均灰度值
        balanced_mean_intensity = (mean_intensity - lower) / (upper - lower) * 255
        # 计算平衡因子，使得平均灰度值保持不变
        factor = mean_intensity / balanced_mean_intensity
        # 对图像的每个像素进行平衡因子的缩放
        balanced = image * factor

        # 将图像的数值范围限制在0到255之间
        balanced = np.clip(balanced, 0, 255)

        # 返回白平衡后的图像，转换为整数类型
        return balanced.astype(np.int16)

    # 边缘强化
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

        # 将原图像与掩膜叠加
        # 关于数字参数：原图细节基本不变=1，突出边缘程度=0.5
        # TODO: 刘铠嘉，获取前端的原图像权重和掩膜权重数据
        # from app import APP_EISeg  # 先不要挪引用的位置，不然会报错QwQ
        # app_eiseg = APP_EISeg()  # 引入主界面的控件，便于获取前端数据
        # src_img_weight = app_eiseg.srcImgWeight_setted / 10.0
        # mask_weight = app_eiseg.maskWeight_setted / 10.0
        src_img_weight = 1
        mask_weight = 0.5
        print("$测试$ 原图像权重=", src_img_weight, ", 掩膜权重=", mask_weight)

        result = cv2.addWeighted(img, src_img_weight, mask, mask_weight, 0)
        return result

    def run(self):
        Classified_nii_path = r'D:\Medical_image\tx\all\2_Classified_nii'
        Classified_dcm_path = r'D:\Medical_image\tx\all\3_Classified_dcm'
        afterfolder_nii_path = r'D:\Medical_image\tx\all\4_afterfolder_nii'
        Unmarked_mp4_path = r'D:\Medical_image\tx\all\5_Unmarked_mp4'
        marked_mp4_path = r'D:\Medical_image\tx\all\6_marked_mp4'
        src_dcm_rath = r'D:\Medical_image\tx\all\7_src_dcm\XU YUAN BINM00135657_t1_vibe_fs_tra_post_two scans-19'

        self.process_max.emit(len(self.nii_path) * 5)
        process_num = 0

        for nii in self.nii_path:
            self.mutex.lock()  # 线程锁，用于控制暂停终止
            if self.isPause:
                self.cond.wait(self.mutex)
            if self.isCancel:
                # TODO: 线程终止后，是否要删除创建的文件
                self.process_num.emit(0)
                return
            nii2 = nii
            nii2 = nii.replace(Classified_nii_path, Classified_dcm_path)
            nii2 = nii2.replace('.nii', '')
            print(nii)
            print(nii2)
            Classified_dcm_path2 = nii2
            afterfolder_nii_path2 = nii2.replace(Classified_dcm_path, afterfolder_nii_path)
            print(Classified_dcm_path2)
            # TODO: 注释掉用于挑选

            img_in_nii = nib.load(nii)  # 去除非z轴图像
            orientation = nib.aff2axcodes(img_in_nii.affine)
            if not str(orientation[2]) == 'S':  # 'S' 和 'I' 表示“从头到脚”这个方向
                if not str(orientation[2]) == 'I':
                    print("非z轴，筛掉")
                    process_num += 5  # 进度条
                    self.process_num.emit(process_num)
                    self.mutex.unlock()  # 解锁
                    continue
            if not self.nii2dcm_single(nii, Classified_dcm_path2, True):  # 去除z轴太短的图像
                print("z轴太短，筛掉")
                process_num += 5  # 进度条
                self.process_num.emit(process_num)
                self.mutex.unlock()  # 解锁
                continue

            if self.isPause:  # 是否有暂停事件
                self.cond.wait(self.mutex)
            if self.isCancel:
                # TODO: 线程终止后，是否要删除创建的文件
                self.process_num.emit(0)
                return
            process_num += 1  # 进度条
            self.process_num.emit(process_num)

            self.aveBatch(nii, Classified_dcm_path2, afterfolder_nii_path2, src_dcm_rath)

            if self.isPause:  # 是否有暂停事件
                self.cond.wait(self.mutex)
            if self.isCancel:
                # TODO: 线程终止后，是否要删除创建的文件
                self.process_num.emit(0)
                return
            process_num += 1  # 进度条
            self.process_num.emit(process_num)
            # TODO: 此处传参应为 afterfolder_nii_path2
            image = sitk.ReadImage(afterfolder_nii_path2)  # nii
            # image = setPixel(image)
            image = self.setResample(image)
            if self.isPause:  # 是否有暂停事件
                self.cond.wait(self.mutex)
            if self.isCancel:
                # TODO: 线程终止后，是否要删除创建的文件
                self.process_num.emit(0)
                return
            process_num += 1  # 进度条
            self.process_num.emit(process_num)

            total_data = self.load_medical_data(image)
            if self.isPause:  # 是否有暂停事件
                self.cond.wait(self.mutex)
            if self.isCancel:
                # TODO: 线程终止后，是否要删除创建的文件
                self.process_num.emit(0)
                return
            process_num += 1  # 进度条
            self.process_num.emit(process_num)

            Unmarked_mp4_path2 = afterfolder_nii_path2.replace(afterfolder_nii_path, Unmarked_mp4_path) + '.mp4'
            for data in total_data:
                self.array_to_video(data, Unmarked_mp4_path2)
            Index = Unmarked_mp4_path2.rfind(r"\\")
            a = Unmarked_mp4_path2[Index:]
            Unmarked_mp4_path3 = Unmarked_mp4_path2.replace(a, "")

            if self.isPause:  # 是否有暂停事件
                self.cond.wait(self.mutex)
            if self.isCancel:
                # TODO: 线程终止后，是否要删除创建的文件
                self.process_num.emit(0)
                return
            process_num += 1
            self.process_num.emit(process_num)
            self.mutex.unlock()  # 解锁

        self.targetDir.emit(Unmarked_mp4_path3)
        # return Unmarked_mp4_path3
