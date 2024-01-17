import os.path as osp

import SimpleITK as sitk
import cv2
import numpy as np
from tqdm import tqdm


def load_medical_data(f):
    filename = osp.basename(f).lower()  # 获取文件名，并将其转换为小写
    # 根据文件扩展名选择加载数据的方式
    if filename.endswith((".nii", ".nii.gz", ".dcm")):
        itkimage = sitk.ReadImage(f)
        # 检查图像的维度是否为4，如果是，则进行切片处理
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
            # 如果维度不为4，直接进行方向调整并转换为numpy数组
            image = sitk.DICOMOrient(itkimage, "SLP")
            f_np = sitk.GetArrayFromImage(image)
            f_nps = [f_np]
    else:
        raise NotImplementedError

    return f_nps


def normalize(frame, ww, wc=300):
    wl = wc - ww / 2
    wh = wc + ww / 2
    frame = frame.astype("float16")
    np.clip(frame, wl, wh, out=frame)
    frame = (frame - wl) / ww * 255
    frame = frame.astype(np.uint8)
    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    return frame


def array_to_video(array_data, video_path, www, fps=15):
    h, w, s = array_data.shape
    fourcc = cv2.VideoWriter.fourcc(*"mp4v")  # 编码器
    videoWriter = cv2.VideoWriter(video_path, fourcc, fps, (w, h))
    for idx in tqdm(range(s)):
        frame = array_data[:, :, idx]
        frame = normalize(frame, www)  # 对帧进行归一化处理
        videoWriter.write(frame)
    videoWriter.release()
    cv2.destroyAllWindows()
    print("转视频结束！")


if __name__ == "__main__":
    path = r"E:\Project\medical-image-preprocessing\data\3_classified_nii\XU YUAN BIN_M00135657"
    nii_path = path + r"\20-t1_vibe_fs_tra_post_two scans.nii"
    for ww in range(0, 1001, 100):
        video_path = path + r"\temp-" + str(ww) + ".mp4"
        total_data = load_medical_data(nii_path)
        for data in total_data:
            array_to_video(data, video_path, ww)

