import SimpleITK as sitk
import os


def correct_bias(in_file, out_file, image_type=sitk.sitkFloat64):
    """
    使用SimpleITK N4BiasFieldCorrection校正MRI图像的偏置场
    :param in_file: .nii.gz 文件的输入路径
    :param out_file: .nii.gz 校正后的文件保存路径
    :return: 校正后的nii文件全路径名
    """
    print("START: 准备进行偏置场矫正")
    input_image = sitk.ReadImage(in_file, image_type)
    output_image_s = sitk.N4BiasFieldCorrection(input_image, input_image > 0)
    print("LOADING: 偏置场矫正完成，正在写入图像")
    sitk.WriteImage(output_image_s, out_file)
    print("SUCCESS: 偏置场矫正完成")
    return os.path.abspath(out_file)


# in_file = r"../data/2_Classified_nii/XU YUAN BINM00135657t1/t1_vibe_fs_tra_post_two scans-19.nii"
in_file = r"../data/2_Classified_nii/ZHANG YA LING/t1_vibe_fs_cor_bh_p2-15.nii"
out_file = r"../data/2_Classified_nii/ZHANG YA LING/t1_vibe_fs_cor_bh_p2-15-corr.nii"
correct_bias(in_file, out_file)
