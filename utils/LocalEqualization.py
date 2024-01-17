import cv2


def local_equalization(input_path, mask_path, output_path):
    """
    对输入图像的对应掩模局部进行直方图均衡化
    :param input_path: 待规定化png图像
    :param mask_path: 局部规定化的掩模
    :param output_path: 规定化后的掩模下的图像
    """
    # 读取PNG图像和掩膜
    image = cv2.imread(input_path)
    mask = cv2.imread(mask_path, 0)  # 将掩膜图像转换为灰度图像

    # print(image.dtype, mask.dtype)

    if image.shape[:2] != mask.shape:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]))

    masked_image = cv2.bitwise_and(image, image, mask=mask)

    # 将掩膜位置的像素提取出来
    masked_pixels = masked_image[mask > 0]

    equ_histogram = cv2.equalizeHist(masked_pixels)

    # 将直方图均衡化后的像素放回图像中
    equ_index = 0
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if mask[i, j] > 0:
                masked_image[i, j] = equ_histogram[equ_index]
                equ_index += 1

    cv2.imwrite(output_path, masked_image)
    print("SUCCESS: 局部直方图均衡化完成")


input_path = '../localData/slice_0108.png'
mask_path = '../localData/mask00108.png'
output_path = '../localData/output.png'
local_equalization(input_path, mask_path, output_path)
