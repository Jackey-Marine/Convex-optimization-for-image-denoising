import cv2
import os

def to_grayscale(img_folder, save_folder):
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    for filename in os.listdir(img_folder):
        if filename.endswith(".jpg") or filename.endswith(".png"): 
            # 构造源文件的完整路径
            img_path = os.path.join(img_folder, filename)
            # 读取图像
            img = cv2.imread(img_path)
            # 转换图像为灰度
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # 构造保存的文件路径
            save_path = os.path.join(save_folder, filename)
            # 将灰度图像保存到指定路径
            cv2.imwrite(save_path, gray_img)

    print("所有图片已成功转换为灰度图，并保存在 " + save_folder + " 目录下。")


# main
# 将彩色图片转换为灰度图片
img_folder = 'D:\MyFile\Development\InsulatorDataSet\image\Insulators\color_images'
save_folder = 'D:\MyFile\Development\InsulatorDataSet\image\Insulators\gray_images'
# 调用函数并输入你的图片目录和保存目录
to_grayscale(img_folder, save_folder)