from osgeo import gdal
from skimage import io
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
import glob
import Norm
import ShowMaskNpy
import skimage.io as io

def crop(img_path_list, save_path ,slide_window_size=1024, overlap_rate=0.25, mask = None):
    """
    opt:初始化参数。
    img：图像
    detector：由CtdetDetector生成的类，包含测试等函数。
    slide_window_size：滑框尺寸。
    overlap_rate：重叠率。取值在[0, 1]

    函数简介：本函是是将img大图进行滑框取图，然后检测每个小图。最后将小图的结果拼接到大图上。

    return：大图的检测框 tensor[num, 6]
    """
    if not os.path.exists(save_path + '/images'):
        os.makedirs(save_path + '/images')

    if not os.path.exists(save_path + '/masks'):
        os.makedirs(save_path + '/masks')
    count = 0
    for img_path in img_path_list:
        count +=1
        print('{}/{}'.format(count, len(img_path_list)))
        file_name = os.path.basename(img_path).split('.')[0]
        img = gdal_read_img(img_path)
        if mask == None:
            img = img.transpose(1, 2, 0)
            # img = Norm.norm_band(img)

        elif mask == True:
            img [img > 811] = 900
            img = img//100
        else:
            print('error mask')
            exit(0)

        height = img.shape[0]
        width = img.shape[1]
        # plt.subplot(1, 2, 1)
        # plt.imshow(img[:,:, 0])
        # plt.subplot(1, 2, 2)
        # plt.imshow(img_mask)
        # plt.show()

        # 滑框的重叠率
        overlap_pixel = int(slide_window_size * (1 - overlap_rate))

        # ------------------------------------------------------------------#
        #                处理图像各个维度尺寸过小的情况。
        # ------------------------------------------------------------------#
        if height - slide_window_size < 0:  # 判断x是否超边界，为真则表示超边界
            x_idx = [0]
        else:
            x_idx = [x for x in range(0, height, overlap_pixel)]

        if width - slide_window_size < 0:
            y_idx = [0]
        else:
            y_idx = [y for y in range(0, width, overlap_pixel)]
        # ----------------------------------------------------------------------#
        #                判断下x,y的尺寸问题，并且设置裁剪大小，方便后续进行padding。
        # ----------------------------------------------------------------------#
        cut_width = slide_window_size
        cut_height = slide_window_size

        if height - slide_window_size < 0 and width - slide_window_size >= 0:  # x小，y正常
            cut_width = slide_window_size
            cut_height = height
            switch_flag = 1
        elif height - slide_window_size < 0 and width - slide_window_size < 0:  # x小， y小
            cut_width = width
            cut_height = height
            switch_flag = 3
        elif height - slide_window_size >= 0 and width - slide_window_size < 0:  # x正常， y小
            cut_height = slide_window_size
            cut_width = width
            switch_flag = 2
        elif height - slide_window_size >= 0 and width - slide_window_size >= 0:
            switch_flag = 0

        # ----------------------------------------------------------------------#
        #                开始滑框取图，并且保存。
        # ----------------------------------------------------------------------#
        # count = 0
        for x_start in x_idx:
            if x_start + cut_height > height:
                x_start = height - cut_height
            for y_start in y_idx:
                # count += 1
                # print('now we deal {}th image'.format(count))
                if y_start + cut_width > width:
                    y_start = width - cut_width
                croped_img = img[x_start:x_start + cut_height, y_start:y_start + cut_width]

                # ----------------------------------------------------------------------#
                #                依据switch_flag的设置，进行padding。
                # ----------------------------------------------------------------------#
                if mask ==None:
                    temp = np.zeros((slide_window_size, slide_window_size, 3), dtype=np.uint8)
                    if switch_flag == 1:
                        # temp = np.zeros((croped_img.shape[0], cut_height, croped_img.shape[2]), dtype=np.uint8) #此为遥感图像
                        temp[0:cut_height, 0:croped_img.shape[1], :] = croped_img
                        croped_img = temp
                    elif switch_flag == 2:
                        # temp = np.zeros((cut_size, croped_img.shape[1], croped_img.shape[2]), dtype=np.uint8)
                        temp[0:croped_img.shape[0], 0:cut_width, :] = croped_img
                        croped_img = temp
                    elif switch_flag == 3:
                        temp[0:cut_height, 0:cut_width, :] = croped_img
                        croped_img = temp
                elif mask == True:
                    temp = np.zeros((slide_window_size, slide_window_size), dtype=np.uint8)
                    if switch_flag == 1:
                        # temp = np.zeros((croped_img.shape[0], cut_height, croped_img.shape[2]), dtype=np.uint8) #此为遥感图像
                        temp[0:cut_height, 0:croped_img.shape[1]] = croped_img
                        croped_img = temp
                    elif switch_flag == 2:
                        # temp = np.zeros((cut_size, croped_img.shape[1], croped_img.shape[2]), dtype=np.uint8)
                        temp[0:croped_img.shape[0], 0:cut_width] = croped_img
                        croped_img = temp
                    elif switch_flag == 3:
                        temp[0:cut_height, 0:cut_width] = croped_img
                        croped_img = temp

                # 剔除全零的图
                if max(croped_img.reshape(-1)) == 0:
                    continue

                if mask == None:
                    if croped_img.shape[2] > 3:
                        np.save(save_path + '/images/' +
                                img_path.split('.')[0].split('\\')[-1] + '_' + img_path.split('.')[0].split('\\')[-2] + '_'
                                + '0' * (6 - len(str(count))) + str(count) + '.npy', croped_img)
                    else:
                        io.imsave(save_path + '/images/' +
                                file_name + '_' + str(x_start) + '_' +str(y_start) + '.tif', croped_img.astype(np.uint8))

                elif mask == True:
                    # if croped_img.shape[2] > 3:
                    #     np.save(save_path + '/masks/' +
                    #             img_path.split('.')[0].split('\\')[-1].replace('mask', 'data') + '_' + img_path.split('.')[0].split('\\')[-2] + '_'
                    #             +'0' * (6-len(str(count))) + str(count) + '.npy', croped_img)
                    # else:
                    io.imsave(save_path + '/masks/' +
                              file_name + '_' + str(x_start) + '_' + str(y_start) + '.tif', croped_img.astype(np.uint8))



def gdal_read_img(img_path):
    """
    用gdal读取遥感数据
    :param
        img_path: 遥感数据地址 -> str
    :return
        遥感数据 -> np.ndarray
    """
    dataset = gdal.Open(img_path)
    width = dataset.RasterXSize
    height = dataset.RasterYSize

    im_data = dataset.ReadAsArray(0, 0, width, height)  # .astype(np.float32)
    return im_data



if __name__ == "__main__":
    # 初始化参数
    label_root_path = r'D:\database\高分2\clip_data\train\masks'
    img_root_path = r'D:\database\高分2\clip_data\train\images'
    slide_window_size = 512
    overlap_rate = 0.25

    save_path = './GF1_data_' + str(slide_window_size) + '_' + str(overlap_rate).split('.')[1]


    # band_file = os.listdir(label_root_path)
    # img_path = []
    # label_path = []
    # for band_name in band_file:
    #     img_path = img_path + glob.glob(os.path.join(img_root_path, band_name, '*.tif'))
    #     label_path = label_path + glob.glob(os.path.join(label_root_path, band_name, '*.tif'))
    img_path = glob.glob(os.path.join(img_root_path, '*.tif'))
    label_path = glob.glob(os.path.join(label_root_path, '*.tif'))

    crop(img_path, save_path, slide_window_size=slide_window_size, overlap_rate=overlap_rate, mask=None)
    crop(label_path, save_path, slide_window_size=slide_window_size, overlap_rate=overlap_rate, mask=True)
    # ShowMaskNpy.show_mut_band_img(os.path.join(save_path, 'images'), save_path)

    # show mask
    # mask_path = glob.glob(os.path.join(save_path, 'masks', '*.tif'))
    # count = 0
    if not os.path.exists(os.path.join(save_path, 'show_mask')):
        os.mkdir(os.path.join(save_path, 'show_mask'))
    mask_save_path = os.path.join(save_path, 'show_mask')
    #
    # for mask_name in mask_path:
    #     count += 1
    #     print('save mask img [{}/{}]'.format(count, len(mask_path)))
    #     # mask = np.load(mask_name)
    #     mask = io.imread(mask_name)
    #     mm = ShowMaskNpy.show_mask_npy(mask)
    #     cv2.imwrite(os.path.join(mask_save_path, os.path.basename(mask_name).split('.')[0] + '.jpg'), mm)
