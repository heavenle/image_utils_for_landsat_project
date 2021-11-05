from osgeo import gdal
from skimage import io
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
import glob
from collections import  Counter

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


def statistical(mask_path, n_class, show=None):
    """
    本函数是为了统计mask中各个类别站总类的数量。
    :param n_class: 类别总数
    :param mask_path: mask所在地址。
    :return: list
    """
    result = [0 for i in range(0, n_class)]
    count = 0
    for mask_img_path in mask_path:
        count += 1
        print('deal the {}/{} img'.format(count, len(mask_path)))
        if os.path.basename(mask_img_path).split('.')[-1] == 'tif':
            # img = gdal_read_img(mask_img_path)
            img = io.imread(mask_img_path)
            # img[img > 811] = 900
            # img = img // 100
        elif os.path.basename(mask_img_path).split('.')[-1] == 'npy':
            img = np.load(mask_img_path)
        else:
            print('current suffix is not exist!')
            exit(0)

        for label, num in Counter(img.reshape(-1)).items():
            result[int(label)] += num
    print(result)
        # with open(mask_img_path.split('.')[0].split('\\')[-1] + '.txt', 'w') as f:
        #     for l in range(0, n_class):
        #         print('label {} is {:.2f}%'.format(l, result[l] / sum(result) * 100))
        #         f.write('label {} is {:.2f}%'.format(l, result[l] / sum(result) * 100))
        #         f.write('\n')
        # plt_bingzhuantu(result, n_class, mask_img_path.split('.')[0].split('\\')[-1] + '.png')
    if show == True:
        with open('统计结果/GF2/320_org_result.txt', 'w') as f:
            for l in range(0, len(result)):
                print('label {} is {:.2f}%'.format(l, result[l]/sum(result)*100))
                f.write('label {} is {:.2f}%'.format(l, result[l]/sum(result)*100))
                f.write('\n')
        plt_bingzhuantu(result, n_class)


def plt_bingzhuantu(list, n_classes, img_name=None):
    plt.figure(figsize=(6, 9))  # 调节图形大小
    labels = [i for i in range(0, n_classes)]  # 定义标签
    sizes = list  # 每块值
    # colors = ['red', 'yellowgreen', 'lightskyblue', 'yellow', 'black', '']  # 每块颜色定义
    # explode = (0, 0, 0, 0)  # 将某一块分割出来，值越大分割出的间隙越大
    patches, text1, text2 = plt.pie(sizes,
                                    labels=labels,
                                    autopct='%3.2f%%',  # 数值保留固定小数位
                                    shadow=False,  # 无阴影设置
                                    startangle=90,  # 逆时针起始角度设置
                                    pctdistance=0.6)  # 数值距圆心半径倍数距离
    # patches饼图的返回值，texts1饼图外label的文本，texts2饼图内部的文本
    # x，y轴刻度设置一致，保证饼图为圆形
    plt.axis('equal')
    if img_name is None:
        plt.savefig('统计结果/GF2/320_org_all.png', )
    else:
        plt.savefig(img_name)
    # plt.show()
    plt.close()



if __name__ == '__main__':
    # plt_bingzhuantu([0.1401, 0.1088, 0.0403, 0.2455, 0.0471, 0.1441, 0.0366, 0.1057, 0.0323, 0.0995], 10)

    root_path = r'D:\project\crop\GF1_data_320_25\masks'
    # band_file  = os.listdir(root_path)
    # mask_list = []
    # for band_name in band_file:
    #     mask_list =  mask_list + glob.glob(os.path.join(root_path, band_name, '*.tif'))
    # print(mask_list)
    # mask_list = []
    # mask_list = mask_list + glob.glob(os.path.join(root_path, '*.npy'))
    # new = []
    # for i in range(0, len(mask_list)):
    #     if len(mask_list[i]) >= 70:
    #         new.append(mask_list[i])
    mask_list = glob.glob(os.path.join(root_path, '*.tif'))
    statistical(mask_list, 10, show=True)