# {'background': '背景', 'forest': '森林', 'bush_wood': '荒丛', 'grass': '草地', 'wetland': '湿地',
#  'farmland': '农田', 'city': '城镇', 'desert': '荒漠', 'glacier': '冰川', 'bare_land': '裸地'}
import glob
import shutil

import numpy as np
from collections import Counter
import os
import skimage.io as io

def statistical(mask, n_class):
    """
    本函数是为了统计mask中各个类别站总类的数量。
    :param n_class: 类别总数
    :param mask_path: mask所在地址。
    :return: list
    """
    result = [0 for i in range(0, n_class)]

    for label, num in Counter(mask.reshape(-1)).items():
        result[label] = num/(mask.shape[0]**2)*100
    return result


def filter_small_label(mask_path_list, root_path, small_label, count_name):
    """
    如果mask中存在少样本的label，则将该mask提取出来。

    :param mask_path_list:所欲mask所在的path
    :return:
    """
    if not os.path.exists(os.path.join(root_path, 'small_data')):
        os.mkdir(os.path.join(root_path, 'small_data'))
        os.mkdir(os.path.join(root_path, 'small_data', 'all'))
        os.mkdir(os.path.join(root_path, 'small_data', 'all', 'images'))
        os.mkdir(os.path.join(root_path, 'small_data', 'all', 'masks'))

    # if not os.path.exists(os.path.join(root_path, 'zeros_img')):
    #     os.mkdir(os.path.join(root_path, 'zeros_img'))
    #     os.mkdir(os.path.join(root_path, 'zeros_img',  'images'))
    #     os.mkdir(os.path.join(root_path, 'zeros_img',  'masks'))

    count = 0
    for mask_path in mask_path_list:
        count +=1
        print('eopch: {}/{}'.format(count, len(mask_path_list)))
        suffix = os.path.splitext(mask_path)
        if suffix[1] == '.npy':
            mask = np.load(mask_path)
        elif suffix[1] == '.tif':
            mask = io.imread(mask_path)

        # if max(mask.reshape(-1)) == 0:
        #     print('exist zeros_img:', mask_path.split('\\')[-1])
        #     # print(sum(mask.reshape(-1)))
        #     shutil.move(mask_path, os.path.join(root_path, 'zeros_img', 'masks', mask_path.split('\\')[-1]))
        #     shutil.move(mask_path.replace('masks', 'images').replace('mask', 'data'),
        #                 os.path.join(root_path, 'zeros_img', 'images', mask_path.split('\\')[-1].replace('mask', 'data')))

        result = statistical(mask, 10)

        repeat_mask_list = []
        repeat_data_list = []

        for key, value in small_label.items():

            if not os.path.exists(os.path.join(root_path, 'small_data', small_label[key])):
                os.mkdir(os.path.join(root_path, 'small_data', small_label[key]))
                os.mkdir(os.path.join(root_path, 'small_data', small_label[key], 'images'))
                os.mkdir(os.path.join(root_path, 'small_data', small_label[key], 'masks'))
            if result[int(key)] > 10:
                count_name[key] += 1
                if os.path.exists(os.path.join(root_path, 'small_data', small_label[key])):
                    shutil.copyfile(mask_path,
                                    os.path.join(root_path, 'small_data', small_label[key], 'masks',
                                                 value + '0' * (6-len(str(count_name[key]))) + str(count_name[key]) + '.tif'))
                    shutil.copyfile(mask_path.replace('masks', 'images').replace('mask', 'data'),
                                    os.path.join(root_path, 'small_data', small_label[key], 'images',
                                                 value + '0' * (6-len(str(count_name[key]))) + str(count_name[key]) + '.tif')
                                    )
                    print('shutil',
                          os.path.join(root_path, 'small_data', small_label[key], 'masks' ,
                                       value + '0' * (6-len(str(count_name[key]))) + str(count_name[key]) + '.tif'),
                          '\n',
                          os.path.join(root_path, 'small_data', small_label[key], 'images',
                                       value + '0' * (6-len(str(count_name[key]))) + str(count_name[key]) + '.tif'))
                else:
                    print('path is error:{}'.format(os.path.join(root_path, 'small_data', small_label[key])))
                repeat_mask_list.append(mask_path)
                repeat_data_list.append(mask_path.replace('masks', 'images'))

        # copy_mask_file = np.unique(repeat_mask_list)
        # copy_data_file = np.unique(repeat_data_list)
        # for file in copy_mask_file:
        #     # print('file', file.split('\\')[-1])
        #     shutil.move(file, os.path.join(root_path, 'small_data', 'all', 'masks', file.split('\\')[-1]))
        #
        # for file in copy_data_file:
        #     # print('file', file.split('\\')[-1])
        #     shutil.move(file, os.path.join(root_path, 'small_data', 'all', 'images', file.split('\\')[-1]))




if __name__ == '__main__':
    mask_path_list = glob.glob(r'D:\project\crop\GF1_data_320_25\masks\*.tif')
    root_path = r'D:\project\crop\GF1_data_320_25'

    small_label = {'2':'bush_wood',
                   '4':'wetland',
                   '6':'city',
                   # '8':'glacier',
                   }

    count_name = {'2':0,
                   '4':0,
                   '6':0,
                   # '8':0,
                   }
    filter_small_label(mask_path_list, root_path , small_label, count_name)
