import os
import glob

import cv2
import numpy as np
import skimage.io as io
from PIL import Image
import Norm
import ShowMaskNpy
from DataRandomCat import crop, put_back_the_sample_cat

def extract_fore_ground(img_path, mask_path, label_path, small_label_key):
    """
    分别提取原图和标签的前景。
    :param img_path: 原始图像文件夹路径
    :param mask_path: 原始图像标签文件夹路径
    :return:
    """
    if not os.path.exists(os.path.join(label_path, 'foreground')):
        print('we will create new file[{}]'.format(os.path.join(label_path), 'foreground'))
        os.mkdir(os.path.join(label_path, 'foreground'))
        os.mkdir(os.path.join(label_path, 'foreground', 'images'))
        os.mkdir(os.path.join(label_path, 'foreground', 'masks'))
        os.mkdir(os.path.join(label_path, 'foreground', 'img'))
        os.mkdir(os.path.join(label_path, 'foreground', 'mask'))
    count = 0
    for mask_name in mask_path:
        count += 1
        print('{}:img [{}/{}]'.format(os.path.basename(label_path), count, len(mask_path)))
        mask = np.load(mask_name)
        temp  = np.where(mask == int(small_label_key), 1, 0)
        new_mask = mask * temp
        np.save(os.path.join(label_path, 'foreground', 'masks',
                             os.path.basename(mask_name)), new_mask)
        show_mask = ShowMaskNpy.show_mask_npy(new_mask)
        cv2.imwrite(os.path.join(label_path, 'foreground', 'mask',
                                 os.path.basename(mask_name).split('.')[0] + '.jpg'), show_mask)

        img_ = os.path.join(label_path, 'images', os.path.basename(mask_name))
        if img_ not in img_path:
            print('the img path[{}] is error'.format(img_))
            exit(0)
        else:
            img = np.load(img_)
            print(img.shape)
            img = img[:, :, 0:3]
            img = Norm.norm_band(img)
            new_img = np.zeros((img.shape[0], img.shape[1], img.shape[2]))
            for channel in range(0, 3):
                new_img[:, :, channel] = img[:, :, channel] * temp
            new_img = new_img.astype(np.uint8)
            np.save(os.path.join(label_path, 'foreground', 'images',
                                 os.path.basename(mask_name)), new_img)

            new_img = Image.fromarray(new_img)
            new_img.save(os.path.join(label_path, 'foreground', 'img',
                                 os.path.basename(mask_name).split('.')[0] + '.jpg'))


def fore_ground(root_path, small_label, num):
    for key, value in small_label.items():
        label_path = os.path.join(root_path, value)
        print('label[{}]'.format(value))
        images_list = glob.glob(os.path.join(label_path, 'images', '*.npy'))
        masks_list = glob.glob(os.path.join(label_path, 'masks', '*.npy'))
        extract_fore_ground(images_list, masks_list, label_path, key)

if __name__ == '__main__':
    root_path = r'D:\project\crop\data_256_25new\small_data'
    if os.path.exists(os.path.join(root_path, 'fore_ground')):
        os.mkdir(os.path.join(root_path, 'fore_ground'))
    save_path = os.path.join(root_path, 'fore_ground')

    small_label = {
                   '4':'wetland',
                   '6':'city',
                   '7':'desert',
                   '8':'glacier',
                   '9':'bare_land'
                   }

    fore_ground(root_path, small_label, 4)

    # big_pic = glob.glob(r'D:\project\crop\data_256_25new\images\*.npy')
    # count = 0
    # for name in big_pic:
    #     count += 1
    #     print('{}/{}'.format(count, len(big_pic)))
    #     img = np.load(name)
    #     img, m0, m1, m2 = Norm.norm_band(img)
    #
    #     new_img = Image.fromarray(img)
    #     new_img.save(os.path.join('output',
    #                               os.path.basename(name).split('.')[0]
    #                               + '_{}'.format(m0)
    #                               + '_{}'.format(m1)
    #                               + '_{}'.format(m2)
    #                               + '.jpg'))