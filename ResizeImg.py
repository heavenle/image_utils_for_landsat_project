import glob
import os
import cv2
import ShowMaskNpy
import random
import numpy as np

def Resize_img(mask_list, size, save_path, num, shuffle=None):
    """
    将mask 图片resize到指定尺寸并保存。
    :param mask_list: 包含所有mask图片的路径
    :param size: resize的大小
    :param save_path: 保存路径
    :param num:保存的个数
    :param shuffle:是否打乱
    :return:
    """
    if shuffle == True:
        random.shuffle(mask_list)

    if not os.path.exists(os.path.join(save_path, 'org_mask')):
        os.mkdir(os.path.join(save_path, 'org_mask'))
    if not os.path.exists(os.path.join(save_path, 'resize_mask')):
        os.mkdir(os.path.join(save_path, 'resize_mask'))

    for i in range(0, num):
        print(i)
        mask = np.load(mask_list[i])
        if not os.path.exists(os.path.join(save_path, 'resize_mask', os.path.basename(mask_list[i]).split('.')[0])):
            os.mkdir(os.path.join(save_path, 'resize_mask', os.path.basename(mask_list[i]).split('.')[0]))

        mask_img = ShowMaskNpy.show_mask_npy(mask)
        cv2.imwrite(os.path.join(save_path, 'org_mask', os.path.basename(mask_list[i]).replace('.npy', '.jpg')), mask_img)

        resize_mask_img1r = cv2.resize(mask, (size,size), interpolation=cv2.INTER_NEAREST)
        resize_mask_img1 = ShowMaskNpy.show_mask_npy(resize_mask_img1r)

        cv2.imwrite(os.path.join(save_path, 'resize_mask', os.path.basename(mask_list[i]).split('.')[0],
                                 'INTER_NEAREST' + os.path.basename(mask_list[i]).replace('.npy', '.jpg')),
                    resize_mask_img1)


        resize_mask_img2r = cv2.resize(mask, (size,size), interpolation=cv2.INTER_LINEAR)
        resize_mask_img2 = ShowMaskNpy.show_mask_npy(resize_mask_img2r)

        cv2.imwrite(os.path.join(save_path, 'resize_mask', os.path.basename(mask_list[i]).split('.')[0],
                                 'INTER_LINEAR' + os.path.basename(mask_list[i]).replace('.npy', '.jpg')),
                    resize_mask_img2)

        resize_mask_img3r = cv2.resize(mask, (size,size), interpolation=cv2.INTER_CUBIC)
        resize_mask_img3 = ShowMaskNpy.show_mask_npy(resize_mask_img3r)
        cv2.imwrite(os.path.join(save_path, 'resize_mask', os.path.basename(mask_list[i]).split('.')[0],
                                 'INTER_CUBIC' + os.path.basename(mask_list[i]).replace('.npy', '.jpg')),
                    resize_mask_img3)

        resize_mask_img4r = cv2.resize(mask, (size,size), interpolation=cv2.INTER_LANCZOS4)
        resize_mask_img4 = ShowMaskNpy.show_mask_npy(resize_mask_img4r)
        cv2.imwrite(os.path.join(save_path, 'resize_mask', os.path.basename(mask_list[i]).split('.')[0],
                                 'INTER_LANCZOS4' + os.path.basename(mask_list[i]).replace('.npy', '.jpg')),
                    resize_mask_img4)

        resize_mask_img5r = cv2.resize(mask, (size,size), interpolation=cv2.INTER_AREA)
        resize_mask_img5 = ShowMaskNpy.show_mask_npy(resize_mask_img5r)
        cv2.imwrite(os.path.join(save_path, 'resize_mask', os.path.basename(mask_list[i]).split('.')[0],
                                 'INTER_AREA' + os.path.basename(mask_list[i]).replace('.npy', '.jpg')),
                    resize_mask_img5)


if __name__ == '__main__':
    mask_list = glob.glob(r'D:\project\crop\landsat\masks\201012639_clip_mask_7band_000046.npy')

    cv2.imwrite('D:\project\crop\output\liyi.jpg', np.load(r'D:\project\crop\landsat\images\201012639_clip_data_7band_000046.npy')[:,:,0:3])
    # save_path = r'D:\project\crop\output'
    # shuffle = True
    # Resize_img(mask_list, 640, save_path, 1, shuffle=True)
