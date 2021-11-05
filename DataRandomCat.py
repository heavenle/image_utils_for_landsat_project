import os
import glob
import cv2
import numpy as np
from Norm import norm_band

# 将data和mask都裁剪并存储
def crop(images_list, mask_list, label_path, crop_num=4):
    if not os.path.exists(os.path.join(label_path, 'crop')):
        os.mkdir(os.path.join(label_path, 'crop'))
        os.mkdir(os.path.join(label_path, 'crop', 'images'))
        os.mkdir(os.path.join(label_path, 'crop', 'masks'))

    for i in range(0, len(images_list)):
        print('nums[{}/{}]'.format(i+1, len(images_list)))
        img = np.load(images_list[i])
        mask = np.load(mask_list[i])
        height, width, _ = img.shape
        slide_window_size = (height * 2) // crop_num
        x_ids = [i for i in range(0, height - slide_window_size + 1, slide_window_size)]
        y_ids = [i for i in range(0, width - slide_window_size + 1, slide_window_size)]

        crop_img_root_path = os.path.join(label_path, 'crop', 'images')
        crop_mask_root_path = os.path.join(label_path, 'crop', 'masks')

        count = 0
        for x_start in x_ids:
            for y_start in y_ids:
                count += 1
                crop_img = img[x_start:x_start + slide_window_size, y_start:y_start + slide_window_size, 0:3]
                crop_img = norm_band(crop_img)
                crop_mask = mask[x_start:x_start + slide_window_size, y_start:y_start + slide_window_size]
                if max(crop_mask.reshape(-1)) == 0:
                    print('current img and mask is zero image and we discard it.')
                    continue
                # cv2.imwrite(os.path.join(crop_img_root_path,
                #                          os.path.basename(images_list[i]).split('.')[0] + '_' + str(count) + '.jpg'),
                #             crop_img)
                np.save(os.path.join(crop_img_root_path,
                                          os.path.basename(images_list[i]).split('.')[0] + '_' + str(count) + '.npy'),
                             crop_img)
                np.save(os.path.join(crop_mask_root_path,
                                     os.path.basename(mask_list[i]).split('.')[0] + '_' + str(count) + '.npy'),
                        crop_mask)


# 将裁剪后的图拼接成
def cat(label_path, crop_num=4):

    crop_img_list = glob.glob(os.path.join(label_path, 'crop', 'images', '*.jpg'))
    # crop_mask_list = glob.glob(os.path.join(label_path, 'crop', 'masks', '*.npy'))
    if not os.path.exists(os.path.join(label_path, 'data_augmentation')):
        os.mkdir(os.path.join(label_path, 'data_augmentation', 'img'))
        os.mkdir(os.path.join(label_path, 'data_augmentation', 'images'))
        os.mkdir(os.path.join(label_path, 'data_augmentation', 'masks'))

    img_count = 1

    height = 0
    width = 0

    while len(crop_img_list) > crop_num:
        print('cat {} imgs'.format(img_count))
        chocie_list = np.random.choice(crop_img_list, crop_num, replace=None)
        img_list = []
        mask_list = []

        for img_name in chocie_list:
            # img = cv2.imread(img_name)
            img = np.load(img_name)
            height, width, channel = img.shape
            img_list.append(img)
            # 读取masks
            mask = np.load(os.path.join(label_path, 'crop', 'masks',
                                        os.path.basename(img_name).replace('data', 'mask').replace('.jpg', '.npy')))
            mask_list.append(mask)
        temp_img = np.zeros((height*len(img_list)//2, width*len(img_list)//2, 3)    )
        temp_mask = np.zeros((height * len(mask_list)// 2, width * len(mask_list) // 2))

        slide_window_size = height
        x_ids = [i for i in range(0, slide_window_size*crop_num//2 - slide_window_size + 1, slide_window_size)]
        y_ids = [i for i in range(0, slide_window_size*crop_num//2 - slide_window_size+ 1, slide_window_size)]

        count = 0
        for x_start in x_ids:
            for y_start in y_ids:
                temp_img[x_start:x_start + slide_window_size, y_start:y_start + slide_window_size, 0:3] = img_list[count]
                temp_mask[x_start:x_start + slide_window_size, y_start:y_start + slide_window_size] = mask_list[count]
                count += 1

        cv2.imwrite(os.path.join(label_path, 'data_augmentation', 'img',
                                 os.path.basename(label_path) + '_' + str(img_count) + '.jpg'), temp_img)
        np.save(os.path.join(label_path, 'data_augmentation', 'images',
                                 os.path.basename(label_path) + '_' + str(img_count) + '.npy'), temp_img)
        np.save(os.path.join(label_path, 'data_augmentation', 'masks',
                             os.path.basename(label_path) + '_' + str(img_count) + '.npy'), temp_mask)
        img_count += 1
        for choice_name in chocie_list:
            crop_img_list.remove(choice_name)


def put_back_the_sample_cat(label_path, crop_num=4, save_num=1000, foreground=None):
    crop_img_list = glob.glob(os.path.join(label_path, 'crop', 'images', '*.npy'))
    # crop_mask_list = glob.glob(os.path.join(label_path, 'crop', 'masks', '*.npy'))
    if not os.path.exists(os.path.join(label_path, 'data_augmentation')):
        os.mkdir(os.path.join(label_path, 'data_augmentation'))
        os.mkdir(os.path.join(label_path, 'data_augmentation', 'img'))
        os.mkdir(os.path.join(label_path, 'data_augmentation', 'images'))
        os.mkdir(os.path.join(label_path, 'data_augmentation', 'masks'))
    height = 0
    width = 0

    for num in range(1, save_num+1):
        print('cat{}:[{}/{}] imgs'.format( os.path.basename(os.path.dirname(label_path)),
                                            num, save_num))
        chocie_list = np.random.choice(crop_img_list, crop_num, replace=None)
        img_list = []
        mask_list = []

        for img_name in chocie_list:
            # img = cv2.imread(img_name)
            img = np.load(img_name)
            height, width, channel = img.shape
            img_list.append(img)
            # 读取masks
            mask = np.load(os.path.join(label_path, 'crop', 'masks',
                                        os.path.basename(img_name).replace('data', 'mask').replace('.jpg', '.npy')))
            mask_list.append(mask)
        temp_img = np.zeros((height * len(img_list) // 2, width * len(img_list) // 2, 3))
        temp_mask = np.zeros((height * len(mask_list) // 2, width * len(mask_list) // 2))

        slide_window_size = height
        x_ids = [i for i in range(0, slide_window_size * crop_num // 2 - slide_window_size + 1, slide_window_size)]
        y_ids = [i for i in range(0, slide_window_size * crop_num // 2 - slide_window_size + 1, slide_window_size)]

        count = 0
        for x_start in x_ids:
            for y_start in y_ids:
                temp_img[x_start:x_start + slide_window_size, y_start:y_start + slide_window_size, 0:3] = img_list[
                    count]
                temp_mask[x_start:x_start + slide_window_size, y_start:y_start + slide_window_size] = mask_list[count]
                count += 1
        if not foreground:
            cv2.imwrite(os.path.join(label_path, 'data_augmentation', 'img',
                                     os.path.basename(label_path) + '_' + str(num) + '.jpg'), temp_img)
            np.save(os.path.join(label_path, 'data_augmentation', 'images',
                                 os.path.basename(label_path) + '_' + str(num) + '.npy'), temp_img)
            np.save(os.path.join(label_path, 'data_augmentation', 'masks',
                                 os.path.basename(label_path) + '_' + str(num) + '.npy'), temp_mask)
        else:
            cv2.imwrite(os.path.join(label_path, 'data_augmentation', 'img',
                                     os.path.basename(os.path.dirname(label_path)) + '_' + str(num) + '.jpg'), temp_img)
            np.save(os.path.join(label_path, 'data_augmentation', 'images',
                                 os.path.basename(os.path.dirname(label_path)) + '_' + str(num) + '.npy'), temp_img)
            np.save(os.path.join(label_path, 'data_augmentation', 'masks',
                                 os.path.basename(os.path.dirname(label_path)) + '_' + str(num) + '.npy'), temp_mask)


# 通过将数据随机裁剪，然后拼接的方式完成数据增强。生成的数据都是3band。
def data_random_cat(root_path, small_label, num=4, foreground=None):
    for key, value in small_label.items():
        if not foreground:
            label_path = os.path.join(root_path, value)
        else:
            label_path = os.path.join(root_path, value, 'foreground')
        print('label[{}]'.format(value))
        images_list = glob.glob(os.path.join(label_path, 'images', '*.npy'))
        masks_list = glob.glob(os.path.join(label_path, 'masks', '*.npy'))
        # crop(images_list, masks_list, label_path, crop_num=num)
        # cat(label_path, crop_num=num)
        put_back_the_sample_cat(label_path, crop_num=num, save_num=2000, foreground=foreground)

if __name__ == '__main__':
    root_path = r'D:\project\crop\data_256_25new\small_data'
    small_label = {
                   '4':'wetland',
                   '6':'city',
                   '7':'desert',
                   '8':'glacier',
                   '9':'bare_land'
                   }
    data_random_cat(root_path, small_label, 4, foreground=True)

    # root_path = r'D:\project\crop\data_256_25new\small_data'
    # small_label = {
    #                '-1':'all_foreground',
    #                }
    # data_random_cat(root_path, small_label, 4)
