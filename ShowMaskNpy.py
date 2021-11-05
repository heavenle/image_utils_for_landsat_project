import numpy as np
import cv2
import glob
import os
import skimage.io as io
colors = [(0, 0, 0), (0, 255, 0), (0, 255, 255), (255, 0, 255), (0, 0, 255), (255, 255, 0), (255, 0, 0), (139, 0, 130),
          (255, 255, 255), (145, 145, 50)]
# colors = [(0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (255, 0, 0), (0, 0, 0)]


class_value = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
def show_mask_npy(mask):
    """
    将mask用RGB显示出来

    :param mask: 路径
    :return:
    """
    if isinstance(mask, np.ndarray):
        one_channel_mask = mask
    else:
        sufix = os.path.splitext(mask)
        if sufix[1] == '.tif':
            one_channel_mask = io.imread(mask)
        elif sufix[1] == '.npy':
            one_channel_mask = np.load(mask)
        else:
            print('error suffix, suffix ==', sufix)
            exit(0)

    height, width = one_channel_mask.shape
    img = np.zeros((height, width, 3))
    # one_channel_mask = np.argmax(mask_temp.detach().cpu().numpy(), axis=-1)

    for n_class_value in class_value:
        # linshi shezhi
        img[:, :, 0] += ((one_channel_mask[:, :] == n_class_value) * colors[n_class_value][0]).astype(
            'uint8')
        img[:, :, 1] += ((one_channel_mask[:, :] == n_class_value) * colors[n_class_value][1]).astype(
            'uint8')
        img[:, :, 2] += ((one_channel_mask[:, :] == n_class_value) * colors[n_class_value][2]).astype(
            'uint8')
    if isinstance(mask, str):
        if not os.path.exists(os.path.join(os.path.dirname(os.path.dirname(mask)),'show_mask')):
            os.mkdir(os.path.join(os.path.dirname(os.path.dirname(mask)), 'show_mask'))
        io.imsave(os.path.join(os.path.join(os.path.dirname(os.path.dirname(mask)),'show_mask', os.path.basename(mask).split('.')[0] + '.jpg')), img)
    else:
        return img

def show_mut_band_img(img_path, save_path, bandnum=3):
    if os.path.exists(os.path.join(save_path, 'show_img')):
        for i in glob.glob(os.path.join(save_path, 'show_img', '*')):
            os.remove(i)
    else:
        os.mkdir(os.path.join(save_path, 'show_img'))

    img_list = glob.glob(os.path.join(img_path, '*.npy'))
    count = 0
    for img_name in img_list:
        count += 1
        print('save data img [{}/{}]'.format(count, len(img_list)))
        img = np.load(img_name)
        img = img[:, :, 0:bandnum]
        cv2.imwrite(os.path.join(save_path, 'show_img', os.path.basename(img_name).split('.')[0] + '.jpg'), img)


if __name__ == '__main__':
    img_path = glob.glob(r'D:\project\crop\GF1_data_320_25\small_data\bush_wood\masks\*.tif')
    count = 0
    for img_name_path in img_path:
        count += 1
        print('{}/{}'.format(count, len(img_path)))
        show_mask_npy(img_name_path)
    # show_mask_npy(r'D:\project\crop\data_320_25\small_data\city\201008_clip_mask_7band_000007.npy')
    # img = np.load(r'D:\project\crop\data_320_25\images\201008_clip_data_7band_000007.npy')
    # cv2.imwrite('liyi.jpg', img[:, :, 0:3])
