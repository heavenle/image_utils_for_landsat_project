from skimage import io
import numpy as np
import glob
import cv2
from crop import gdal_read_img

def norm_band(label):

    # 将图片中负值赋值为零
    label[label < 0] = 0

    img1 = np.zeros((label.shape[0], label.shape[1], 3))
    imgRGBA = np.zeros((label.shape[0], label.shape[1], 3))

    img1[:, :, 0] = label[:, :, 2]
    min_bit0 = np.min(img1[:, :, 0])
    max_bit0 = np.max(img1[:, :, 0])
    if max_bit0 > 0:
        if min_bit0 == 0:
            ff = img1[:, :, 0].flatten()
            min_bit0 = sorted(set(list(ff)))[1]
            max_bit0 = sorted(set(list(ff)))[-1]
            img1[:, :, 0][img1[:, :, 0] == 0] = min_bit0
            imgRGBA[:, :, 0] = np.rint(255 * ((img1[:, :, 0] - min_bit0) / (max_bit0 - min_bit0)))
        else:
            imgRGBA[:, :, 0] = np.rint(255 * ((img1[:, :, 0] - min_bit0) / (max_bit0 - min_bit0)))
    imgRGBA[:, :, 0] = imgRGBA[:, :, 0].astype(np.uint8)

    img1[:, :, 1] = label[:, :, 1]
    min_bit1 = np.min(img1[:, :, 1])
    max_bit1 = np.max(img1[:, :, 1])
    if max_bit1 > 0:
        if min_bit1 == 0:
            ff = img1[:, :, 1].flatten()
            min_bit1 = sorted(set(list(ff)))[1]
            max_bit1 = sorted(set(list(ff)))[-1]
            img1[:, :, 1][img1[:, :, 1] == 0] = min_bit1
            imgRGBA[:, :, 1] = np.rint(255 * ((img1[:, :, 1] - min_bit1) / (max_bit1 - min_bit1)))
        else:
            imgRGBA[:, :, 1] = np.rint(255 * ((img1[:, :, 1] - min_bit1) / (max_bit1 - min_bit1)))
    imgRGBA[:, :, 1] = imgRGBA[:, :, 1].astype(np.uint8)

    img1[:, :, 2] = label[:, :, 0]
    min_bit2 = np.min(img1[:, :, 2])
    max_bit2 = np.max(img1[:, :, 2])

    if max_bit2 > 0:
        if min_bit2 == 0:
            ff = img1[:, :, 2].flatten()
            min_bit2 = sorted(set(list(ff)))[1]
            max_bit2 = sorted(set(list(ff)))[-1]
            img1[:, :, 2][img1[:, :, 2] == 0] = min_bit2
            imgRGBA[:, :, 2] = np.rint(255 * ((img1[:, :, 2] - min_bit2) / (max_bit2 - min_bit2)))
        else:
            imgRGBA[:, :, 2] = np.rint(255 * ((img1[:, :, 2] - min_bit2) / (max_bit2 - min_bit2)))
    imgRGBA[:, :, 2] = imgRGBA[:, :, 2].astype(np.uint8)
    imgRGBA = imgRGBA.astype(np.uint8)

    return imgRGBA

if __name__ == '__main__':
    p = r'D:\database\84_albers\data\7band\201008_clip_data.tif'
    img = gdal_read_img(p)
    nor = norm_band(img)
    io.imsave('201008.png', nor)