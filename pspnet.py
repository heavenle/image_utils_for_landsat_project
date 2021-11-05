import sys

import os
import random
import cv2
import numpy as np
from skimage import io
import glob
import shutil


class DataAugmentation:
    """
    基于语义分割数据的数据增强方法。
    方法有：
    1. random_flip
    2. random_rotation
    3. add_noise
    4. random_scale
    5. random_crop
    6. random_shift
    7. guess_blur
    8. motion_blur
    9. del_file[删除文件]
    10. function[自定义方法]
    """
    def __init__(self,width,heigth,prob=0.5):
        self.width = width
        self.heigth = heigth
        self.prob = prob

    def random_flip(self,image,label):
        if random.random() > self.prob:
            flip_method = random.randint(-1,1)
            # 1 水平翻转
            # 0 垂直翻转
            # -1 水平垂直翻转
            image = cv2.flip(image,flip_method)
            label = cv2.flip(label,flip_method)
        return image, label

    def random_rotation(self,image,label):
        if random.random() > self.prob:
            center = (image.shape[0] // 2, image.shape[1] // 2)
            angle = random.randint(0, 360)
            M = cv2.getRotationMatrix2D(center, angle, 1)
            image = cv2.warpAffine(src=image, M=M, dsize=(self.width,self.heigth))
            label = cv2.warpAffine(src=label, M=M, dsize=(self.width,self.heigth))
        return image, label

    def add_noise(self,image,label):
        if random.random() > self.prob:
            image = np.array(image / 255, dtype=float)
            std = 0.001 * random.randint(0, 15)
            noise = np.random.normal(0, std ** 0.5, image.shape)
            image = image + noise
            if image.min() < 0:
                low_clip = -1.
            else:
                low_clip = 0.
            image = np.clip(image, low_clip, 1.0)
            image = np.uint8(image * 255)
        return image,label

    def random_scale(self,image,label,min_scale,max_scale):
        if random.random() > self.prob:
            w = image.shape[1]
            h = image.shape[0]
            scale_x = np.random.uniform(min_scale,max_scale)
            scale_y = np.random.uniform(min_scale,max_scale)
            scale_matrix = np.array([[scale_x,0,(1.-scale_x)*w//2],[0,scale_y,(1.-scale_y)*h//2]],dtype=float)
            image = cv2.warpAffine(image,scale_matrix,(self.width,self.heigth),flags=cv2.INTER_NEAREST)
            label = cv2.warpAffine(label, scale_matrix, (self.width, self.heigth), flags=cv2.INTER_NEAREST)
        return image,label

    def random_crop(self,image,label,min_ratio,max_ratio):
        if random.random() > self.prob:
            w, h = image.shape[:2]
            ratio = random.random()
            scale = min_ratio + ratio * (max_ratio - min_ratio)
            new_h = int(h * scale)
            new_w = int(w * scale)
            x = np.random.randint(0, w - new_w)
            y = np.random.randint(0, h - new_h)
            image = image[x:x + new_w, y:y + new_h, :]
            #image = cv2.resize(image,(self.width,self.heigth))
            label = label[x:x + new_w, y:y + new_h]
            #label = cv2.resize(label,(self.width,self.heigth))
        return image,label

    def random_shift(self,image, label):
        if random.random() > self.prob:
            w, h, _ = image.shape
            shift_x = random.randint(0, w // 4)
            shift_y = random.randint(0, h // 4)
            if random.randint(0, 1) == 0:
                shift_x = shift_x
                shift_y = shift_y
            elif random.randint(0, 1) == 1:
                shift_x = -shift_x
                shift_y = -shift_y
            M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
            image = cv2.warpAffine(image, M, (w, h))
            label = cv2.warpAffine(label, M, (w, h))
        return image, label

    def guess_blur(self,image,label):
        if random.random() > self.prob:
            min_size = 5
            addition = random.choice((0, 2, 4, 6, 8, 10, 12))
            size = min_size + addition
            kernel_size = (size, size)
            image = cv2.GaussianBlur(image, kernel_size, 0)
        return image,label

    def motion_blur(self,image,label):
        if random.random() > self.prob:
            angle = random.randint(0,360)
            degree = random.randint(1,12)
            M = cv2.getRotationMatrix2D((degree / 2, degree / 2), angle, 1)
            motion_blur_kernel = np.diag(np.ones(degree))
            motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M, (degree, degree))
            motion_blur_kernel = motion_blur_kernel / degree
            blurred = cv2.filter2D(image, -1, motion_blur_kernel)
            cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)
            image = np.array(blurred, dtype=np.uint8)
        return image,label

    # def random_bright(self,image,label):
    #     if random.random() > self.prob:


    def del_file(self,path):
        if os.path.exists(path):
            for file in os.listdir(path):
                os.remove(os.path.join(path,file))
        else:
            shutil.rmtree(path)

    def function(self,func,image,label):
        return func(image,label)

if __name__ == '__main__':
    target_width = 320
    target_heigth = 320
    base_num = 2068
    path1 = r'D:\project\crop\GF1_data_320_25\small_data\glacier'
    path = r'D:\project\crop\GF1_data_320_25\augmentation'
    data = DataAugmentation(target_width, target_heigth, prob= -1)
    if not os.path.exists(path+'/images/') & os.path.exists(path+'/masks/'):
        os.mkdir(path)
        os.makedirs(path+'/images/')
        os.makedirs(path+'/masks/')
    # else:
    #     data.del_file(path+'/images/')
    #     data.del_file(path+'/masks/')

    img_src = [img for img in glob.glob(path1+'/images/'+'*.tif')]
    img_label = [label for label in glob.glob(path1+'/masks/'+'*.tif')]

    index_list = [i for i in range(0, len(img_src))]

    random_index = np.random.choice(index_list, base_num, replace=True)
    count = 0
    for i in random_index:
    # for i in range(len(img_src)):
        image = io.imread(img_src[i])
        label = io.imread(img_label[i])
        image_name = os.path.splitext(os.path.basename(img_src[i]))

        count += 1
        image_flip,label_flip = data.random_flip(image, label)
        io.imsave(path+'/images/{}_flip.tif'.format(image_name[0] + '_' + str(count)), image_flip)
        io.imsave(path + '/masks/{}_flip.tif'.format(image_name[0] + '_' + str(count)), label_flip)

        count += 1
        image_rotation,label_rotation = data.random_rotation(image, label)
        io.imsave(path + '/images/{}_rotation.tif'.format(image_name[0] + '_' + str(count)), image_rotation)
        io.imsave(path + '/masks/{}_rotation.tif'.format(image_name[0] + '_' + str(count)), label_rotation)

        # count += 1
        # image_noise,label_noise = data.add_noise(image,label)
        # io.imsave(path + '/images/{}_noise.tif'.format(image_name[0] + '_' + str(count)), image_noise)
        # io.imsave(path + '/masks/{}_noise.tif'.format(image_name[0] + '_' + str(count)), label_noise)

        count += 1
        image_scale,label_scale = data.random_scale(image, label,0.7,1.2)
        io.imsave(path + '/images/{}_scale.tif'.format(image_name[0] + '_' + str(count)), image_scale)
        io.imsave(path + '/masks/{}_scale.tif'.format(image_name[0] + '_' + str(count)), label_scale)

        count += 1
        image_crop,label_crop = data.random_crop(image, label,0.7,1.)
        io.imsave(path + '/images/{}_crop.tif'.format(image_name[0] + '_' + str(count)), image_crop)
        io.imsave(path + '/masks/{}_crop.tif'.format(image_name[0] + '_' + str(count)), label_crop)

        count += 1
        image_shift,label_shift = data.random_shift(image, label)
        io.imsave(path + '/images/{}_shift.tif'.format(image_name[0] + '_' + str(count)), image_shift)
        io.imsave(path + '/masks/{}_shift.tif'.format(image_name[0] + '_' + str(count)), label_shift)

        # count += 1
        # image_motion_blur, label_motion_blur = data.motion_blur(image, label)
        # io.imsave(path + '/images/{}_motion_blur.tif'.format(image_name[0] + '_' + str(count)), image_motion_blur)
        # io.imsave(path + '/masks/{}_motion_blur.tif'.format(image_name[0] + '_' + str(count)), label_motion_blur)

        # count += 1
        # image_guess_bluer, label_guess_bluer = data.guess_blur(image, label)
        # io.imsave(path + '/images/{}_guess_bluer.tiff'.format(image_name[0] + '_' + str(count)), image_guess_bluer)
        # io.imsave(path + '/masks/{}_guess_bluer.tiff'.format(image_name[0] + '_' + str(count)), label_guess_bluer)



















