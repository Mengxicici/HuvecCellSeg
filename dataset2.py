import os
import torch.backends.cudnn as cudnn
import torch.nn as nn
import cv2 as cv
import numpy as np
import tifffile
import torch
import torch.utils.data
from PIL import Image
import seaborn as sns
from skimage.exposure import match_histograms

class Dataset(torch.utils.data.Dataset):
    def __init__(self, img_ids, img_dir, mask_dir, img_ext, mask_ext, num_classes, transform=None):

        self.img_ids = img_ids
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_ext = img_ext
        self.mask_ext = mask_ext
        self.num_classes = num_classes
        self.transform = transform

    def __len__(self):
        return len(self.img_ids)

    def preseg(self, image):
        # dirrr1 = r"C:\Users\Mengxi\Box\Data\20220112_groundtruth\GFP_original\0001.tif"
        # reference = tifffile.imread(dirrr1)
        # image1 = match_histograms(image, reference)
        filterSize = (30, 30)
        kernel = cv.getStructuringElement(cv.MORPH_RECT,
                                          filterSize)

        # # Reading the image named 'input.jpg'
        # input_image = cv.imread(dirrr3)

        # input_image = cv.cvtColor(input_image, cv.COLOR_BGR2GRAY)

        # Applying the Top-Hat operation
        tophat_img = cv.morphologyEx(image,
                                     cv.MORPH_TOPHAT,
                                     kernel)



        cliplimit = np.mean(tophat_img) + 2 * np.std(tophat_img)
        clahe1 = cv.createCLAHE(clipLimit=cliplimit, tileGridSize=(20, 20))
        imagefin = clahe1.apply(np.array(tophat_img, dtype='uint16'))
        return imagefin

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]

        img = tifffile.imread(os.path.join(self.img_dir, img_id + self.img_ext))

        dirrr1 = r"C:\Users\Mengxi\Box\Data\20220112_groundtruth\GFP_original\0001.tif"
        reference = tifffile.imread(dirrr1)
        img = match_histograms(img, reference)
        # tophat
        #img = self.preseg(img)
        # filterSize = (30, 30)
        # kernel = cv.getStructuringElement(cv.MORPH_RECT,
        #                                   filterSize)
        # img = cv.morphologyEx(image,
        #                         cv.MORPH_TOPHAT,
        #                         kernel)
        # # 直方图均衡化+标准化
        # #cliplimit = np.mean(image) + 2 * np.std(image)
        # clahe = cv.createCLAHE(clipLimit=400.0, tileGridSize=(20, 20))
        # img = clahe.apply(image)
        # mean = np.mean(img)
        # std=np.std(img)
        # max=np.max(img)
        #
        # img=img/max
        # img=(img-mean/max)/(std/max)
        h, w = img.shape[0], img.shape[1]
        img = np.expand_dims(img, 2)
        mask = []

        # ---------------------------------
        """
        不同文件的读取方式不同需要注意修改
        """
        # ------------------------------------

        # mask.append(tifffile.imread(os.path.join(self.mask_dir,
        #                                     img_id + self.mask_ext)), -1)
        mask.append(Image.open(os.path.join(self.mask_dir,
                                                img_id + self.mask_ext)))
        mask = np.dstack(mask)
        mask = np.eye(self.num_classes)[np.array(mask, dtype='int').reshape([-1])]
        mask = mask.reshape((int(h), int(w), self.num_classes))

        if self.transform is not None:
            augmented = self.transform(image=img, mask=mask)  # 这个包比较方便，能把mask也一并做掉
            img = augmented['image']  # 参考https://github.com/albumentations-team/albumentations
            mask = augmented['mask']

        img = img.astype('float32')
        # img_blur = cv2.bilateralFilter(img[:,:,0], -1, 5, 100)
        # img_fin=img[:,:,0]-img_blur
        #img = np.expand_dims(img, 2)
        imgtranspose = img.transpose(2, 0, 1)

        mask = mask.astype('float32')
        mask = mask.transpose(2, 0, 1)

        return imgtranspose, mask, {'img_id': img_id}


if __name__=="__main__":
    images_directory = r'C:\Users\Mengxi\Box\Data\20210730_HCAEC_groundtruth\cyto'
    masks_directory = r'C:\Users\Mengxi\Box\Data\20210730_HCAEC_groundtruth\groundtruth'
    root_directory = r'C:\Users\Mengxi\Box\Data\20210730_HCAEC_groundtruth'
    img_ids = open(os.path.join(root_directory, "segmentationTest/train.txt"), 'r').read().splitlines()

    img_dir = r'C:\Users\Mengxi\Box\Data\20210730_HCAEC_groundtruth\cyto'
    mask_dir = r'C:\Users\Mengxi\Box\Data\20210730_HCAEC_groundtruth\groundtruth'
    img_ext = ".tiff"
    mask_ext = ".tif"
    num_classes = 4

    dataset=Dataset(img_ids, img_dir, mask_dir, img_ext, mask_ext, num_classes, transform=None)
    print(np.array(dataset[0][0]).shape)
    print(np.array(dataset[0][1]).shape)
    print(np.unique(np.array(dataset)[0][1]))
    print(np.array(dataset[0][0][:10,:10]))
    print(np.array(dataset[0][1][0][:10, :10]))



