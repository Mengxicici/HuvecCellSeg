import argparse
import os
import tifffile
import parser
import pylab
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import yaml
import albumentations as A
from albumentations.core.composition import Compose
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from PIL import Image
import archs
import cv2 as cv
from dataset2 import Dataset
from metrics import iou_score
from utils import AverageMeter
import colorsys
from skimage.exposure import match_histograms


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default="20220112_groundtruth_NestedUNet_Mx_noNorm_DsT7_3",
                        help='model name')
    parser.add_argument('--num_classes', default=4, type=int,
                        help='number of classes')

    args = parser.parse_args()

    return args


def preseg(imageori):
    image=imageori.copy()

    # Applying -average 减去average图片
    # kernel = np.ones((30, 30), np.float32) / 900
    # average = cv.filter2D(image, -1, kernel)
    # image = image - average


    # Applying the CV Top-Hat operation
    # The top-hat filter is used to enhance bright objects of interest
    # in a dark background.
    # The black-hat operation is used to do the opposite,
    # enhance dark objects of interest in a bright background.

    filterSize = (30, 30)
    kernel = cv.getStructuringElement(cv.MORPH_RECT,
                                      filterSize)

    image = cv.morphologyEx(image,
                                 cv.MORPH_TOPHAT,
                                 kernel)

    #Applying the CV clahe operation
    cliplimit = np.mean(image) + 2 * np.std(image) # 400.0
    clahe1 = cv.createCLAHE(clipLimit=cliplimit, tileGridSize=(30, 30))
    image = clahe1.apply(np.array(image, dtype='uint16'))

    # Apllying the match histograms operation
    dirrr1 = r"C:\Users\Mengxi\Box\Data\20220112_groundtruth\GFP_original\0001.tif"
    reference = tifffile.imread(dirrr1)
    image1 = match_histograms(image, reference)

    max=np.max(image1)
    mean=np.mean(image1)
    std=np.std(image1)

    return image1,mean/max,std/max



def calculate_max(names):

    max_list = []
    mean_list = []
    std_list=[]
    for img_filename in tqdm(names):
        #img = tifffile.imread(ori_Path + '/' + img_filename)
        img = tifffile.imread(img_filename)
        #img = img/np.max(img)
        #m, s = cv2.meanStdDev(img)
        mean = np.mean(img)
        std = np.std(img)
        max = np.max(img)
        mean_list.append(mean)
        std_list.append(std)
        max_list.append(max)
        # print(np.max(img))
    max_in_max = np.max(max_list)
    meanmean = np.mean(mean_list)
    stdmean=np.mean(std)

    print(f"pics in total: {len(max_list)}")
    print("最大值是：")
    print(np.max(max_list))
    print("平均值是：")
    print(meanmean)
    print("std是：")
    print(stdmean)
    print(f"最大值出现在几号图中（position of the max value）:{np.argmax(max_list)}",)
    # print("平均值，平均值/最大值，平均值中位数，图片数：")
    # print(m, m / max,m_median,m_median/max, len(m_list))
    # print("std, std/max,std 中位数:")
    # print(s, s / max, s_median,s_median/max,len(s_list))
    return max_in_max, meanmean,stdmean

def one_image_detection(dir,max_value,model_state_dict):
    args = parse_args()

    with open('models/%s/config.yml' % args.name, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    print('-' * 20)
    for key in config.keys():
        print('%s: %s' % (key, str(config[key])))
    print('-' * 20)

    cudnn.benchmark = True

    # create model
    print("=> creating model %s" % config['arch'])
    model = archs.__dict__[config['arch']](config['num_classes'],
                                           config['input_channels'],
                                           config['deep_supervision'])

    model = model.cuda()
    model.load_state_dict(torch.load(model_state_dict %
                                     config['name']))
    model.eval()
    with torch.no_grad():

        imageori = tifffile.imread(dir)
        image=imageori.copy()

        image,mean,std = preseg(image)


        filterSize = (30, 30)
        kernel = cv.getStructuringElement(cv.MORPH_RECT,
                                          filterSize)
        image = cv.morphologyEx(image,
                                cv.MORPH_TOPHAT,
                                kernel)


        cliplimit = np.mean(image) + 2 * np.std(image)
        clahe = cv.createCLAHE(clipLimit=cliplimit, tileGridSize=(30, 30))  # 400 this should be one cell size
        image = clahe.apply(image)
        # mean = np.mean(image)
        # std = np.std(image)
        # max=np.max(image)
        # # #
        # image = image / max
        # image = (image - mean / max) / (std / max)
        h = np.array(image).shape[0]
        w = np.array(image).shape[1]
        # mean = 0.04053384470681673
        # std = 0.02312417137999376

        # max_value
        #7200 _CFP-427-4_6_000.tif

        Trans= A.Compose([A.ToFloat(max_value= max_value*1.0),
                               #A.RandomCrop(512,512),
                               A.Resize(2400,2400),
                               #A.Normalize(mean=0.14,std=0.18, max_pixel_value=1),
                               #A.Normalize(mean,std,max_pixel_value=1)
                               ])

        image = Trans(image=image)['image']
        #image[image > mean+std*3] = mean+std*3

        image_expand = np.expand_dims(image, 0)
        image_expand = np.expand_dims(image_expand, 0)

        input = torch.tensor(image_expand, dtype=torch.float32).cuda()
        # target = target.cuda()
        outimage=[]
        # compute output
        if config['deep_supervision']:
            output = model(input)[-1]
        else:
            output = model(input)
        output = output.squeeze(dim=0)
        # iou = iou_score(output, target)
        # avg_meter.update(iou, input.size(0))

        #output = torch.sigmoid(output).cpu().numpy()



        pr_soft_last = F.softmax(output.permute(1 ,2 ,0),dim = -1).cpu().numpy()

        pr_soft_last_arg = pr_soft_last.argmax(axis=-1)

        colors = [(0, 0, 0), (0, 128, 0), (128, 0, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128), (0, 128, 128),
                  (128, 128, 128)]



        seg_img = np.zeros((np.shape(pr_soft_last_arg)[0],
                                np.shape(pr_soft_last_arg)[1], 3))
    #---------------------
    #        需要注意的是：cv写图像的时候也是按照BRG，与RGB不一样，因此是需要调整顺序才能获得正确颜色
    #---------------------
        for c in range(4):
            seg_img[:, :, 0] += (
                        (pr_soft_last_arg[ :, :] == c) * (colors[c][0])).astype('uint8')
            seg_img[:, :, 1] += (
                        (pr_soft_last_arg[ :, :] == c) * (colors[c][1])).astype('uint8')
            seg_img[:, :, 2] += (
                        (pr_soft_last_arg[ :, :] == c) * (colors[c][2])).astype('uint8')
        seggg=seg_img

    plt.imshow(seggg.astype("uint8"))
    pylab.show()
        # cv2.imwrite(os.path.join(inputdir,name_your_result_folder, image_id + '.jpg'),
        #                 seggg.astype('uint8'))
        #
        #
        # segggr=pr_soft_last_arg
        # #segggr=cv2.resize(segggr, (1200, 1200), interpolation=cv2.INTER_NEAREST_EXACT)
        # cv2.imwrite(os.path.join(inputdir, name_your_result_folder, image_id + '.png'),
        #             segggr.astype('uint8'))

    torch.cuda.empty_cache()
    print("test has done")

def get_img_binary(pred, thresh):
    ret, thresh_pred_1 = cv.threshold(pred, thresh, 1, cv.THRESH_BINARY)
    return thresh_pred_1


def main(model_state_dict, inputdir, name_your_result_folder, image_type_name_last_part, mean_max):



    args = parse_args()

    with open('models/%s/config.yml' % args.name, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    print('-'*20)
    for key in config.keys():
        print('%s: %s' % (key, str(config[key])))
    print('-'*20)

    cudnn.benchmark = True

    # create model
    print("=> creating model %s" % config['arch'])
    model = archs.__dict__[config['arch']](config['num_classes'],
                                           config['input_channels'],
                                           config['deep_supervision'])

    model = model.cuda()
    model.load_state_dict(torch.load(model_state_dict %
                                     config['name']))
    model.eval()


    # Data loading code
    img_ids1 = glob(os.path.join(inputdir, '*' + image_type_name_last_part))
    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids1]

    os.makedirs(os.path.join(inputdir,name_your_result_folder), exist_ok=True)
    # for c in range(config['num_classes']):
    #     os.makedirs(os.path.join('outputs', config['name'], str(c)), exist_ok=True)
    with torch.no_grad():
        for image_id in tqdm(img_ids):
            image_path = os.path.join(inputdir, image_id + ".tif")
            imageori = tifffile.imread(image_path)
            image = imageori.copy()
            h=np.array(image).shape[0]
            w=np.array(image).shape[1]

            # # 去除背景噪音
            # ori_imgmask = np.array(tifffile.imread(image_path))
            # ori_imgmaskmax=np.max(ori_imgmask)
            # threshmask = get_img_binary(ori_imgmask, 2000)#ori_imgmaskmax/30   #越高留下的内容就越少 #440,530
            # ori_ones=np.ones((h,w,3))
            # ori_ones[:, :, 0] = threshmask
            # ori_ones[:, :, 1] = threshmask
            # ori_ones[:, :, 2] = threshmask
            # ori_3ch_imgmask=ori_ones



            image, mean, std = preseg(image)
            # image,mean,std = preseg(image)
            # filterSize = (30, 30)
            # kernel = cv.getStructuringElement(cv.MORPH_RECT,
            #                                   filterSize)
            # image = cv.morphologyEx(image,
            #                         cv.MORPH_TOPHAT,
            #                         kernel)

            # cliplimit = np.mean(image) + 2 * np.std(image)
            # clahe = cv.createCLAHE(clipLimit=400.0, tileGridSize=(20, 20)) # this should be one cell size
            # image = clahe.apply(image)
            # mean = np.mean(image)
            # std = np.std(image)
            # max = np.max(image)
            # # #
            # image = image / max
            # image = (image - mean / max) / (std / max)
            # mean = 0.04053384470681673
            # std = 0.02312417137999376

            # max_value
            #7200 _CFP-427-4_6_000.tif
            if mean_max != 0:
                Trans = A.Compose([A.ToFloat(max_value=mean_max * 1.0),
                                   #A.RandomCrop(512,512),
                                   A.Resize(1600, 1600)
                                   #A.Resize(768,768)
                                   # A.Normalize(mean, std, max_pixel_value=1),
                                   #A.Normalize(mean,std,max_pixel_value=1)
                                   ])
            else:
                Trans= A.Compose([A.ToFloat(max_value=65535.0),
                                   #A.RandomCrop(512,512),
                                   A.Resize(1600,1600)
                                   #A.Resize(768,768)
                                   # A.Normalize(mean, std, max_pixel_value=1),
                                   #A.Normalize(mean,std,max_pixel_value=1)
                                   ])

            image = Trans(image=image)['image']
            #image[image > mean+std*3] = mean+std*3

            image_expand = np.expand_dims(image, 0)
            image_expand = np.expand_dims(image_expand, 0)


            input = torch.tensor(image_expand, dtype=torch.float32).cuda()







            # target = target.cuda()
            outimage=[]
            # compute output
            if config['deep_supervision']:
                output = model(input)[-1]
            else:
                output = model(input)
            output = output.squeeze(dim=0)
            # iou = iou_score(output, target)
            # avg_meter.update(iou, input.size(0))

            #output = torch.sigmoid(output).cpu().numpy()

            #
            pr_soft_last = F.softmax(output.permute(1 ,2 ,0),dim = -1).cpu().numpy()

            pr_soft_last_arg = pr_soft_last.argmax(axis=-1)

            colors = [(0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128), (0, 128, 128),
                      (128, 128, 128)]



            seg_img = np.zeros((np.shape(pr_soft_last_arg)[0],
                                    np.shape(pr_soft_last_arg)[1], 3))
        #---------------------
        #        需要注意的是：cv写图像的时候也是按照BRG，与RGB不一样，因此是需要调整顺序才能获得正确颜色
        #---------------------
            for c in range(4):
                seg_img[:, :, 0] += (
                            (pr_soft_last_arg[ :, :] == c) * (colors[c][2])).astype('uint8')
                seg_img[:, :, 1] += (
                            (pr_soft_last_arg[ :, :] == c) * (colors[c][0])).astype('uint8')
                seg_img[:, :, 2] += (
                            (pr_soft_last_arg[ :, :] == c) * (colors[c][1])).astype('uint8')
            seggg=seg_img
            seggg = cv2.resize(seggg, (h, w), interpolation=cv2.INTER_NEAREST_EXACT) #* ori_3ch_imgmask
            cv2.imwrite(os.path.join(inputdir,name_your_result_folder, image_id + '.jpg'),
                            seggg.astype('uint8'))


            segggr=pr_soft_last_arg
            segggr=cv2.resize(segggr, (h, w), interpolation=cv2.INTER_NEAREST_EXACT) #* threshmask
            cv2.imwrite(os.path.join(inputdir, name_your_result_folder, image_id + '.png'),
                        segggr.astype('uint8'))

    torch.cuda.empty_cache()
    print("work has done")



    
    torch.cuda.empty_cache()



if __name__ == '__main__':
    """
    Do not use it on non-16bit image, never tested
    """

    model_state_dict = r'models/%s/model-0.7070950910303718-2022_04_01_18_35_00_orig.pth'
    #model_state_dict = r'models/%s/model-0.6906760292479016-2022_03_29_19_36_22.pth'

    #model_state_dict = r'models/%s/model-0.6933685189112619-2022_03_29_08_41_59.pth'
    #model_state_dict = r'models/%s/model-0.7453783502968084-2022_02_10_16_45_49.pth'
    #model_state_dict = r'models/%s/model-0.7277277664667737-2022_02_28_16_15_24.pth'
    #model_state_dict = r'models/%s/model-0.7152322042514863-2022_03_06_19_07_37.pth'


    # dir = input("input the dir if this is single image detection(press return if in batch):")
    #
    # if bool(dir):
    #     print("now calculate max value:")
    #     single_max = np.max(tifffile.imread(str(dir)))
    #     print(f"max value is:{single_max},85% max is {single_max*0.85},120% max is {single_max*1.2}")
    #     max_value = float(input("input max_value (input 65535.0 as default):"))
    #     one_image_detection(dir, max_value, model_state_dict)
    #
    # else:
    #
    #     """
    #     Note for every one:
    #     all the things that you need to change are here
    #     any change outside # ==== is not expected
    #     in most situation default setting is good enough
    #     """


    # ==============================================================================================================

    inputdir1 = r'C:\Users\Mengxi\Box\Data\SingleCell\20020804_cancer_8well_LCIS_finger_AZD_DMOD_Dmso_Stresstest_0.1D\test1_0.1D_Dmso_stresstest_with_F=380_1\Pos0'
    inputdir2 = r'C:\Users\Mengxi\Box\Data\SingleCell\20020804_cancer_8well_LCIS_finger_AZD_DMOD_Dmso_Stresstest_0.1D\test1_0.1D_Dmso_stresstest_with_F=380_1\Pos1'
    inputdir3 = r'C:\Users\Mengxi\Box\Data\SingleCell\20020804_cancer_8well_LCIS_finger_AZD_DMOD_Dmso_Stresstest_0.1D\test1_0.1D_Dmso_stresstest_with_F=380_1\Pos2'
    inputdir4 = r'C:\Users\Mengxi\Box\Data\SingleCell\20020804_cancer_8well_LCIS_finger_AZD_DMOD_Dmso_Stresstest_0.1D\test2_0.1D_stress_test_Dmog=380_1\Pos0'
    inputdir5 = r'C:\Users\Mengxi\Box\Data\SingleCell\20020804_cancer_8well_LCIS_finger_AZD_DMOD_Dmso_Stresstest_0.1D\test2_0.1D_stress_test_Dmog=380_1\Pos1'
    inputdir6 = r'C:\Users\Mengxi\Box\Data\SingleCell\20020804_cancer_8well_LCIS_finger_AZD_DMOD_Dmso_Stresstest_0.1D\test2_0.1D_stress_test_Dmog=380_1\Pos2'

    inputdir7 = r'C:\Users\Mengxi\Box\Data\SingleCell\20020804_cancer_8well_LCIS_finger_AZD_DMOD_Dmso_Stresstest_0.1D\test3_0.1D_stress_test_AZD=380_1\Pos0'
    inputdir8 = r'C:\Users\Mengxi\Box\Data\SingleCell\20020804_cancer_8well_LCIS_finger_AZD_DMOD_Dmso_Stresstest_0.1D\test3_0.1D_stress_test_AZD=380_1\Pos1'

    inputdir9 = r'C:\Users\Mengxi\Box\Data\SingleCell\20020804_cancer_8well_LCIS_finger_AZD_DMOD_Dmso_Stresstest_0.1D\test3_0.1D_stress_test_AZD=380_1\Pos2'
    inputdir10 = r'C:\Users\Mengxi\Box\Data\SingleCell\20020804_cancer_8well_LCIS_finger_AZD_DMOD_Dmso_Stresstest_0.1D\test4_0.1D_stress_test_Blank=380_1\Pos0'
    inputdir11 = r'C:\Users\Mengxi\Box\Data\SingleCell\20020804_cancer_8well_LCIS_finger_AZD_DMOD_Dmso_Stresstest_0.1D\test4_0.1D_stress_test_Blank=380_1\Pos1'
    inputdir12 = r'C:\Users\Mengxi\Box\Data\SingleCell\20020804_cancer_8well_LCIS_finger_AZD_DMOD_Dmso_Stresstest_0.1D\test4_0.1D_stress_test_Blank=380_1\Pos2'



#image_type_name_last_part = '_DAPI-1 GFP-2 TRITC-3_000.tif'
    image_type_name_last_part = '_438-542-4_6_000.tif'
    #image_type_name_last_part = '_GFP-Cube5_000.tif'
    #image_type_name_last_part = 'TRITC-Cube5_000.tif'
    #image_type_name_last_part = '_c000002.tif'


    name_your_result_folder = 'Unet++_of_434_Tophat_clahe_Norm_withmodel-0.707_orig0413_pth'

    # ===============================================================================================================

    # origin_names = glob(os.path.join(inputdir1, '*' + image_type_name_last_part))
    #
    # print(f"loaded the image from: {inputdir1}, type: {image_type_name_last_part}, please find the result in:{name_your_result_folder}")

    # comput = np.array(input("Do you need max value calculation? please input 1 or 0:"))
    #
    # if comput == 1:
    #
    #     print("now,comput the figure values:")
    #     maxvalue, maxvalue_list = calculate_max(origin_names)
    #     mean_max = np.mean(maxvalue_list)
    #     print(f"remember this mean_max_value:{mean_max}")
    #     print(f"50% is:{0.50 * mean_max}, 85% is {0.85 * mean_max}")
    #     mean_max = np.float64(input("Plase input adaptive mean_max_value:"))
    #
    # else:
    #
    #     print("please use Fret results.")
    #     mean_max = np.float64(input("Plase input adaptive mean_max_value(input 0 if using default setting):"))

    mean_max=0


    main(model_state_dict, inputdir1, name_your_result_folder, image_type_name_last_part, mean_max)
    main(model_state_dict, inputdir2, name_your_result_folder, image_type_name_last_part, mean_max)
    main(model_state_dict, inputdir3, name_your_result_folder, image_type_name_last_part, mean_max)
    main(model_state_dict, inputdir4, name_your_result_folder, image_type_name_last_part, mean_max)
    main(model_state_dict, inputdir5, name_your_result_folder, image_type_name_last_part, mean_max)
    main(model_state_dict, inputdir6, name_your_result_folder, image_type_name_last_part, mean_max)
    main(model_state_dict, inputdir7, name_your_result_folder, image_type_name_last_part, mean_max)
    main(model_state_dict, inputdir8, name_your_result_folder, image_type_name_last_part, mean_max)
    main(model_state_dict, inputdir9, name_your_result_folder, image_type_name_last_part, mean_max)
    main(model_state_dict, inputdir10, name_your_result_folder, image_type_name_last_part, mean_max)
    main(model_state_dict, inputdir11, name_your_result_folder, image_type_name_last_part, mean_max)
    main(model_state_dict, inputdir12, name_your_result_folder, image_type_name_last_part, mean_max)
