import argparse
import os
import parser
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
from dataset2 import Dataset
from metrics import iou_score
from utils import AverageMeter
import colorsys

"""
需要指定参数：--name dsb2018_96_NestedUNet_woDS
"""


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default="20220112_groundtruth_NestedUNet_Mx",
                        help='model name')
    parser.add_argument('--num_classes', default=4, type=int,
                        help='number of classes')

    args = parser.parse_args()

    return args


# def get_miou_png(self, image):
#     # ---------------------------------------------------------#
#     #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
#     #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
#     # ---------------------------------------------------------#
#
#     orininal_h = np.array(image).shape[0]
#     orininal_w = np.array(image).shape[1]
#     # ---------------------------------------------------------#
#     #   给图像增加灰条，实现不失真的resize
#     #   也可以直接resize进行识别
#     # ---------------------------------------------------------#
#     image_data, nw, nh = resize_image(image, (self.input_shape[1], self.input_shape[0]))
#     # ---------------------------------------------------------#
#     #   添加上batch_size维度
#     # ---------------------------------------------------------#
#     image_data = np.expand_dims(np.transpose(np.array(image_data, np.float32), (2, 0, 1)), 0)
#
#     with torch.no_grad():
#         images = torch.from_numpy(image_data)
#         if self.cuda:
#             images = images.cuda()
#
#         # ---------------------------------------------------#
#         #   图片传入网络进行预测
#         # ---------------------------------------------------#
#         pr = self.mod   (images)[0]
#         # ---------------------------------------------------#
#         #   取出每一个像素点的种类
#         # ---------------------------------------------------#
#         pr = F.softmax(pr.permute(1, 2, 0), dim=-1).cpu().numpy()
#         # --------------------------------------#
#         #   将灰条部分截取掉
#         # --------------------------------------#
#         pr = pr[int((self.input_shape[0] - nh) // 2): int((self.input_shape[0] - nh) // 2 + nh), \
#              int((self.input_shape[1] - nw) // 2): int((self.input_shape[1] - nw) // 2 + nw)]
#         # ---------------------------------------------------#
#         #   进行图片的resize
#         # ---------------------------------------------------#
#         pr = cv2.resize(pr, (orininal_w, orininal_h), interpolation=cv2.INTER_LINEAR)
#         # ---------------------------------------------------#
#         #   取出每一个像素点的种类
#         # ---------------------------------------------------#
#         pr = pr.argmax(axis=-1)
#
#     image = Image.fromarray(np.uint8(pr))
#     return image


def main():
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
    model.load_state_dict(torch.load('models/%s/model-0.7530250338064476-2022_03_24_09_38_24.pth' %
                                     config['name']))
    model.eval()

    inputdir = r'C:\Users\Mengxi\Box\Data'
    # Data loading code
    img_ids = glob(os.path.join(inputdir, config['dataset'], 'GFP_original', '*' + config['img_ext']))
    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]

    _, val_img_ids = train_test_split(img_ids, test_size=0.15, random_state=45)

    # model.load_state_dict(torch.load('models/%s/model.pth' %
    #                                  config['name']))
    # model.eval()

    val_transform = A.Compose([
        A.ToFloat(max_value=1.0),
        #A.Resize(config['input_h'], config['input_w']),
        #A.Normalize(mean=(0.03684), std=(0.01488), max_pixel_value=1),

        #        A.FromFloat(max_value=13108.0)
    ])

    val_dataset = Dataset(
        img_ids=val_img_ids,
        img_dir=os.path.join(inputdir, config['dataset'], 'GFP_original'),
        mask_dir=os.path.join(inputdir, config['dataset'], 'groundtruth_unet++'),
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        num_classes=config['num_classes'],
        transform=val_transform)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False)

    avg_meter = AverageMeter()

    os.makedirs(os.path.join('outputs', config['name'], 'png_results'), exist_ok=True)
    # for c in range(config['num_classes']):
    #     os.makedirs(os.path.join('outputs', config['name'], str(c)), exist_ok=True)
    with torch.no_grad():

        for input, target, meta in tqdm(val_loader, total=len(val_loader)):
            input = input.cuda()
            target = target.cuda()
            outimage = []
            # compute output
            if config['deep_supervision']:
                output = model(input)[-1]
            else:
                output = model(input)

            iou = iou_score(output, target)
            avg_meter.update(iou, input.size(0))

            # output = torch.sigmoid(output).cpu().numpy()
            #
            pr_soft_last = F.softmax(output.permute(0, 2, 3, 1), dim=-1).cpu().numpy()

            pr_soft_last_arg = pr_soft_last.argmax(axis=-1)

            colors = [(0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128), (0, 128, 128),
                      (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0), (64, 0, 128),
                      (192, 0, 128),
                      (64, 128, 128), (192, 128, 128), (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0),
                      (0, 64, 128),
                      (128, 64, 12)]

            for i in range(np.shape(pr_soft_last_arg)[0]):
                seg_img = np.zeros((np.shape(pr_soft_last_arg)[0],
                                    np.shape(pr_soft_last_arg)[1],
                                    np.shape(pr_soft_last_arg)[2], 3))
                # ---------------------
                #        需要注意的是：cv写图像的时候也是按照BRG，与RGB不一样，因此是需要调整顺序才能获得正确颜色
                # ---------------------
                for c in range(4):
                    seg_img[i, :, :, 0] += (
                            (pr_soft_last_arg[i, :, :] == c) * (colors[c][2])).astype('uint8')
                    seg_img[i, :, :, 1] += (
                            (pr_soft_last_arg[i, :, :] == c) * (colors[c][0])).astype('uint8')
                    seg_img[i, :, :, 2] += (
                            (pr_soft_last_arg[i, :, :] == c) * (colors[c][1])).astype('uint8')
                seggg = seg_img[i]
                #seggg = cv2.resize(seggg, (1200, 1200), interpolation=cv2.INTER_LINEAR)
                cv2.imwrite(os.path.join('outputs', config['name'], meta['img_id'][i] + '.jpg'),
                            seggg.astype('uint8'))

                segggr = pr_soft_last_arg[i]
                #segggr = cv2.resize(segggr, (1200, 1200), interpolation=cv2.INTER_NEAREST_EXACT)
                cv2.imwrite(os.path.join('outputs', config['name'],'png_results', meta['img_id'][i] + '.png'),
                            segggr.astype('uint8'))

            # ------------------------------------------------#
            #   将新图片转换成Image的形式
            # ------------------------------------------------#

        # for i in range(len(outimage)):
        #      cv2.imwrite(os.path.join('outputs', config['name'], meta['img_id'][i] + '.jpg'),
        #                  (np.array(i * 255).astype('uint8')))

    print('IoU: %.4f' % avg_meter.avg)

    #    plot_examples(input, target, model,num_examples=3)

    torch.cuda.empty_cache()


#
# def plot_examples(datax, datay, model,num_examples=6):
#     fig, ax = plt.subplots(nrows=num_examples, ncols=3, figsize=(18,4*num_examples))
#     m = datax.shape[0]
#     for row_num in range(num_examples):
#         image_indx = np.random.randint(m)
#         image_arr = model(datax[image_indx:image_indx+1]).squeeze(0).detach().cpu().numpy()
#         ax[row_num][0].imshow(np.transpose(datax[image_indx].cpu().numpy(), (1,2,0))[:,:,0])
#         ax[row_num][0].set_title("Orignal Image")
#         ax[row_num][1].imshow(np.squeeze((image_arr > 0.40)[0,:,:].astype(int)))
#         ax[row_num][1].set_title("Segmented Image localization")
#         ax[row_num][2].imshow(np.transpose(datay[image_indx].cpu().numpy(), (1,2,0))[:,:,0])
#         ax[row_num][2].set_title("Target image")
#     plt.show()


if __name__ == '__main__':
    main()
