import argparse
import os
from collections import OrderedDict
from glob import glob
import torch.nn.functional as F
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import yaml
import albumentations as A
from sklearn.model_selection import train_test_split
from torch.optim import lr_scheduler
from tqdm import tqdm
# from tversky import TverskyLoss
import numpy as np
import matplotlib.pyplot as plt
import archs
import losses
from dataset2 import Dataset
from metrics import iou_score
from utils import AverageMeter, str2bool
import datetime

time = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

ARCH_NAMES = archs.__all__
LOSS_NAMES = losses.__all__
LOSS_NAMES.append('TverskyLoss')

"""

指定参数：
--dataset  
--arch NestedUNet

"""


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None,
                        help='model name: (default: arch+timestamp)')
    parser.add_argument('--epochs', default=150, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch_size', default=4, type=int,
                        metavar='N', help='mini-batch size (default: 16)')

    # model
    parser.add_argument('--arch', '-a', metavar='ARCH', default='NestedUNet',
                        choices=ARCH_NAMES,
                        help='model architecture: ' +
                             ' | '.join(ARCH_NAMES) +
                             ' (default: NestedUNet)')
    parser.add_argument('--deep_supervision', default=True, type=str2bool)
    parser.add_argument('--input_channels', default=1, type=int,
                        help='input channels')
    parser.add_argument('--num_classes', default=4, type=int,
                        help='number of classes')
    parser.add_argument('--input_w', default=512, type=int,
                        help='image width')
    parser.add_argument('--input_h', default=512, type=int,
                        help='image height')

    # loss   TverskyLoss BCEDiceLoss(BCEWithLogitsLoss)
    parser.add_argument('--loss', default='TverskyLoss',
                        choices=LOSS_NAMES,
                        help='loss: ' +
                             ' | '.join(LOSS_NAMES) +
                             ' (default: BCEDiceLoss)')

    # dataset
    parser.add_argument('--dataset', default='20210730_HCAEC_groundtruth',
                        help='dataset name')
    parser.add_argument('--img_ext', default='.tiff',
                        help='image file extension')
    parser.add_argument('--mask_ext', default='.tif',
                        help='mask file extension')

    # optimizer
    parser.add_argument('--optimizer', default='Adam',
                        choices=['Adam', 'SGD'],
                        help='loss: ' +
                             ' | '.join(['Adam', 'SGD']) +
                             ' (default: Adam)')
    parser.add_argument('--lr', '--learning_rate', default=1e-3, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--weight_decay', default=1e-6, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', default=False, type=str2bool,
                        help='nesterov')

    # scheduler
    parser.add_argument('--scheduler', default='CosineAnnealingLR',
                        choices=['CosineAnnealingLR', 'ReduceLROnPlateau', 'MultiStepLR', 'ConstantLR'])
    parser.add_argument('--min_lr', default=1e-6, type=float,
                        help='minimum learning rate')
    parser.add_argument('--factor', default=0.1, type=float)
    parser.add_argument('--patience', default=2, type=int)
    parser.add_argument('--milestones', default='1,2', type=str)
    parser.add_argument('--gamma', default=2 / 3, type=float)
    parser.add_argument('--early_stopping', default=-1, type=int,
                        metavar='N', help='early stopping (default: -1)')

    parser.add_argument('--num_workers', default=4, type=int)

    config = parser.parse_args()

    return config


def train(config, train_loader, model, criterion, optimizer):
    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter()}

    model.train()

    pbar = tqdm(total=len(train_loader))
    for input, target, _ in train_loader:
        input = input.cuda()
        target = target.cuda()

        # compute output
        if config['deep_supervision']:
            outputs = model(input)
            loss = 0
            for output in outputs:
                loss += criterion(output, target)
            loss /= len(outputs)
            iou = iou_score(outputs[-1], target)
        else:
            output = model(input)
            loss = criterion(output, target)
            iou = iou_score(output, target)

        # compute gradient and do optimizing step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_meters['loss'].update(loss.item(), input.size(0))
        avg_meters['iou'].update(iou, input.size(0))

        postfix = OrderedDict([
            ('loss', avg_meters['loss'].avg),
            ('iou', avg_meters['iou'].avg),
        ])
        pbar.set_postfix(postfix)
        pbar.update(1)
    pbar.close()

    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg)])


def validate(config, val_loader, model, criterion):
    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter()}

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        pbar = tqdm(total=len(val_loader))
        for input, target, _ in val_loader:
            input = input.cuda()
            target = target.cuda()

            # compute output
            if config['deep_supervision']:
                outputs = model(input)
                loss = 0
                for output in outputs:

                    loss += criterion(output, target)
                loss /= len(outputs)
                iou = iou_score(outputs[-1], target)
            else:
                output = model(input)

                loss = criterion(output, target)
                iou = iou_score(output, target)

            avg_meters['loss'].update(loss.item(), input.size(0))
            avg_meters['iou'].update(iou, input.size(0))

            postfix = OrderedDict([
                ('loss', avg_meters['loss'].avg),
                ('iou', avg_meters['iou'].avg),
            ])
            pbar.set_postfix(postfix)
            pbar.update(1)
        pbar.close()

    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg)])


def main():
    config = vars(parse_args())

    if config['name'] is None:
        if config['deep_supervision']:
            config['name'] = '%s_%s_Mx_434+histoNorm625_DsT7_3' % (config['dataset'], config['arch'])
        else:
            config['name'] = '%s_%s_Mx_NoNormT' % (config['dataset'], config['arch'])
    os.makedirs('models/%s' % config['name'], exist_ok=True)

    print('-' * 20)
    for key in config:
        print('%s: %s' % (key, config[key]))
    print('-' * 20)

    with open('models/%s/config.yml' % config['name'], 'w') as f:
        yaml.dump(config, f)




    # define loss function (criterion)
    #weight2 = [1.06537722, 0.94218266, 0.5665378, 2.42376917] #434
    weight1=[1.8770822,  0.81234476, 0.39363897, 1.30039762] #625

    if config['loss'] == 'TverskyLoss':
        criterion = losses.__dict__['TverskyLoss'](weight=torch.Tensor(weight1)).cuda()
    else:
        criterion = losses.__dict__[config['loss']](weight=torch.Tensor(weight1)).cuda()


    cudnn.benchmark = True

    # create model
    print("=> creating model %s" % config['arch'])
    model = archs.__dict__[config['arch']](config['num_classes'],
                                           config['input_channels'],
                                           config['deep_supervision'])

    model = model.cuda()
    # ============
    pth_path= r"C:\Users\Mengxi\Box\Mengxi\notebook\NEO-conv\python\DL_for_Med\8-unet_for_cell_segmentation\unet++\unet++\models\20220112_groundtruth_NestedUNet_Mx_noNorm_DsT7_3\model-0.7183015143121985-2022_04_05_09_25_53_with625training_badBG.pth"
    model.load_state_dict(torch.load(pth_path))

    params = filter(lambda p: p.requires_grad, model.parameters())
    if config['optimizer'] == 'Adam':
        optimizer = optim.Adam(
            params, lr=config['lr'], weight_decay=config['weight_decay'])
    elif config['optimizer'] == 'SGD':
        optimizer = optim.SGD(params, lr=config['lr'], momentum=config['momentum'],
                              nesterov=config['nesterov'], weight_decay=config['weight_decay'])
    else:
        raise NotImplementedError

    if config['scheduler'] == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config['epochs'], eta_min=config['min_lr'])
    elif config['scheduler'] == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=config['factor'], patience=config['patience'],
                                                   verbose=1, min_lr=config['min_lr'])
    elif config['scheduler'] == 'MultiStepLR':
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[int(e) for e in config['milestones'].split(',')],
                                             gamma=config['gamma'])
    elif config['scheduler'] == 'ConstantLR':
        scheduler = None
    else:
        raise NotImplementedError

    # Data loading code
    inputdir = r'C:\Users\Mengxi\Box\Data'

    # img_ids = glob(os.path.join(inputdir, config['dataset'], 'GFP_original', '*' + config['img_ext']))
    img_ids_all = glob(
        os.path.join(r'C:\Users\Mengxi\Box\Data\20210730_HCAEC_groundtruth\cyto', '*' + config['img_ext']))
    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids_all]

    train_img_ids, val_img_ids = train_test_split(img_ids, test_size=0.15, random_state=45)
    # 数据增强：

    train_transform = A.Compose(
        [
            A.ToFloat(max_value=65535.0),
            A.RandomCrop(config['input_h'], config['input_w']),
            # A.RandomRotate90(),
            # A.Flip(),
            A.OneOf([
                # A.HueSaturationValue(),
                A.RandomBrightnessContrast(),

            ], p=0.5),  # 按照归一化的概率选择执行哪一个

            # A.HorizontalFlip(p=0.5),
            # A.Normalize(mean=0.485,std=0.229,max_pixel_value=1.0),
            # A.Normalize(mean=0.037, std=0.015),
            # # A.Transpose(),
            # A.OneOf([
            #     A.GaussNoise(),
            #     # A.GaussNoise(),
            # ], p=0.3),
            # # A.OneOf([
            # #     A.MotionBlur(p=0.2),
            # #     A.MedianBlur(blur_limit=3, p=0.1),
            # #     A.Blur(blur_limit=3, p=0.1),
            # # ], p=0.5),
            # A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
            # A.OneOf([
            #     A.OpticalDistortion(p=0.3),
            #     A.GridDistortion(p=0.1),
            #     A.PiecewiseAffine(p=0.3),
            # ], p=0.5),
            # A.OneOf([
            #     #A.CLAHE(clip_limit=2),
            #     A.Sharpen(),
            #     A.Emboss(),
            #     A.RandomBrightnessContrast(),
            # ], p=0.5),
            #A.HueSaturationValue(p=0.3),
        ])







    # train_transform = A.Compose([
    #     A.ToFloat(max_value=1.0),
    #            A.RandomRotate90(),
    #            A.Flip(),
    #             A.OneOf([
    #                 #A.HueSaturationValue(),
    #                 A.RandomBrightness(),
    #                 A.RandomContrast(),
    #             ], p=1),#按照归一化的概率选择执行哪一个
    #     A.RandomCrop(config['input_h'], config['input_w']),
    #
    #     # A.Normalize(mean=(0.03684), std=(0.01488), max_pixel_value=1),
    #     #        A.Normalize(mean=(0.465,),std=(0.229,)),
    # ])

    """
    img = (img - mean * max_pixel_value) / (std * max_pixel_value)
    每一批数据，需要自己计算这个数值
    """

    #        A.FromFloat(max_value=13108.0)

    val_transform = A.Compose([
        A.ToFloat(max_value=65535.0),
        #A.Normalize(mean=0.491, std=0.272, max_pixel_value=1.0),
        A.RandomCrop(config['input_h'], config['input_w']),
        # A.Crop(config['input_h'], config['input_w']),
        # A.Normalize(mean=0.037, std=0.015),
        # A.Normalize(mean=0.485,std=0.229,max_pixel_value=1.0),
        # A.Normalize(),
        #        A.FromFloat(max_value=13108.0)
    ])

    train_dataset = Dataset(
        img_ids=train_img_ids,
        img_dir=os.path.join(inputdir, config['dataset'], 'cyto'),
        mask_dir=os.path.join(inputdir, config['dataset'], 'groundtruth'),
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        num_classes=config['num_classes'],
        transform=train_transform)
    val_dataset = Dataset(
        img_ids=val_img_ids,
        img_dir=os.path.join(inputdir, config['dataset'], 'cyto'),
        mask_dir=os.path.join(inputdir, config['dataset'], 'groundtruth'),
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        num_classes=config['num_classes'],
        transform=val_transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        drop_last=True)  # 不能整除的batch是否就不要了
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False)

    log = OrderedDict([
        ('epoch', []),
        ('lr', []),
        ('loss', []),
        ('iou', []),
        ('val_loss', []),
        ('val_iou', []),
    ])

    best_iou = 0
    trigger = 0
    for epoch in range(config['epochs']):
        print('Epoch [%d/%d]' % (epoch, config['epochs']))

        # train for one epoch
        train_log = train(config, train_loader, model, criterion, optimizer)
        # evaluate on validation set
        val_log = validate(config, val_loader, model, criterion)

        if config['scheduler'] == 'CosineAnnealingLR':
            scheduler.step()
        elif config['scheduler'] == 'ReduceLROnPlateau':
            scheduler.step(val_log['loss'])

        print('loss %.4f - iou %.4f - val_loss %.4f - val_iou %.4f'
              % (train_log['loss'], train_log['iou'], val_log['loss'], val_log['iou']))

        log['epoch'].append(epoch)
        log['lr'].append(config['lr'])
        log['loss'].append(train_log['loss'])
        log['iou'].append(train_log['iou'])
        log['val_loss'].append(val_log['loss'])
        log['val_iou'].append(val_log['iou'])

        pd.DataFrame(log).to_csv('models/%s/log.csv' %
                                 config['name'], index=False)

        trigger += 1

        if val_log['iou'] > best_iou:
            best_iou = val_log['iou']
            torch.save(model.state_dict(), f'models/%s/model-{best_iou}-{time}.pth' %
                       config['name'])
            print("=> saved best model")
            trigger = 0

        # early stopping
        if config['early_stopping'] >= 0 and trigger >= config['early_stopping']:
            print("=> early stopping")
            break

        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
