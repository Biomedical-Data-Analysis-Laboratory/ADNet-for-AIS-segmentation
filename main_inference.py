#!/usr/bin/env python

import argparse
import os.path
import random
import numpy as np
import cv2

import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torch.utils.data import DataLoader
import torchvision.transforms as T

from models.fewshot_anom import FewShotSeg
from dataloading.datasets import TestDataset
from dataloading.dataset_specifics import *
from utils import *


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--save_root', type=str, required=True)
    parser.add_argument('--pretrained_root', type=str, required=True)
    parser.add_argument('--fold', type=int, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--n_shot', default=1, type=int)
    parser.add_argument('--all_slices', default=False, type=bool)
    parser.add_argument('--EP1', default=False, type=bool)
    parser.add_argument('--seed', default=None, type=int)
    parser.add_argument('--workers', default=0, type=int)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--k', default=0.5, type=float)

    return parser.parse_args()


def main():
    args = parse_arguments()
    torch.cuda.empty_cache()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    # Deterministic setting for reproducability.
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    # Set up logging.
    logger = set_logger(args.save_root, 'train.log')
    logger.info(args)

    # Setup the path to save.
    args.save = os.path.join(args.save_root)

    # Init model and load state_dict.
    model = FewShotSeg(args.dataset, use_coco_init=False,k=args.k)
    model = model.cuda()
    # model = nn.DataParallel(model.cuda())
    model.load_state_dict(torch.load(args.pretrained_root, map_location="cpu"), strict=False)

    # Data loader.
    test_dataset = TestDataset(args)
    query_loader = DataLoader(test_dataset,
                              batch_size=1,
                              shuffle=False,
                              num_workers=args.workers,
                              pin_memory=True,
                              drop_last=True)

    # Inference.
    logger.info('  Start inference ... Note: EP1 is ' + str(args.EP1))
    logger.info('  Support: ' + str(test_dataset.support_dir[len(args.data_root):]))
    logger.info('  Query: ' + str([elem[len(args.data_root):] for elem in test_dataset.image_dirs]))

    # Get unique labels (classes).
    labels = get_label_names(args.dataset)

    # Loop over classes.
    class_dice = {}
    class_iou = {}
    class_mcc = {}
    for label_val, label_name in labels.items():

        # Skip BG class.
        if label_name == 'BG': continue

        logger.info('  *------------------Class: {}--------------------*'.format(label_name))
        logger.info('  *--------------------------------------------------*')

        # Get support sample + mask for current class.
        support_sample = test_dataset.getSupport(label=label_val, all_slices=args.all_slices, N=args.n_shot)
        test_dataset.label = label_val

        # Infer.
        with torch.no_grad():
            scores = infer(model, query_loader, support_sample, args, logger, label_name, args.dataset)

        # Log class-wise results
        class_dice[label_name] = torch.tensor(scores.patient_dice).mean().item()
        class_iou[label_name] = torch.tensor(scores.patient_iou).mean().item()
        class_mcc[label_name] = torch.tensor(scores.patient_mcc).mean().item()

        logger.info('      Mean class IoU: {}'.format(class_iou[label_name]))
        logger.info('      Mean class Dice: {}'.format(class_dice[label_name]))
        logger.info('      Mean class MCC: {}'.format(class_mcc[label_name]))
        logger.info('  *--------------------------------------------------*')

        for k in scores.patient_dice_class.keys():
            dice = torch.tensor(scores.patient_dice_class[k]).mean().item()
            iou = torch.tensor(scores.patient_iou_class[k]).mean().item()
            mcc = torch.tensor(scores.patient_mcc_class[k]).mean().item()

            logger.info('      CLASS {}'.format(k))
            logger.info('      Mean IoU: {}'.format(dice))
            logger.info('      Mean Dice: {}'.format(iou))
            logger.info('      Mean MCC: {}'.format(mcc))
            logger.info('  *--------------------------------------------------*')

    # Log final results.
    logger.info('  *-----------------Final results--------------------*')
    logger.info('  *--------------------------------------------------*')
    logger.info('  Mean IoU: {}'.format(class_iou))
    logger.info('  Mean Dice: {}'.format(class_dice))
    logger.info('  *--------------------------------------------------*')


def infer(model, query_loader, support_sample, args, logger, label_name, dataset):

    # Test mode.
    model.eval()

    # Unpack support data.
    support_image = [support_sample['image'][[i]].float().cuda() for i in range(support_sample['image'].shape[0])]  # n_shot x 3 x H x W
    support_fg_mask = [support_sample['label'][[i]].float().cuda() for i in range(support_sample['image'].shape[0])]  # n_shot x H x W

    # Loop through query volumes.
    flag_ctp = False if "CTP" not in dataset and "DWI" not in dataset else True
    scores = Scores(flag_ctp)
    for i, sample in enumerate(query_loader):
        torch.cuda.empty_cache()
        prefix = 'image_' if "CTP" not in dataset and "DWI" not in dataset else 'study_'
        # Unpack query data.
        query_image = [sample['image'][i].float().cuda() for i in range(sample['image'].shape[0])]  # [C x 3 x H x W]
        query_label = sample['label'].long() if "CTP" not in dataset else sample['label'][:,:,0,...].long()  # C x H x W
        query_id = sample['id'][0].split(prefix)[1][:-len('.nii.gz')] if "CTP" not in dataset else sample['id'][0].split(prefix)[1]

        # Compute output.
        if args.EP1 is True:
            # Match support slice and query sub-chunck.
            query_pred = torch.zeros(query_label.shape[-3:])
            C_q = sample['image'].shape[1]
            idx_ = np.linspace(0, C_q, args.n_shot+1).astype('int')
            for sub_chunck in range(args.n_shot):
                support_image_s = [support_image[sub_chunck]]  # 1 x 3 x H x W
                support_fg_mask_s = [support_fg_mask[sub_chunck]]  # 1 x H x W
                query_image_s = query_image[0][idx_[sub_chunck]:idx_[sub_chunck+1]]  # C' x 3 x H x W
                query_pred_s, _, _ = model([support_image_s], [support_fg_mask_s], [query_image_s], train=False)  # C x 2 x H x W
                query_pred_s = query_pred_s.argmax(dim=1).cpu()  # C x H x W
                query_pred[idx_[sub_chunck]:idx_[sub_chunck+1]] = query_pred_s

        else:  # EP 2
            query_pred, _, _ = model([support_image], [support_fg_mask], query_image, train=False)  # C x 2 x H x W
            query_pred = query_pred.argmax(dim=1).cpu()  # C x H x W
            query_pred = query_pred.type(torch.uint8)
        # Record scores.
        scores.record(query_pred, query_label[0,...], query_id=query_id)

        # Log.
        logger.info('    Tested query volume: ' + sample['id'][0][len(args.data_root):]
                    + '. Dice score:  ' + str(round(scores.patient_dice[-1].item(),3))
                    + '. IoU score:  ' + str(round(scores.patient_iou[-1].item(), 3))
                    + '. MCC score:  ' + str(round(scores.patient_mcc[-1].item(), 3)))

        # Save predictions.
        file_name = 'image_' + query_id + '_' + label_name + '.pt'
        torch.save(query_pred, os.path.join(args.save, file_name))

        save_vol(args,query_pred,query_id,label_name)
        save_vol(args, query_label[0,...], query_id, label_name+"_label")

    return scores


def save_vol(args,query_pred,query_id,label_name):
    pat_dir = os.path.join(args.save,query_id)
    if not os.path.isdir(pat_dir): os.makedirs(pat_dir)
    if not os.path.isdir(os.path.join(pat_dir,label_name)): os.makedirs(os.path.join(pat_dir,label_name))
    for s in range(query_pred.shape[0]):
        s_id = str(s)
        if len(s_id) == 1: s_id = "0" + s_id
        img = (query_pred[s,...]*255.).numpy()
        cv2.imwrite(os.path.join(pat_dir,label_name, s_id+".png"), img)


if __name__ == '__main__':
    main()
