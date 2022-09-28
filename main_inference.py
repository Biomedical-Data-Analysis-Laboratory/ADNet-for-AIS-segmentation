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
import matplotlib.pyplot as plt

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
    parser.add_argument('--original_ds', default=True, action="store_true")
    parser.add_argument('--CGM', default=False, type=bool)  # Cross-Guided Multiple Shot Learning

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
    logger.info('  Support: ' + str([elem[len(args.data_root):] for elem in test_dataset.support_dir]))  # str(test_dataset.support_dir[len(args.data_root):]))
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
        class_dice[label_name] = round(torch.tensor(scores.patient_dice).mean().item(),4)
        class_iou[label_name] = round(torch.tensor(scores.patient_iou).mean().item(),4)
        class_mcc[label_name] = round(torch.tensor(scores.patient_mcc).mean().item(),4)

        logger.info('  *-----------------Final results--------------------*')
        logger.info('  *--------------------------------------------------*')
        logger.info('      Mean class IoU: {}'.format(class_iou[label_name]))
        logger.info('      Mean class Dice: {}'.format(class_dice[label_name]))
        logger.info('      Mean class MCC: {}'.format(class_mcc[label_name]))
        logger.info('  *--------------------------------------------------*')

        for k in scores.patient_dice_class.keys():
            dice = round(torch.tensor(scores.patient_dice_class[k]).mean().item(),4)
            iou = round(torch.tensor(scores.patient_iou_class[k]).mean().item(),4)
            mcc = round(torch.tensor(scores.patient_mcc_class[k]).mean().item(),4)

            logger.info('      CLASS {}'.format(k))
            logger.info('      Mean IoU: {}'.format(dice))
            logger.info('      Mean Dice: {}'.format(iou))
            logger.info('      Mean MCC: {}'.format(mcc))
            logger.info('  *--------------------------------------------------*')

    # Log final results.
    logger.info('  *---------------Aggregate results------------------*')
    logger.info('  *--------------------------------------------------*')
    logger.info('  Aggregate IoU: {}'.format(scores.compute_iou()))
    logger.info('  Aggregate Dice: {}'.format(scores.compute_dice()))
    logger.info('  *--------------------------------------------------*')
    logger.info('  *----------------------LVO----------------------*')
    logger.info('  Aggregate IoU: {}'.format(scores.compute_iou("LVO")))
    logger.info('  Aggregate Dice: {}'.format(scores.compute_dice("LVO")))
    logger.info('  *--------------------------------------------------*')
    logger.info('  *----------------------SVO----------------------*')
    logger.info('  Aggregate IoU: {}'.format(scores.compute_iou("SVO")))
    logger.info('  Aggregate Dice : {}'.format(scores.compute_dice("SVO")))
    logger.info('  *--------------------------------------------------*')

    for flag in ["LVO","SVO","ALL"]:
        plt.scatter(scores.get_coords(flag, 0), scores.get_coords(flag, 1))
        plt.xlabel("N. pixels")
        plt.ylabel("Dice")
        plt.title(flag)
        plt.ylim(0, 1)
        plt.savefig(os.path.join(args.save_root,"scatter_{}.png".format(flag)))
        plt.cla()


def infer(model, query_loader, support_samples, args, logger, label_name, dataset):
    # Test mode.
    model.eval()

    # Unpack support data.
    support_images, support_fg_masks = [], []
    for support_sample in support_samples:
        support_image = [support_sample['image'][[i]].float().cuda() for i in range(support_sample['image'].shape[0])]  # n_shot x 3 x H x W
        support_fg_mask = [support_sample['label'][[i]].float().cuda() for i in range(support_sample['image'].shape[0])]  # n_shot x H x W
        support_images.append(support_image)
        support_fg_masks.append(support_fg_mask)

    K = len(support_images)
    # Loop through query volumes.
    flag_ctp = False if "CTP" not in dataset and "DWI" not in dataset else True
    scores = Scores(flag_ctp)
    for i, sample in enumerate(query_loader):
        torch.cuda.empty_cache()
        prefix = 'image_' if "CTP" not in dataset and "DWI" not in dataset else 'study_'
        # Unpack query data.
        query_image = [sample['image'][i].float().cuda() for i in range(sample['image'].shape[0])]  # [C x 3 x H x W]
        query_label = sample['label'].long() if "CTP" not in dataset else sample['label'][:,:,0,...].long()  # C x H x W
        query_label = query_label if "DWI" not in dataset else query_label[0,...]
        query_id = sample['id'][0].split(prefix)[1][:-len('.nii.gz')] if "CTP" not in dataset else sample['id'][0].split(prefix)[1]

        # Compute output.
        if args.EP1 is True:  # TODO: add multiple support images
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
            query_pred = 0
            if i == 0: confident_score = [0] * K
            if args.CGM and K > 1:  # Cross-Guided Multiple Shot Learning
                P_hat_q = [0] * K
                for k, (support_image, support_fg_mask) in enumerate(zip(support_images, support_fg_masks)):

                    slice_idx = [int(len(support_image)/2)-1,int(len(support_image)/2),int(len(support_image)/2)+1]  # [int(len(support_image)/2)]
                    # slice_idx = [int(len(support_image)/2)-2,int(len(support_image)/2)-1,int(len(support_image)/2),int(len(support_image)/2)+1,int(len(support_image)/2)+2] # [int(len(support_image)/2)]
                    # slice_idx = range(len(support_image))

                    if i == 0:
                        confident_score[k] = [confident_score[k]] * len(slice_idx)
                        for I_s_k, I_s_k_mask in zip(support_images, support_fg_masks):
                            n_queries = len(I_s_k)
                            img_size = I_s_k[0].shape[-2:]
                            # the query image is the current support image
                            qry_img = torch.stack(I_s_k, dim=0).view(n_queries, -1, img_size[0], img_size[1])
                            # stack the masks accordingly for later IOU calculation
                            masks_stack = torch.stack([torch.stack([mask[0][0]], dim=0) for mask in I_s_k_mask],
                                                      dim=0).view(n_queries, img_size[0], img_size[1])
                            for idx, s in enumerate(slice_idx):
                                # cur_qry_img = [qry_img] if "CTP" in dataset else [qry_img[[s]]]
                                G_i_s, _, _ = model([[support_image[s]]], [[support_fg_mask[s]]], [qry_img], train=False)
                                m_hat = G_i_s.argmax(dim=1).cpu()
                                m_hat = m_hat.type(torch.uint8)
                                confident_score[k][idx] += IOU(m_hat, masks_stack.cpu())
                        for idx in range(len(confident_score[k])): confident_score[k][idx] /= len(support_images)
                        # confident_score[k] /= float(len(slice_idx))
                    for idx, s in enumerate(slice_idx):
                        # cur_query_image = query_image if "CTP" in dataset else [query_image[0][[s]]]
                        query_pred_tmp, _, _ = model([[support_image[s]]], [[support_fg_mask[s]]], query_image, train=False)
                        # query_pred_tmp = query_pred_tmp.argmax(dim=1).cpu()
                        # query_pred_tmp = query_pred_tmp.type(torch.uint8)
                        P_hat_q[k] += (confident_score[k][idx] * query_pred_tmp.cpu())
                    P_hat_q[k] /= float(len(slice_idx))
                P_hat_q_fin = 0
                for P_hat_q_tmp in P_hat_q: P_hat_q_fin += P_hat_q_tmp
                P_hat_q_fin /= float(K)
                query_pred = []

                soft = nn.Softmax(dim=0)
                for s in range(P_hat_q_fin.shape[0]): query_pred.append(soft(P_hat_q_fin[s, ...]))
                query_pred = torch.stack(query_pred, dim=0)
                query_pred = query_pred.argmax(dim=1).cpu()
                query_pred = query_pred.type(torch.uint8)
            else:
                for support_image, support_fg_mask in zip(support_images, support_fg_masks):
                    query_pred_tmp, _, _ = model([[support_image[int(len(support_image)/2)]]], [[support_fg_mask[int(len(support_image)/2)]]], query_image, train=False)  # C x 2 x H x W
                    query_pred_tmp = query_pred_tmp.argmax(dim=1).cpu()  # C x H x W
                    query_pred_tmp = query_pred_tmp.type(torch.uint8)
                    query_pred += query_pred_tmp
                query_pred = query_pred / float(len(support_images))
                query_pred[query_pred >= 0.5] = 1
                query_pred[query_pred < 0.5] = 0
        # Record scores.
        scores.record(query_pred, query_label[0,...], query_id=query_id)

        # Log.
        logger.info('    Tested query volume: ' + sample['id'][0][len(args.data_root):]
                    + '. Dice score:  ' + str(round(scores.patient_dice[-1].item(),3))
                    + '. IoU score:  ' + str(round(scores.patient_iou[-1].item(), 3))
                    + '. MCC score:  ' + str(round(scores.patient_mcc[-1].item(), 3))
                    + '. ROI pixels:  ' + str(round(scores.patient_roi[-1].item(), 3)))

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
