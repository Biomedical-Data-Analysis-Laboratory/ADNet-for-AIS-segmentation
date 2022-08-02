#!/usr/bin/env python

import argparse
import time
import random
import glob
import wandb
import PIL
import numpy as np

import tqdm
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torch.optim.lr_scheduler import MultiStepLR

from models.fewshot_anom import FewShotSeg
from dataloading.datasets import TrainDataset as TrainDataset
from utils import *


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--save_root', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--n_sv', type=int, required=True)  # flag number supervoxels
    parser.add_argument('--fold', type=int, required=True)

    # Training specs.
    parser.add_argument('--workers', default=4, type=int)
    parser.add_argument('--steps', default=50000, type=int)
    parser.add_argument('--n_shot', default=1, type=int)
    parser.add_argument('--n_query', default=1, type=int)
    parser.add_argument('--n_way', default=1, type=int)
    parser.add_argument('--batch-size', default=1, type=int)
    parser.add_argument('--max_iterations', default=1000, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--lr_gamma', default=0.95, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--weight-decay', default=0.0005, type=float)
    parser.add_argument('--seed', default=None, type=int)
    parser.add_argument('--bg_wt', default=0.1, type=float)
    parser.add_argument('--fg_wt', default=1.0, type=float)
    parser.add_argument('--t_loss_scaler', default=1.0, type=float)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--k', default=0.5, type=float)
    parser.add_argument('--min_size', default=200, type=int)
    parser.add_argument('--sweep', default=False, action="store_true")
    parser.add_argument('--original_ds', default=True, action="store_true")
    parser.add_argument('--use_labels_intrain', default=False, action="store_true")

    return parser.parse_args()


def main():
    args = parse_arguments()
    torch.cuda.empty_cache()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    # Deterministic setting for reproducibility.
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    # Set up logging.
    logger = set_logger(args.save_root, 'train.log')
    logger.info(args)

    # Set up the path to save.
    add = ""
    if args.sweep:
        add += ".sweep"
        n_save_folds = len(glob.glob(os.path.join(args.save_root, 'model'+add+"*")))
        n_save_folds = str(n_save_folds)
        if len(n_save_folds) == 1: n_save_folds = "0" + n_save_folds
        add += ("."+n_save_folds)
    args.save_model_path = os.path.join(args.save_root, 'model'+add+'.pth')

    init_wandb(args)

    # Init model.
    model = FewShotSeg(args.dataset, use_coco_init=True, k=args.k)
    model = model.cuda()
    # model = nn.DataParallel(model.cuda())

    # Init optimizer.
    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    milestones = [(ii + 1) * 5000 for ii in range(args.steps // 5000 - 1)]
    scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=args.lr_gamma)  # Decay LR based on milestones

    # Define loss function.
    my_weight = torch.cuda.FloatTensor([args.bg_wt, args.fg_wt]).cuda()
    criterion = nn.NLLLoss(ignore_index=255, weight=my_weight)

    # Enable cuDNN benchmark mode to select the fastest convolution algorithm.
    cudnn.enabled = True
    cudnn.benchmark = True

    # Define data set and loader.
    train_dataset = TrainDataset(args)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=args.workers,
                                               pin_memory=True,
                                               drop_last=True)
    logger.info('  Training on ' + str(len(train_dataset.image_dirs)) + ' images not in test fold: ' +
                str([elem[len(args.data_root):] for elem in train_dataset.image_dirs]))

    # Start training.
    sub_epochs = args.steps // args.max_iterations
    logger.info('  Start training ...')

    for epoch in range(sub_epochs):
        # Train.
        batch_time, data_time, losses, q_loss, align_loss, t_loss = train(train_loader, model, criterion, optimizer,
                                                                          scheduler, args, epoch)

        # Log
        logger.info('============== Epoch [{}] =============='.format(epoch))
        logger.info('  Batch time: {:6.3f}'.format(batch_time))
        logger.info('  Loading time: {:6.3f}'.format(data_time))
        logger.info('  Total Loss  : {:.5f}'.format(losses))
        logger.info('  Query Loss  : {:.5f}'.format(q_loss))
        logger.info('  Align Loss  : {:.5f}'.format(align_loss))
        logger.info('  Threshold Loss  : {:.5f}'.format(t_loss))

        wandb.log({"epoch": epoch,"loss": losses,"q_loss": q_loss,"align_loss": align_loss,"t_loss": t_loss})

    # Save trained model.
    logger.info('  Saving model ...')
    torch.save(model.state_dict(), args.save_model_path)


def train(train_loader, model, criterion, optimizer, scheduler, args, epoch):

    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4f')
    q_loss = AverageMeter('Query loss', ':.4f')
    a_loss = AverageMeter('Align loss', ':.4f')
    t_loss = AverageMeter('Threshold loss', ':.4f')

    # Unfreeze encoder layers
    if epoch == 10 and "CTP" in args.dataset:
        for param in model.parameters(): param.requires_grad = True
    # Train mode.
    model.train()

    end = time.time()
    i=0
    for sample in tqdm.tqdm(train_loader):
        # Extract episode data.
        if not sample: continue  # the dict is empty
        support_images = [[shot.float().cuda() for shot in way] for way in sample['support_images']]
        support_fg_mask = [[shot.float().cuda() for shot in way] for way in sample['support_fg_labels']]

        query_images = [query_image.float().cuda() for query_image in sample['query_images']]
        query_labels = torch.cat([query_label.long().cuda() for query_label in sample['query_labels']], dim=0)
        if "CTP" in args.dataset: query_labels = query_labels[:,0,...]

        sprvxl_toexp = sample['sprvxl_toexp'][0,...]

        # Log loading time.
        data_time.update(time.time() - end)

        # Compute outputs and losses.
        query_pred, align_loss, thresh_loss = model(support_images, support_fg_mask, query_images,
                                                    train=True, t_loss_scaler=args.t_loss_scaler)

        query_loss = criterion(torch.log(torch.clamp(query_pred, torch.finfo(torch.float32).eps,
                                                     1 - torch.finfo(torch.float32).eps)), query_labels)
        loss = query_loss + align_loss + thresh_loss

        # compute gradient and do SGD step
        for param in model.parameters(): param.grad = None

        loss.backward()
        optimizer.step()
        scheduler.step()

        # Log loss.
        losses.update(loss.item(), query_pred.size(0))
        q_loss.update(query_loss.item(), query_pred.size(0))
        a_loss.update(align_loss.item(), query_pred.size(0))
        t_loss.update(thresh_loss.item(), query_pred.size(0))

        # Log elapsed time.
        batch_time.update(time.time() - end)
        end = time.time()
        i+=1
        if i % 500==0:
            support_img = support_images[0][0].squeeze().cpu().detach().numpy()[0]
            support_img = PIL.Image.fromarray(normalize(support_img).astype('uint8'))
            if support_img.mode == "F": support_img = support_img.convert("L")

            qry_img = query_images[0][0].squeeze().cpu().detach().numpy()[0]
            qry_img = PIL.Image.fromarray(normalize(qry_img).astype('uint8'))
            if qry_img.mode == "F": qry_img = qry_img.convert("L")

            sprvxl_toexp = PIL.Image.fromarray(sprvxl_toexp.squeeze().cpu().detach().numpy().astype('uint8'))
            if sprvxl_toexp.mode == "F": sprvxl_toexp = sprvxl_toexp.convert("L")

            wandb.log({"support_img": wandb.Image(support_img),
                       "qry_img": wandb.Image(qry_img, masks={
                           "predictions": {
                               "mask_data": np.array(query_pred.squeeze()[1].cpu().detach().numpy()>0.5, dtype=np.uint8)
                           },
                           "ground_truth": {
                               "mask_data": np.array((query_labels / query_labels.max()).squeeze().cpu().detach().numpy(), dtype=np.uint8)
                           }
                       }),
                       "sprvxl_toexp": wandb.Image(sprvxl_toexp, masks={
                           "predictions": {
                               "mask_data": np.array(query_pred.squeeze()[1].cpu().detach().numpy() > 0.5, dtype=np.uint8)
                           },
                           "ground_truth": {
                               "mask_data": np.array((query_labels / query_labels.max()).squeeze().cpu().detach().numpy(), dtype=np.uint8)
                           }
                       })})
            if args.dataset=="CTP":
                feats = model.encoder.features["layers_pre"].cpu().detach().numpy()
                wandb.log({
                    "first_conv_layer_1": wandb.Image(feats[0, ...].transpose(1, 2, 0)),
                    "first_conv_layer_2": wandb.Image(feats[1, ...].transpose(1, 2, 0))
                })

    return batch_time.avg, data_time.avg, losses.avg, q_loss.avg, a_loss.avg, t_loss.avg


if __name__ == '__main__':
    main()
