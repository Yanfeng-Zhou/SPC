from torchvision import transforms, datasets
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader
import argparse
import time
import os
import numpy as np
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.backends import cudnn
import random
import torchio as tio
from skimage import segmentation
from scipy.ndimage import distance_transform_edt

from config.dataset_config.dataset_cfg import dataset_cfg
from config.train_test_config.train_test_config import print_train_loss_SPC, print_val_loss_SPC, print_train_eval_SPC, print_val_eval, save_val_best_3d, print_best
from config.visdom_config.visual_visdom import visdom_initialization_SPC, visualization_SPC
from config.warmup_config.warmup import GradualWarmupScheduler
from config.augmentation.online_aug import data_transform_3d
from loss.loss_function import segmentation_loss
from models.getnetwork import get_network
from dataload.dataset_3d import dataset_it
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

def init_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_trained_models', default='/mnt/data1/SPC/checkpoints')
    parser.add_argument('--path_seg_results', default='/mnt/data1/SPC/seg_pred')
    parser.add_argument('--path_dataset', default='/mnt/data1/BDMC/dataset/P-CT')
    parser.add_argument('--dataset_name', default='P-CT', help='P-CT, LA')
    parser.add_argument('--sup_mark', default='20')
    parser.add_argument('--unsup_mark', default='80')
    parser.add_argument('-b', '--batch_size', default=1, type=int)
    parser.add_argument('-e', '--num_epochs', default=200, type=int)
    parser.add_argument('-s', '--step_size', default=50, type=int)
    parser.add_argument('-l', '--lr', default=0.5, type=float)
    parser.add_argument('-g', '--gamma', default=0.5, type=float)
    parser.add_argument('-u', '--unsup_weight', default=3, type=float)
    parser.add_argument('--loss', default='CE', type=str)
    parser.add_argument('-w', '--warm_up_duration', default=20)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--wd', default=-5, type=float, help='weight decay pow')

    parser.add_argument('-i', '--display_iter', default=5, type=int)
    parser.add_argument('-2D', '--network_2D', default='unet')
    parser.add_argument('-3D', '--network_3D', default='unet3d', type=str)
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--rank_index', default=0, help='0, 1, 2, 3')
    parser.add_argument('-v', '--vis', default=True, help='need visualization or not')
    parser.add_argument('--visdom_port', default=16672, help='16672')
    args = parser.parse_args()

    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend='nccl', init_method='env://')

    rank = torch.distributed.get_rank()
    ngpus_per_node = torch.cuda.device_count()
    init_seeds(rank + 1)

    dataset_name = args.dataset_name
    cfg = dataset_cfg(dataset_name)

    print_num = 77 + (cfg['NUM_CLASSES'] - 3) * 14
    print_num_minus = print_num - 2
    print_num_half = int(print_num / 2 - 1)

    path_trained_models = args.path_trained_models+'/'+str(os.path.split(args.path_dataset)[1])
    if not os.path.exists(path_trained_models) and rank == args.rank_index:
        os.mkdir(path_trained_models)
    path_trained_models = path_trained_models+'/'+'SPC'+'-l='+str(args.lr)+'-e='+str(args.num_epochs)+'-s=' + str(args.step_size)+'-g='+str(args.gamma)+'-b='+str(args.batch_size)+'-uw='+str(args.unsup_weight)+'-w='+str(args.warm_up_duration)+'-'+str(args.sup_mark)+'-'+str(args.unsup_mark)
    if not os.path.exists(path_trained_models) and rank == args.rank_index:
        os.mkdir(path_trained_models)

    if args.vis and rank == args.rank_index:
        visdom_env = str('SPC-'+str(os.path.split(args.path_dataset)[1])+'-l='+str(args.lr)+'-e='+str(args.num_epochs)+'-s='+ str(args.step_size)+'-g='+str(args.gamma)+'-b='+str(args.batch_size)+'-uw='+str(args.unsup_weight)+'-w='+str(args.warm_up_duration)+'-'+str(args.sup_mark)+'-'+str(args.unsup_mark))
        visdom = visdom_initialization_SPC(env=visdom_env, port=args.visdom_port)


    # Dataset
    data_transform = data_transform_3d(cfg['NORMALIZE'])

    dataset_train_unsup = dataset_it(
        data_dir=args.path_dataset + '/train_unsup_' + args.unsup_mark,
        transform_1=data_transform['train'],
        queue_length=cfg['QUEUE_LENGTH'],
        samples_per_volume=cfg['NUM_SAMPLE_TRAIN'],
        patch_size=cfg['PATCH_SIZE'],
        num_workers=8,
        shuffle_subjects=True,
        shuffle_patches=True,
        sup=False,
        num_images=None
    )
    num_images_unsup = len(dataset_train_unsup.dataset_1)

    dataset_train_sup = dataset_it(
        data_dir=args.path_dataset + '/train_sup_' + args.sup_mark,
        transform_1=data_transform['train'],
        queue_length=cfg['QUEUE_LENGTH'],
        samples_per_volume=cfg['NUM_SAMPLE_TRAIN'],
        patch_size=cfg['PATCH_SIZE'],
        num_workers=8,
        shuffle_subjects=True,
        shuffle_patches=True,
        sup=True,
        num_images=num_images_unsup
    )
    dataset_val = dataset_it(
        data_dir=args.path_dataset + '/val',
        transform_1=data_transform['val'],
        queue_length=cfg['QUEUE_LENGTH'],
        samples_per_volume=cfg['NUM_SAMPLE_VAL'],
        patch_size=cfg['PATCH_SIZE'],
        num_workers=8,
        shuffle_subjects=False,
        shuffle_patches=False,
        sup=True,
        num_images=None
    )

    train_sampler_unsup = torch.utils.data.distributed.DistributedSampler(dataset_train_unsup.queue_train_set_1, shuffle=True)
    train_sampler_sup = torch.utils.data.distributed.DistributedSampler(dataset_train_sup.queue_train_set_1, shuffle=True)
    val_sampler = torch.utils.data.distributed.DistributedSampler(dataset_val.queue_train_set_1, shuffle=False)

    dataloaders = dict()
    dataloaders['train_sup'] = DataLoader(dataset_train_sup.queue_train_set_1, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=0, sampler=train_sampler_sup)
    dataloaders['train_unsup'] = DataLoader(dataset_train_unsup.queue_train_set_1, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=0, sampler=train_sampler_unsup)
    dataloaders['val'] = DataLoader(dataset_val.queue_train_set_1, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=0, sampler=val_sampler)

    num_batches = {'train_sup': len(dataloaders['train_sup']), 'train_unsup': len(dataloaders['train_unsup']), 'val': len(dataloaders['val'])}

    # Model
    model1 = get_network(args.network_2D, cfg['IN_CHANNELS'], cfg['NUM_CLASSES'])
    model2 = get_network(args.network_3D, cfg['IN_CHANNELS'], cfg['NUM_CLASSES'])

    model1 = model1.cuda()
    model2 = model2.cuda()
    model1 = DistributedDataParallel(model1, device_ids=[args.local_rank])
    model2 = DistributedDataParallel(model2, device_ids=[args.local_rank])
    dist.barrier()

    # Training Strategy
    criterion = segmentation_loss(args.loss, False).cuda()

    optimizer1 = optim.SGD(model1.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=5*10 ** args.wd)
    exp_lr_scheduler1 = lr_scheduler.StepLR(optimizer1, step_size=args.step_size, gamma=args.gamma)
    scheduler_warmup1 = GradualWarmupScheduler(optimizer1, multiplier=1.0, total_epoch=args.warm_up_duration, after_scheduler=exp_lr_scheduler1)

    optimizer2 = optim.SGD(model2.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=5*10 ** args.wd)
    exp_lr_scheduler2 = lr_scheduler.StepLR(optimizer2, step_size=args.step_size, gamma=args.gamma)
    scheduler_warmup2 = GradualWarmupScheduler(optimizer2, multiplier=1.0, total_epoch=args.warm_up_duration, after_scheduler=exp_lr_scheduler2)

    # Train & Val
    since = time.time()
    count_iter = 0

    best_model = model1
    best_result = 'Result1'
    best_val_eval_list = [0 for i in range(4)]

    for epoch in range(args.num_epochs):

        count_iter += 1
        if (count_iter - 1) % args.display_iter == 0:
            begin_time = time.time()

        dataloaders['train_sup'].sampler.set_epoch(epoch)
        dataloaders['train_unsup'].sampler.set_epoch(epoch)
        model1.train()
        model2.train()

        train_loss_sup_3D = 0.0
        train_loss_sup_2D = 0.0
        train_loss_unsup = 0.0
        train_loss = 0.0

        val_loss_sup_3D = 0.0
        val_loss_sup_2D = 0.0

        unsup_weight = args.unsup_weight * (epoch + 1) / args.num_epochs

        dist.barrier()

        dataset_train_sup = iter(dataloaders['train_sup'])
        dataset_train_unsup = iter(dataloaders['train_unsup'])

        for i in range(num_batches['train_sup']):

            unsup_index = next(dataset_train_unsup)
            img_train_unsup_3D = Variable(unsup_index['image'][tio.DATA].cuda())
            B, C, H, W, D = img_train_unsup_3D.shape
            img_train_unsup_2D = img_train_unsup_3D.permute(0, 4, 1, 2, 3).reshape(B*D, C, H, W)
            idx = torch.randperm(img_train_unsup_2D.shape[0])
            img_train_unsup_2D = img_train_unsup_2D[idx, :]
            torch.cuda.empty_cache()

            optimizer1.zero_grad()
            optimizer2.zero_grad()

            pred_train_unsup_2D = model1(img_train_unsup_2D)
            pred_train_unsup_2D_re = torch.zeros_like(pred_train_unsup_2D)
            for m, n in enumerate(idx):
                pred_train_unsup_2D_re[n, :] = pred_train_unsup_2D[m, :]
            pred_train_unsup_2D_re = pred_train_unsup_2D_re.reshape(B, D, cfg['NUM_CLASSES'], H, W).permute(0, 2, 3, 4, 1)
            torch.cuda.empty_cache()
            pred_train_unsup_3D = model2(img_train_unsup_3D)

            # pred -> mask
            max_train_unsup_2D_re = torch.max(pred_train_unsup_2D_re, dim=1)[1].long()
            max_train_unsup_3D = torch.max(pred_train_unsup_3D, dim=1)[1].long()

            loss_train_unsup_3D_2D = criterion(pred_train_unsup_3D, max_train_unsup_2D_re)
            loss_train_unsup_2D_3D = criterion(pred_train_unsup_2D_re, max_train_unsup_3D)
            loss_train_unsup = loss_train_unsup_3D_2D + loss_train_unsup_2D_3D
            loss_train_unsup = loss_train_unsup * unsup_weight
            loss_train_unsup.backward(retain_graph=True)

            sup_index = next(dataset_train_sup)
            img_train_sup_3D = Variable(sup_index['image'][tio.DATA].cuda())
            B, C, H, W, D = img_train_sup_3D.shape
            img_train_sup_2D = img_train_sup_3D.permute(0, 4, 1, 2, 3).reshape(B*D, C, H, W)
            idx = torch.randperm(img_train_sup_2D.shape[0])
            img_train_sup_2D = img_train_sup_2D[idx, :]
            mask_train_sup = Variable(sup_index['mask'][tio.DATA].squeeze(1).long().cuda())

            pred_train_sup_2D = model1(img_train_sup_2D)
            pred_train_sup_2D_re = torch.zeros_like(pred_train_sup_2D)
            for m, n in enumerate(idx):
                pred_train_sup_2D_re[n, :] = pred_train_sup_2D[m, :]
            pred_train_sup_2D_re = pred_train_sup_2D_re.reshape(B, D, cfg['NUM_CLASSES'], H, W).permute(0, 2, 3, 4, 1)

            pred_train_sup_3D = model2(img_train_sup_3D)
            torch.cuda.empty_cache()

            if count_iter % args.display_iter == 0:
                if i == 0:
                    score_list_train2D = pred_train_sup_2D_re
                    score_list_train3D = pred_train_sup_3D
                    mask_list_train = mask_train_sup
                # else:
                elif 0 < i <= num_batches['train_sup'] / 64:
                    score_list_train2D = torch.cat((score_list_train2D, pred_train_sup_2D_re), dim=0)
                    score_list_train3D = torch.cat((score_list_train3D, pred_train_sup_3D), dim=0)
                    mask_list_train = torch.cat((mask_list_train, mask_train_sup), dim=0)

            loss_train_sup_2D = criterion(pred_train_sup_2D_re, mask_train_sup)
            loss_train_sup_3D = criterion(pred_train_sup_3D, mask_train_sup)
            loss_train_sup = loss_train_sup_2D + loss_train_sup_3D
            loss_train_sup.backward()

            optimizer1.step()
            optimizer2.step()
            torch.cuda.empty_cache()

            loss_train = loss_train_unsup + loss_train_sup
            train_loss_unsup += loss_train_unsup.item()
            train_loss_sup_2D += loss_train_sup_2D.item()
            train_loss_sup_3D += loss_train_sup_3D.item()
            train_loss += loss_train.item()

        scheduler_warmup1.step()
        scheduler_warmup2.step()
        torch.cuda.empty_cache()

        if count_iter % args.display_iter == 0:

            score_gather_list_train2D = [torch.zeros_like(score_list_train2D) for _ in range(ngpus_per_node)]
            torch.distributed.all_gather(score_gather_list_train2D, score_list_train2D)
            score_list_train2D = torch.cat(score_gather_list_train2D, dim=0)

            score_gather_list_train3D = [torch.zeros_like(score_list_train3D) for _ in range(ngpus_per_node)]
            torch.distributed.all_gather(score_gather_list_train3D, score_list_train3D)
            score_list_train3D = torch.cat(score_gather_list_train3D, dim=0)

            mask_gather_list_train = [torch.zeros_like(mask_list_train) for _ in range(ngpus_per_node)]
            torch.distributed.all_gather(mask_gather_list_train, mask_list_train)
            mask_list_train = torch.cat(mask_gather_list_train, dim=0)

            if rank == args.rank_index:
                torch.cuda.empty_cache()
                print('=' * print_num)
                print('| Epoch {}/{}'.format(epoch + 1, args.num_epochs).ljust(print_num_minus, ' '), '|')
                train_epoch_loss_sup_2D, train_epoch_loss_sup_3D, train_epoch_loss_unsup, train_epoch_loss = print_train_loss_SPC(train_loss_sup_2D, train_loss_sup_3D, train_loss_unsup, train_loss, num_batches, print_num, print_num_half)
                train_eval_list_2D, train_eval_list_3D, train_m_jc_2D, train_m_jc_3D = print_train_eval_SPC(cfg['NUM_CLASSES'], score_list_train2D, score_list_train3D, mask_list_train, print_num_half)
                torch.cuda.empty_cache()

            with torch.no_grad():
                model1.eval()
                model2.eval()

                for i, data in enumerate(dataloaders['val']):

                    # if 0 <= i <= num_batches['val']:
                    img_val_3D = Variable(data['image'][tio.DATA].cuda().cuda())
                    B, C, H, W, D =img_val_3D.shape
                    img_val_2D = img_val_3D.permute(0, 4, 1, 2, 3).reshape(B * D, C, H, W)
                    idx = torch.randperm(img_val_2D.shape[0])
                    img_val_2D = img_val_2D[idx, :]
                    mask_val = Variable(data['mask'][tio.DATA].squeeze(1).long().cuda())

                    optimizer1.zero_grad()
                    optimizer2.zero_grad()

                    pred_val_2D = model1(img_val_2D)
                    pred_val_2D_re = torch.zeros_like(pred_val_2D)
                    for m, n in enumerate(idx):
                        pred_val_2D_re[n, :] = pred_val_2D[m, :]
                    pred_val_2D_re = pred_val_2D_re.reshape(B, D, cfg['NUM_CLASSES'], H, W).permute(0, 2, 3, 4, 1)

                    pred_val_3D = model2(img_val_3D)
                    torch.cuda.empty_cache()

                    if i == 0:
                        score_list_val_2D = pred_val_2D_re
                        score_list_val_3D = pred_val_3D
                        mask_list_val = mask_val
                    else:
                        score_list_val_2D = torch.cat((score_list_val_2D, pred_val_2D_re), dim=0)
                        score_list_val_3D = torch.cat((score_list_val_3D, pred_val_3D), dim=0)
                        mask_list_val = torch.cat((mask_list_val, mask_val), dim=0)

                    loss_val_sup_2D = criterion(pred_val_2D_re, mask_val)
                    loss_val_sup_3D = criterion(pred_val_3D, mask_val)

                    val_loss_sup_2D += loss_val_sup_2D.item()
                    val_loss_sup_3D += loss_val_sup_3D.item()

                torch.cuda.empty_cache()
                score_gather_list_val_2D = [torch.zeros_like(score_list_val_2D) for _ in range(ngpus_per_node)]
                torch.distributed.all_gather(score_gather_list_val_2D, score_list_val_2D)
                score_list_val_2D = torch.cat(score_gather_list_val_2D, dim=0)

                score_gather_list_val_3D = [torch.zeros_like(score_list_val_3D) for _ in range(ngpus_per_node)]
                torch.distributed.all_gather(score_gather_list_val_3D, score_list_val_3D)
                score_list_val_3D = torch.cat(score_gather_list_val_3D, dim=0)

                mask_gather_list_val = [torch.zeros_like(mask_list_val) for _ in range(ngpus_per_node)]
                torch.distributed.all_gather(mask_gather_list_val, mask_list_val)
                mask_list_val = torch.cat(mask_gather_list_val, dim=0)
                torch.cuda.empty_cache()

                if rank == args.rank_index:
                    val_epoch_loss_2D, val_epoch_loss_3D = print_val_loss_SPC(val_loss_sup_2D, val_loss_sup_3D, num_batches, print_num, print_num_half)
                    val_eval_list_2D, val_eval_list_3D, val_m_jc_2D, val_m_jc_3D = print_val_eval(cfg['NUM_CLASSES'], score_list_val_2D, score_list_val_3D, mask_list_val, print_num_half)
                    best_val_eval_list, best_model, best_result = save_val_best_3d(best_model, best_val_eval_list, best_result, model1, model2, val_eval_list_2D, val_eval_list_3D, path_trained_models)
                    torch.cuda.empty_cache()

                    if args.vis:
                        visualization_SPC(visdom, epoch + 1, train_epoch_loss, train_epoch_loss_sup_2D, train_epoch_loss_sup_3D, train_epoch_loss_unsup, train_m_jc_2D, train_m_jc_3D, val_epoch_loss_2D, val_epoch_loss_3D, val_m_jc_2D, val_m_jc_3D)

                    print('-' * print_num)
                    print('| Epoch Time: {:.4f}s'.format((time.time() - begin_time) / args.display_iter).ljust(print_num_minus, ' '), '|')
            torch.cuda.empty_cache()
        torch.cuda.empty_cache()

    if rank == args.rank_index:
        time_elapsed = time.time() - since
        m, s = divmod(time_elapsed, 60)
        h, m = divmod(m, 60)

        print('=' * print_num)
        print('| Training Completed In {:.0f}h {:.0f}mins {:.0f}s'.format(h, m, s).ljust(print_num_minus, ' '), '|')
        print('-' * print_num)
        print_best(cfg['NUM_CLASSES'], best_val_eval_list, best_model, best_result, path_trained_models, print_num_minus)
        print('=' * print_num)