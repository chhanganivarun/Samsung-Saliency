from torchsummary import summary
import argparse
import glob
import os
import torch
import sys
import time
import torch.nn as nn
import pickle
from torch.autograd import Variable
from torchvision import transforms, utils
from PIL import Image
from torch.utils.data import DataLoader
import numpy as np
import torch.nn.init as init
import torch.nn.functional as F
from dataloader import *
from loss import *
import cv2
from model import *
from utils import *
from tsm.ops.models import TSN

parser = argparse.ArgumentParser()
parser.add_argument('--no_epochs', default=100, type=int)
parser.add_argument('--lr', default=1e-4, type=float)
parser.add_argument('--kldiv', default=True, type=bool)
parser.add_argument('--cc', default=False, type=bool)
parser.add_argument('--nss', default=False, type=bool)
parser.add_argument('--sim', default=False, type=bool)
parser.add_argument('--nss_emlnet', default=False, type=bool)
parser.add_argument('--nss_norm', default=False, type=bool)
parser.add_argument('--l1', default=False, type=bool)
parser.add_argument('--lr_sched', default=False, type=bool)
parser.add_argument('--optim', default="Adam", type=str)

parser.add_argument('--kldiv_coeff', default=1.0, type=float)
parser.add_argument('--step_size', default=5, type=int)
parser.add_argument('--cc_coeff', default=-1.0, type=float)
parser.add_argument('--sim_coeff', default=-1.0, type=float)
parser.add_argument('--nss_coeff', default=1.0, type=float)
parser.add_argument('--nss_emlnet_coeff', default=1.0, type=float)
parser.add_argument('--nss_norm_coeff', default=1.0, type=float)
parser.add_argument('--l1_coeff', default=1.0, type=float)

parser.add_argument('--batch_size', default=4, type=int)
parser.add_argument('--log_interval', default=5, type=int)
parser.add_argument('--no_workers', default=4, type=int)
parser.add_argument('--model_val_path',
                    default="enet_transformer.pt", type=str)
parser.add_argument('--img_backbone', default='s3d',
                    choices=['s3d', 'squeezenet', 'tsm_resnet50', 'tsm_mobilenetv2', 'combo'], type=str)
parser.add_argument('--clip_size', default=32, type=int)
parser.add_argument('--nhead', default=4, type=int)
parser.add_argument('--num_encoder_layers', default=3, type=int)
parser.add_argument('--num_decoder_layers', default=3, type=int)
parser.add_argument('--transformer_in_channel', default=32, type=int)
parser.add_argument('--train_path_data',
                    default="/ssd_scratch/cvit/varunc/DHF1K/annotation", type=str)
parser.add_argument('--val_path_data',
                    default="/ssd_scratch/cvit/varunc/DHF1K/val", type=str)
parser.add_argument('--decoder_upsample', default=1, type=int)
parser.add_argument('--frame_no', default="last", type=str)
parser.add_argument('--load_weight', default="None", type=str)
parser.add_argument('--num_hier', default=3, type=int)
parser.add_argument('--dataset', default="DHF1KDataset", type=str)
parser.add_argument('--alternate', default=1, type=int)
parser.add_argument('--spatial_dim', default=-1, type=int)
parser.add_argument('--split', default=-1, type=int)
parser.add_argument('--use_sound', default=False, type=bool)
parser.add_argument('--use_transformer', default=False, type=bool)
parser.add_argument('--use_vox', default=False, type=bool)


parser.add_argument('--shift', default=False,
                    action="store_true", help='use shift for models')
parser.add_argument('--shift_div', default=8, type=int,
                    help='number of div for shift (default: 8)')
parser.add_argument('--shift_place', default='blockres',
                    type=str, help='place for shift (default: stageres)')

args = parser.parse_args()
print(args)

if args.img_backbone == 's3d':
    file_weight = './S3D_kinetics400.pt'
elif args.img_backbone == 'squeezenet':
    file_weight = './jester_squeezenet_RGB_16_best.pth'
elif args.img_backbone == 'tsm_resnet50':
    file_weight = './TSM_kinetics_RGB_resnet50_shift8_blockres_avg_segment8_e100_dense.pth'
elif args.img_backbone == 'tsm_mobilenetv2':
    file_weight = './TSM_kinetics_RGB_mobilenetv2_shift8_blockres_avg_segment8_e100_dense.pth'
else:
    raise NotImplementedError(
        'Backbone {backbone} is not defined'.format(backbone=args.img_backbone))

if 'tsm' in args.img_backbone:
    from tsm.ops.models import TSN
    model = TSN(base_model=args.img_backbone.split('_')[
                1], shift_div=args.shift_div, is_shift=args.shift, shift_place=args.shift_place)
else:
    if args.use_sound:
        model = VideoAudioSaliencyModel(
            img_backbone=args.img_backbone,
            transformer_in_channel=args.transformer_in_channel,
            nhead=args.nhead,
            use_transformer=args.use_transformer,
            num_encoder_layers=args.num_encoder_layers,
            use_upsample=bool(args.decoder_upsample),
            num_hier=args.num_hier,
            num_clips=args.clip_size
        )
    else:
        model = VideoSaliencyModel(
            img_backbone=args.img_backbone,
            use_upsample=bool(args.decoder_upsample),
            num_hier=args.num_hier,
            num_clips=args.clip_size
        )

np.random.seed(0)
torch.manual_seed(0)

for (name, param) in model.named_parameters():
    if param.requires_grad:
        print(name, param.size())

if args.dataset == "DHF1KDataset":
    train_dataset = DHF1KDataset(
        args.train_path_data, args.clip_size, mode="train", alternate=args.alternate)
    val_dataset = DHF1KDataset(
        args.val_path_data, args.clip_size, mode="val", alternate=args.alternate)

elif args.dataset == "SoundDataset":
    train_dataset_diem = SoundDatasetLoader(
        args.clip_size, mode="train", dataset_name='DIEM', split=args.split, use_sound=args.use_sound, use_vox=args.use_vox)
    val_dataset_diem = SoundDatasetLoader(
        args.clip_size, mode="test", dataset_name='DIEM', split=args.split, use_sound=args.use_sound, use_vox=args.use_vox)

    train_dataset_coutrout1 = SoundDatasetLoader(
        args.clip_size, mode="train", dataset_name='Coutrot_db1', split=args.split, use_sound=args.use_sound, use_vox=args.use_vox)
    val_dataset_coutrout1 = SoundDatasetLoader(
        args.clip_size, mode="test", dataset_name='Coutrot_db1', split=args.split, use_sound=args.use_sound, use_vox=args.use_vox)

    train_dataset_coutrout2 = SoundDatasetLoader(
        args.clip_size, mode="train", dataset_name='Coutrot_db2', split=args.split, use_sound=args.use_sound, use_vox=args.use_vox)
    val_dataset_coutrout2 = SoundDatasetLoader(
        args.clip_size, mode="test", dataset_name='Coutrot_db2', split=args.split, use_sound=args.use_sound, use_vox=args.use_vox)

    train_dataset_avad = SoundDatasetLoader(
        args.clip_size, mode="train", dataset_name='AVAD', split=args.split, use_sound=args.use_sound, use_vox=args.use_vox)
    val_dataset_avad = SoundDatasetLoader(
        args.clip_size, mode="test", dataset_name='AVAD', split=args.split, use_sound=args.use_sound, use_vox=args.use_vox)

    train_dataset_etmd = SoundDatasetLoader(
        args.clip_size, mode="train", dataset_name='ETMD_av', split=args.split, use_sound=args.use_sound, use_vox=args.use_vox)
    val_dataset_etmd = SoundDatasetLoader(
        args.clip_size, mode="test", dataset_name='ETMD_av', split=args.split, use_sound=args.use_sound, use_vox=args.use_vox)

    train_dataset_summe = SoundDatasetLoader(
        args.clip_size, mode="train", dataset_name='SumMe', split=args.split, use_sound=args.use_sound, use_vox=args.use_vox)
    val_dataset_summe = SoundDatasetLoader(
        args.clip_size, mode="test", dataset_name='SumMe', split=args.split, use_sound=args.use_sound, use_vox=args.use_vox)

    train_dataset = torch.utils.data.ConcatDataset([
        train_dataset_diem, train_dataset_coutrout1,
        train_dataset_coutrout2,
        train_dataset_avad, train_dataset_etmd,
        train_dataset_summe
    ])

    val_dataset = torch.utils.data.ConcatDataset([
        val_dataset_diem, val_dataset_coutrout1,
        val_dataset_coutrout2,
        val_dataset_avad, val_dataset_etmd,
        val_dataset_summe
    ])
else:
    train_dataset = Hollywood_UCFDataset(
        args.train_path_data, args.clip_size, mode="train")
    # print(len(train_dataset))
    val_dataset = Hollywood_UCFDataset(
        args.val_path_data, args.clip_size, mode="val")

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.no_workers)
val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=1, shuffle=False, num_workers=args.no_workers)

if args.img_backbone in ['squeezenet', 's3d']:
    if not (args.use_sound or args.use_vox):
        if os.path.isfile(file_weight):
            print('loading weight file')
            weight_dict = torch.load(file_weight)
            print(weight_dict.keys())
            if args.img_backbone == 'squeezenet':
                weight_dict = weight_dict['state_dict']
            model_dict = model.backbone.state_dict()
            for name, param in weight_dict.items():
                if 'module' in name:
                    name = '.'.join(name.split('.')[1:])
                if 'base.' in name:
                    bn = int(name.split('.')[1])
                    sn_list = [0, 5, 8, 14]
                    sn = sn_list[0]
                    if bn >= sn_list[1] and bn < sn_list[2]:
                        sn = sn_list[1]
                    elif bn >= sn_list[2] and bn < sn_list[3]:
                        sn = sn_list[2]
                    elif bn >= sn_list[3]:
                        sn = sn_list[3]
                    name = '.'.join(name.split('.')[2:])
                    name = 'base%d.%d.' % (sn_list.index(sn)+1, bn-sn)+name
                if name in model_dict:
                    if param.size() == model_dict[name].size():
                        model_dict[name].copy_(param)
                    else:
                        print(' size? ' + name, param.size(),
                              model_dict[name].size())
                else:
                    print(' name? ' + name)

            print(' loaded')
            model.backbone.load_state_dict(model_dict)
        else:
            print('weight file?')
else:
    if args.img_backbone.split('_')[1] in ['resnet50']:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.device_count() > 1 or True:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            model = nn.DataParallel(model)
        model.to(device)

    print(("=> fine-tuning from '{}'".format(file_weight)))
    sd = torch.load(file_weight)
    sd = sd['state_dict']
    model_dict = model.state_dict()
    replace_dict = []
    for k, v in sd.items():
        if k not in model_dict and k.replace('.net', '') in model_dict:
            print('=> Load after remove .net: ', k)
            replace_dict.append((k, k.replace('.net', '')))
    for k, v in model_dict.items():
        if k not in sd and k.replace('.net', '') in sd:
            print('=> Load after adding .net: ', k)
            replace_dict.append((k.replace('.net', ''), k))

    for k, k_new in replace_dict:
        sd[k_new] = sd.pop(k)
    keys1 = set(list(sd.keys()))
    keys2 = set(list(model_dict.keys()))
    set_diff = (keys1 - keys2) | (keys2 - keys1)
    print('#### Notice: keys that failed to load: {}'.format(set_diff))
    print('=> New dataset, do not load fc weights')
    sd = {k: v for k, v in sd.items() if ('fc' if args.img_backbone.split('_')[
        1] in ['resnet50'] else 'classifier') not in k}
    model_dict.update(sd)
    model.load_state_dict(model_dict)

    if args.img_backbone.split('_')[1] not in ['resnet50']:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.device_count() > 1 or True:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            model = nn.DataParallel(model)
        model.to(device)
    # import pdb
    # pdb.set_trace()

    print(str(summary(model, (32, 3, 224, 384))))


if args.load_weight != "None":
    print("Loading weights: ", args.load_weight)
    if args.use_sound or args.use_vox:
        model.module.load_state_dict(torch.load(args.load_weight))
    else:
        model.module.load_state_dict(torch.load(args.load_weight))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if args.img_backbone in ['s3d', 'squeezenet']:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    model.to(device)

    summary(model, (3, 32, 224, 384))


params = list(filter(lambda p: p.requires_grad, model.parameters()))
optimizer = torch.optim.Adam(params, lr=args.lr)

print(device)


def train(model, optimizer, loader, epoch, device, args):
    model.train()
    tic = time.time()

    total_loss = AverageMeter()
    cur_loss = AverageMeter()

    for idx, sample in enumerate(loader):
        img_clips = sample[0]
        gt_sal = sample[1]
        if args.use_sound or args.use_vox:
            audio_feature = sample[2].to(device)
        img_clips = img_clips.to(device)
        if args.img_backbone in ['s3d', 'squeezenet', 'tsm_resnet50', 'tsm_mobilenetv2']:
            img_clips = img_clips.permute((0, 2, 1, 3, 4))
        gt_sal = gt_sal.to(device)

        optimizer.zero_grad()
        if args.use_sound or args.use_vox:
            pred_sal = model(img_clips, audio_feature)
        else:
            pred_sal = model(img_clips)
        # print(pred_sal.size(), gt_sal.size())
        assert pred_sal.size() == gt_sal.size()

        loss = loss_func(pred_sal, gt_sal, args)
        loss.backward()
        optimizer.step()
        total_loss.update(loss.item())
        cur_loss.update(loss.item())

        if idx % args.log_interval == (args.log_interval-1):
            print('[{:2d}, {:5d}] avg_loss : {:.5f}, time:{:3f} minutes'.format(
                epoch, idx, cur_loss.avg, (time.time()-tic)/60))
            cur_loss.reset()
            sys.stdout.flush()

    print('[{:2d}, train] avg_loss : {:.5f}'.format(epoch, total_loss.avg))
    sys.stdout.flush()

    return total_loss.avg


def validate(model, loader, epoch, device, args):
    model.eval()
    tic = time.time()
    total_loss = AverageMeter()
    total_cc_loss = AverageMeter()
    total_sim_loss = AverageMeter()
    tic = time.time()
    for idx, sample in enumerate(loader):
        img_clips = sample[0]
        gt_sal = sample[1]
        if args.use_sound or args.use_vox:
            audio_feature = sample[2].to(device)
        img_clips = img_clips.to(device)
        img_clips = img_clips.permute((0, 2, 1, 3, 4))

        if args.use_sound or args.use_vox:
            pred_sal = model(img_clips, audio_feature)
        else:
            pred_sal = model(img_clips)

        gt_sal = gt_sal.squeeze(0).numpy()

        pred_sal = pred_sal.cpu().squeeze(0).numpy()
        pred_sal = cv2.resize(pred_sal, (gt_sal.shape[1], gt_sal.shape[0]))
        pred_sal = blur(pred_sal).unsqueeze(0).cuda()

        gt_sal = torch.FloatTensor(gt_sal).unsqueeze(0).cuda()

        assert pred_sal.size() == gt_sal.size()

        loss = loss_func(pred_sal, gt_sal, args)
        cc_loss = cc(pred_sal, gt_sal)
        sim_loss = similarity(pred_sal, gt_sal)

        total_loss.update(loss.item())
        total_cc_loss.update(cc_loss.item())
        total_sim_loss.update(sim_loss.item())

    print('[{:2d}, val] avg_loss : {:.5f} cc_loss : {:.5f} sim_loss : {:.5f}, time : {:3f}'.format(
        epoch, total_loss.avg, total_cc_loss.avg, total_sim_loss.avg, (time.time()-tic)/60))
    sys.stdout.flush()

    return total_loss.avg


best_model = None
for epoch in range(0, args.no_epochs):
    loss = train(model, optimizer, train_loader, epoch, device, args)

    with torch.no_grad():
        val_loss = validate(model, val_loader, epoch, device, args)
        if epoch == 0:
            val_loss = np.inf
            best_loss = val_loss
        if val_loss <= best_loss:
            best_loss = val_loss
            best_model = model
            print('[{:2d},  save, {}]'.format(epoch, args.model_val_path))
            if torch.cuda.device_count() > 1:
                torch.save(model.module.state_dict(), args.model_val_path)
            else:
                torch.save(model.state_dict(), args.model_val_path)
    print()

    if args.lr_sched:
        scheduler.step()
